#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/locality.h"
#include "ghost/densemat.h"
#include "ghost/util.h"
#include "ghost/instr.h"
#include "ghost/sparsemat.h"
#include "ghost/spmv_solvers.h"

#ifdef GHOST_HAVE_MPI
#include <mpi.h>
#endif
#include <sys/types.h>
#include <string.h>

#ifdef GHOST_HAVE_OPENMP
#include <omp.h>
#endif

ghost_error_t ghost_spmv_goodfaith(ghost_densemat_t* res, ghost_sparsemat_t* mat, ghost_densemat_t* invec, ghost_spmv_flags_t flags, va_list argp)
{
#ifndef GHOST_HAVE_MPI
    UNUSED(res);
    UNUSED(mat);
    UNUSED(invec);
    UNUSED(flags);
    UNUSED(argp);
    ERROR_LOG("Cannot execute this spMV solver without MPI");
    return GHOST_ERR_UNKNOWN;
#else
    ghost_nnz_t max_dues;
    char *work = NULL;
    ghost_error_t ret = GHOST_SUCCESS;
    int nprocs;

    int localopts = flags|GHOST_SPMV_LOCAL;
    int remoteopts = flags|GHOST_SPMV_REMOTE;

    va_list remote_argp;
    va_copy(remote_argp,argp);
    int me; 
    int i, from_PE, to_PE;
    int msgcount;
    ghost_idx_t c;


    GHOST_CALL_RETURN(ghost_rank(&me, mat->context->mpicomm));
    GHOST_CALL_RETURN(ghost_nrank(&nprocs, mat->context->mpicomm));
    MPI_Request request[invec->traits.ncols*2*nprocs];
    MPI_Status  status[invec->traits.ncols*2*nprocs];

    max_dues = 0;
    for (i=0;i<nprocs;i++)
        if (mat->context->dues[i]>max_dues) 
            max_dues = mat->context->dues[i];

    GHOST_CALL_RETURN(ghost_malloc((void **)&work,invec->traits.ncols*max_dues*nprocs * invec->elSize));

#ifdef __INTEL_COMPILER
 //   kmp_set_blocktime(1);
#endif

    invec->downloadNonHalo(invec);

    msgcount = 0;
    for (i=0;i<invec->traits.ncols*2*nprocs;i++) {
        request[i] = MPI_REQUEST_NULL;
    }
    if (invec->traits.storage == GHOST_DENSEMAT_ROWMAJOR) {
        for (from_PE=0; from_PE<nprocs; from_PE++){
#ifdef GHOST_HAVE_INSTR_DATA
                INFO_LOG("from %d: %zu bytes",from_PE,mat->context->wishes[from_PE]*invec->elSize);
#endif
            if (mat->context->wishes[from_PE]>0){
                MPI_CALL_GOTO(MPI_Irecv(invec->val[mat->context->hput_pos[from_PE]], invec->traits.ncols*mat->context->wishes[from_PE]*invec->elSize,MPI_CHAR, from_PE, from_PE, mat->context->mpicomm,&request[msgcount]),err,ret);
                msgcount++;
#ifdef GHOST_HAVE_INSTR_DATA
                recvBytes += mat->context->wishes[from_PE]*invec->elSize;
                recvMsgs++;
#endif
            }
        }
        
    } else if (invec->traits.storage == GHOST_DENSEMAT_COLMAJOR) {
        for (from_PE=0; from_PE<nprocs; from_PE++){
#ifdef GHOST_HAVE_INSTR_DATA
                INFO_LOG("from %d: %zu bytes",from_PE,mat->context->wishes[from_PE]*invec->elSize);
#endif
            if (mat->context->wishes[from_PE]>0){
                for (c=0; c<invec->traits.ncols; c++) {
                    MPI_CALL_GOTO(MPI_Irecv(&invec->val[c][mat->context->hput_pos[from_PE]*invec->elSize], mat->context->wishes[from_PE]*invec->elSize,MPI_CHAR, from_PE, from_PE, mat->context->mpicomm,&request[msgcount]),err,ret);
                    msgcount++;
#ifdef GHOST_HAVE_INSTR_DATA
                    recvBytes += mat->context->wishes[from_PE]*invec->elSize;
                    recvMsgs++;
#endif
                }
            }
        }
        
    }

    if (invec->traits.storage == GHOST_DENSEMAT_ROWMAJOR) {
        if ((mat->traits->flags & GHOST_SPARSEMAT_PERMUTE) && 
                (mat->permutation->scope == GHOST_PERMUTATION_LOCAL)) {
#pragma omp parallel private(to_PE,i,c)
            for (to_PE=0 ; to_PE<nprocs ; to_PE++){
#pragma omp for 
                for (i=0; i<mat->context->dues[to_PE]; i++){
                    memcpy(work + (to_PE*max_dues+i)*invec->elSize*invec->traits.ncols,invec->val[mat->permutation->perm[mat->context->duelist[to_PE][i]]],invec->elSize*invec->traits.ncols);
                }
            }
        } else {
#pragma omp parallel private(to_PE,i)
            for (to_PE=0 ; to_PE<nprocs ; to_PE++){
#pragma omp for 
                for (i=0; i<mat->context->dues[to_PE]; i++){
                    memcpy(work + (to_PE*max_dues+i)*invec->elSize*invec->traits.ncols,invec->val[mat->context->duelist[to_PE][i]],invec->elSize*invec->traits.ncols);
                }
            }
        }
    } else if (invec->traits.storage == GHOST_DENSEMAT_COLMAJOR) {
        if ((mat->traits->flags & GHOST_SPARSEMAT_PERMUTE) && 
                (mat->permutation->scope == GHOST_PERMUTATION_LOCAL)) {
#pragma omp parallel private(to_PE,i,c)
            for (to_PE=0 ; to_PE<nprocs ; to_PE++){
                for (c=0; c<invec->traits.ncols; c++) {
#pragma omp for 
                    for (i=0; i<mat->context->dues[to_PE]; i++){
                        memcpy(work + c*nprocs*max_dues*invec->elSize + (to_PE*max_dues+i)*invec->elSize,&invec->val[c][mat->permutation->perm[mat->context->duelist[to_PE][i]]*invec->elSize],invec->elSize);
                    }
                }
            }
        } else {
#pragma omp parallel private(to_PE,i,c)
            for (to_PE=0 ; to_PE<nprocs ; to_PE++){
                for (c=0; c<invec->traits.ncols; c++) {
#pragma omp for 
                    for (i=0; i<mat->context->dues[to_PE]; i++){
                        memcpy(work + c*nprocs*max_dues*invec->elSize + (to_PE*max_dues+i)*invec->elSize,&invec->val[c][mat->context->duelist[to_PE][i]*invec->elSize],invec->elSize);
                    }
                }
            }
        }
    }
    
    if (invec->traits.storage == GHOST_DENSEMAT_ROWMAJOR) {
        for (to_PE=0 ; to_PE<nprocs ; to_PE++){
            if (mat->context->dues[to_PE]>0){
                MPI_CALL_GOTO(MPI_Isend( work + to_PE*max_dues*invec->elSize*invec->traits.ncols, mat->context->dues[to_PE]*invec->elSize*invec->traits.ncols, MPI_CHAR, to_PE, me, mat->context->mpicomm, &request[msgcount]),err,ret);
                msgcount++;
            }
        }
    } else if (invec->traits.storage == GHOST_DENSEMAT_COLMAJOR) {
        for (to_PE=0 ; to_PE<nprocs ; to_PE++){
            if (mat->context->dues[to_PE]>0){
                for (c=0; c<invec->traits.ncols; c++) {
                    MPI_CALL_GOTO(MPI_Isend( work + c*nprocs*max_dues*invec->elSize + to_PE*max_dues*invec->elSize, mat->context->dues[to_PE]*invec->elSize, MPI_CHAR, to_PE, me, mat->context->mpicomm, &request[msgcount]),err,ret);
                    msgcount++;
                }
            }
        }
    }
    
    GHOST_INSTR_START(spmv_overlap_local);
    GHOST_CALL_GOTO(mat->localPart->spmv(mat->localPart,res,invec,localopts,argp),err,ret);
    GHOST_INSTR_STOP(spmv_overlap_local);

    GHOST_INSTR_START(spmv_overlap_waitall);
    MPI_CALL_GOTO(MPI_Waitall(msgcount, request, status),err,ret);
    GHOST_INSTR_STOP(spmv_overlap_waitall);

    GHOST_CALL_GOTO(invec->uploadHalo(invec),err,ret);

    GHOST_INSTR_START(spmv_overlap_remote);
    GHOST_CALL_GOTO(mat->remotePart->spmv(mat->remotePart,res,invec,remoteopts,remote_argp),err,ret);
    GHOST_INSTR_STOP(spmv_overlap_remote);

    goto out;
err:

out:
    free(work);

    return ret;
#endif
}
