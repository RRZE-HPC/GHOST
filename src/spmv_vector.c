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
#include <stdio.h>
#include <string.h>

#ifdef GHOST_HAVE_OPENMP
#include <omp.h>
#endif

// if called with mat->context==NULL: clean up variables
ghost_error_t ghost_spmv_vectormode(ghost_densemat_t* res, ghost_sparsemat_t* mat, ghost_densemat_t* invec, ghost_spmv_flags_t flags,va_list argp)
{
#ifndef GHOST_HAVE_MPI
    UNUSED(mat->context);
    UNUSED(res);
    UNUSED(mat);
    UNUSED(invec);
    UNUSED(flags);
    UNUSED(argp);
    ERROR_LOG("Cannot execute this spMV solver without MPI");
    return GHOST_ERR_UNKNOWN;
#else
    if (mat->context == NULL) {
        ERROR_LOG("The mat->context is NULL");
        return GHOST_ERR_INVALID_ARG;
    }

    int i, from_PE, to_PE;
    int msgcount;
    ghost_idx_t c;
    char *work = NULL;
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_nnz_t max_dues;
    ghost_nnz_t max_wishes;
    int nprocs;
    int me; 
    
    GHOST_CALL_RETURN(ghost_rank(&me, mat->context->mpicomm));
    GHOST_CALL_RETURN(ghost_nrank(&nprocs, mat->context->mpicomm));
    
    MPI_Request request[2*nprocs];
    MPI_Status  status[2*nprocs];
    

    max_dues = 0;
    max_wishes = 0;
    for (i=0;i<nprocs;i++) {
        if (mat->context->dues[i]>max_dues) {
            max_dues = mat->context->dues[i];
        }
        if (mat->context->wishes[i]>max_wishes) {
            max_wishes = mat->context->wishes[i];
        }
    }
    char **tmprecv;
    char *tmprecv_mem;
    ghost_malloc((void **)&tmprecv_mem,invec->traits.ncols*invec->elSize*max_wishes*nprocs);
    ghost_malloc((void **)&tmprecv,nprocs*sizeof(char *));
    
    for (from_PE=0; from_PE<nprocs; from_PE++){
        tmprecv[from_PE] = &tmprecv_mem[invec->traits.ncols*invec->elSize*max_wishes];
    }
    
    GHOST_CALL_RETURN(ghost_malloc((void **)&work,invec->traits.ncols*max_dues*nprocs * invec->elSize));

    GHOST_INSTR_START(spmv_vector_comm);
    invec->downloadNonHalo(invec);


#ifdef __INTEL_COMPILER
    //kmp_set_blocktime(1);
#endif

    msgcount=0;
    for (i=0;i<2*nprocs;i++) { 
        request[i] = MPI_REQUEST_NULL;
    }

    char *recv;

    for (from_PE=0; from_PE<nprocs; from_PE++){
        if (invec->traits.storage == GHOST_DENSEMAT_ROWMAJOR) {
            recv = invec->val[mat->context->hput_pos[from_PE]];
        } else {
            recv = tmprecv[from_PE];
        }
#ifdef GHOST_HAVE_INSTR_DATA
            INFO_LOG("from %d: %zu bytes",from_PE,mat->context->wishes[from_PE]*invec->elSize);
#endif
        if (mat->context->wishes[from_PE]>0){
            MPI_CALL_GOTO(MPI_Irecv(recv, invec->traits.ncols*invec->elSize*mat->context->wishes[from_PE],MPI_CHAR, from_PE, from_PE, mat->context->mpicomm,&request[msgcount]),err,ret);
            msgcount++;
#ifdef GHOST_HAVE_INSTR_DATA
            recvBytes += mat->context->wishes[from_PE]*invec->elSize*invec->traits.ncols;
            recvMsgs++;
#endif
        }
    }
        

    GHOST_INSTR_START(spmv_vector_copybuffers)
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
#pragma omp for 
                for (i=0; i<mat->context->dues[to_PE]; i++){
                    for (c=0; c<invec->traits.ncols; c++) {
                        memcpy(work + (to_PE*max_dues+i)*invec->elSize*invec->traits.ncols + c*invec->elSize,&invec->val[c][mat->permutation->perm[mat->context->duelist[to_PE][i]]*invec->elSize],invec->elSize);
                    }
                }
            }
        } else {
#pragma omp parallel private(to_PE,i,c)
            for (to_PE=0 ; to_PE<nprocs ; to_PE++){
#pragma omp for 
                for (i=0; i<mat->context->dues[to_PE]; i++){
                    for (c=0; c<invec->traits.ncols; c++) {
                        memcpy(work + (to_PE*max_dues+i)*invec->elSize*invec->traits.ncols + c*invec->elSize,&invec->val[c][mat->context->duelist[to_PE][i]*invec->elSize],invec->elSize);
                    }
                }
            }
        }
    }

    GHOST_INSTR_STOP(spmv_vector_copybuffers)


    for (to_PE=0 ; to_PE<nprocs ; to_PE++){
        if (mat->context->dues[to_PE]>0){
            MPI_CALL_GOTO(MPI_Isend( work + to_PE*max_dues*invec->elSize*invec->traits.ncols, mat->context->dues[to_PE]*invec->elSize*invec->traits.ncols, MPI_CHAR, to_PE, me, mat->context->mpicomm, &request[msgcount]),err,ret);
            msgcount++;
        }
    }

    GHOST_INSTR_START(spmv_vector_waitall)
    MPI_CALL_GOTO(MPI_Waitall(msgcount, request, status),err,ret);
    GHOST_INSTR_STOP(spmv_vector_waitall)
    if (invec->traits.storage == GHOST_DENSEMAT_COLMAJOR) {
        for (from_PE=0; from_PE<nprocs; from_PE++){
        WARNING_LOG("i 0..%d c 0..%d",mat->context->wishes[from_PE],invec->traits.ncols);
            for (i=0; i<mat->context->wishes[from_PE]; i++){
                for (c=0; c<invec->traits.ncols; c++) {
                    memcpy(&invec->val[c][(mat->context->hput_pos[from_PE]+i)*invec->elSize],&tmprecv[from_PE][(i*invec->traits.ncols+c)*invec->elSize],invec->elSize);
                    if (i==0 && c==0) {
                        WARNING_LOG("foo %f",((double *)tmprecv[0])[0]);
                    }
                }
            }
        }
        INFO_LOG("recvd[0,1] = %f %f",((double *)invec->val[0])[mat->context->hput_pos[0]],((double *)invec->val[1])[mat->context->hput_pos[0]]);
    } else {
        INFO_LOG("recvd[0,1] = %f %f",((double *)invec->val[mat->context->hput_pos[0]])[0],((double *)invec->val[mat->context->hput_pos[0]])[1]);
    }



    GHOST_CALL_GOTO(invec->uploadHalo(invec),err,ret);
    GHOST_INSTR_STOP(spmv_vector_comm);

    GHOST_INSTR_START(spmv_vector_comp);
    GHOST_CALL_GOTO(mat->spmv(mat,res,invec,flags,argp),err,ret);    
    GHOST_INSTR_STOP(spmv_vector_comp);

    goto out;
err:

out:
    free(work);
    free(tmprecv_mem);
    free(tmprecv);

    return ret;
#endif
}

