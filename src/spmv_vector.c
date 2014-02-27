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
    int nprocs;
    int me; 
    
    GHOST_CALL_RETURN(ghost_getRank(mat->context->mpicomm,&me));
    GHOST_CALL_RETURN(ghost_getNumberOfRanks(mat->context->mpicomm,&nprocs));
    
    MPI_Request request[invec->traits->ncols*2*nprocs];
    MPI_Status  status[invec->traits->ncols*2*nprocs];
    

    max_dues = 0;
    for (i=0;i<nprocs;i++) {
        if (mat->context->dues[i]>max_dues) {
            max_dues = mat->context->dues[i];
        }
    }
    
    GHOST_CALL_RETURN(ghost_malloc((void **)&work,invec->traits->ncols*max_dues*nprocs * invec->elSize));

    GHOST_INSTR_START(spmv_vector_comm);
    invec->downloadNonHalo(invec);


#ifdef __INTEL_COMPILER
    //kmp_set_blocktime(1);
#endif

    msgcount=0;
    for (i=0;i<invec->traits->ncols*2*nprocs;i++) { 
        request[i] = MPI_REQUEST_NULL;
    }

    for (from_PE=0; from_PE<nprocs; from_PE++){
        if (mat->context->wishes[from_PE]>0){
            for (c=0; c<invec->traits->ncols; c++) {
                MPI_CALL_GOTO(MPI_Irecv(VECVAL(invec,invec->val,c,mat->context->hput_pos[from_PE]), mat->context->wishes[from_PE]*invec->elSize,MPI_CHAR, from_PE, from_PE, mat->context->mpicomm,&request[msgcount]),err,ret);
                msgcount++;
            }
        }
    }

    GHOST_INSTR_START(spmv_vector_copybuffers)
    if ((mat->traits->flags & GHOST_SPARSEMAT_PERMUTE) && 
            (mat->permutation->scope == GHOST_PERMUTATION_LOCAL)) {
#pragma omp parallel private(to_PE,i,c)
        for (to_PE=0 ; to_PE<nprocs ; to_PE++){
            for (c=0; c<invec->traits->ncols; c++) {
#pragma omp for 
                for (i=0; i<mat->context->dues[to_PE]; i++){
                    memcpy(work + c*nprocs*max_dues*invec->elSize + (to_PE*max_dues+i)*invec->elSize,VECVAL(invec,invec->val,c,mat->permutation->perm[mat->context->duelist[to_PE][i]]),invec->elSize);
                }
            }
        }
    } else {
#pragma omp parallel private(to_PE,i,c)
        for (to_PE=0 ; to_PE<nprocs ; to_PE++){
            for (c=0; c<invec->traits->ncols; c++) {
#pragma omp for 
                for (i=0; i<mat->context->dues[to_PE]; i++){
                    memcpy(work + c*nprocs*max_dues*invec->elSize + (to_PE*max_dues+i)*invec->elSize,VECVAL(invec,invec->val,c,mat->context->duelist[to_PE][i]),invec->elSize);
                }
            }
        }
    }

    GHOST_INSTR_STOP(spmv_vector_copybuffers)


    for (to_PE=0 ; to_PE<nprocs ; to_PE++){
        if (mat->context->dues[to_PE]>0){
            for (c=0; c<invec->traits->ncols; c++) {
                MPI_CALL_GOTO(MPI_Isend( work + c*nprocs*max_dues*invec->elSize + to_PE*max_dues*invec->elSize, mat->context->dues[to_PE]*invec->elSize, MPI_CHAR, to_PE, me, mat->context->mpicomm, &request[msgcount]),err,ret);
                msgcount++;
            }
        }
    }

    GHOST_INSTR_START(spmv_vector_waitall)
    MPI_CALL_GOTO(MPI_Waitall(msgcount, request, status),err,ret);
    GHOST_INSTR_STOP(spmv_vector_waitall)

    GHOST_CALL_GOTO(invec->uploadHalo(invec),err,ret);
    GHOST_INSTR_STOP(spmv_vector_comm);

    GHOST_INSTR_START(spmv_vector_comp);
    GHOST_CALL_GOTO(mat->spmv(mat,res,invec,flags,argp),err,ret);    
    GHOST_INSTR_STOP(spmv_vector_comp);

    goto out;
err:

out:
    free(work);

    return ret;
#endif
}

