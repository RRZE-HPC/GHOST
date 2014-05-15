#include "ghost/spmv.h"
#include "ghost/instr.h"
#include "ghost/error.h"
#include "ghost/util.h"
#include "ghost/sparsemat.h"
#include "ghost/context.h"
#include "ghost/locality.h"
#include "ghost/densemat_cm.h"

#define DL_WHOLE

#ifdef GHOST_HAVE_MPI
#include <mpi.h>


static int msgcount;
static MPI_Request *request = NULL;
static MPI_Status  *status = NULL;
static char **tmprecv = NULL;
static char *tmprecv_mem = NULL;
static char *work = NULL;
static ghost_nnz_t max_dues;
#endif

ghost_error_t ghost_spmv_haloexchange_assemble(ghost_densemat_t *vec, ghost_permutation_t *permutation)
{
#ifdef GHOST_HAVE_MPI
    ghost_error_t ret = GHOST_SUCCESS;
    int nprocs;
    max_dues = 0;
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, vec->context->mpicomm),err,ret);
    int i, to_PE;
    ghost_idx_t c;

    if (nprocs == 1) {
        return GHOST_SUCCESS;
    }
    
    for (i=0;i<nprocs;i++) {
        if (vec->context->dues[i]>max_dues) {
            max_dues = vec->context->dues[i];
        }
    }
    
    GHOST_CALL_RETURN(ghost_malloc((void **)&work,vec->traits.ncols*max_dues*nprocs * vec->elSize));
    
    GHOST_INSTR_START(spmv_haloexchange_assemblebuffers)
#ifdef DL_WHOLE
    vec->downloadNonHalo(vec);
#endif
    if (vec->traits.storage == GHOST_DENSEMAT_ROWMAJOR) {
        if (permutation && permutation->scope == GHOST_PERMUTATION_LOCAL) {
#pragma omp parallel private(to_PE,i,c)
            for (to_PE=0 ; to_PE<nprocs ; to_PE++){
#pragma omp for 
                for (i=0; i<vec->context->dues[to_PE]; i++){
                    memcpy(work + (to_PE*max_dues+i)*vec->elSize*vec->traits.ncols,vec->val[permutation->perm[vec->context->duelist[to_PE][i]]],vec->elSize*vec->traits.ncols);
                }
            }
        } else {
#pragma omp parallel private(to_PE,i)
            for (to_PE=0 ; to_PE<nprocs ; to_PE++){
#pragma omp for 
                for (i=0; i<vec->context->dues[to_PE]; i++){
                    memcpy(work + (to_PE*max_dues+i)*vec->elSize*vec->traits.ncols,vec->val[vec->context->duelist[to_PE][i]],vec->elSize*vec->traits.ncols);
                }
            }
        }
    } else if (vec->traits.storage == GHOST_DENSEMAT_COLMAJOR) {
        if (permutation && permutation->scope == GHOST_PERMUTATION_LOCAL) {
#pragma omp parallel private(to_PE,i,c)
            for (to_PE=0 ; to_PE<nprocs ; to_PE++){
#pragma omp for 
                for (i=0; i<vec->context->dues[to_PE]; i++){
                    for (c=0; c<vec->traits.ncols; c++) {
#ifndef DL_WHOLE
                        if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
                            ghost_cu_download(work + (to_PE*max_dues+i)*vec->elSize*vec->traits.ncols + c*vec->elSize,CUVECVAL_CM(vec,vec->cu_val,c,permutation->perm[vec->context->duelist[to_PE][i]]),vec->elSize);
                        } else 
#endif
                        {
                            memcpy(work + (to_PE*max_dues+i)*vec->elSize*vec->traits.ncols + c*vec->elSize,&vec->val[c][permutation->perm[vec->context->duelist[to_PE][i]]*vec->elSize],vec->elSize);
                        }
                    }
                }
            }
        } else {
#pragma omp parallel private(to_PE,i,c)
            for (to_PE=0 ; to_PE<nprocs ; to_PE++){
#pragma omp for 
                for (i=0; i<vec->context->dues[to_PE]; i++){
                    for (c=0; c<vec->traits.ncols; c++) {
#ifndef DL_WHOLE
                        if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
                            ghost_cu_download(work + (to_PE*max_dues+i)*vec->elSize*vec->traits.ncols + c*vec->elSize,CUVECVAL_CM(vec,vec->cu_val,c,vec->context->duelist[to_PE][i]),vec->elSize);
                        } else 
#endif
                        {
                            memcpy(work + (to_PE*max_dues+i)*vec->elSize*vec->traits.ncols + c*vec->elSize,&vec->val[c][vec->context->duelist[to_PE][i]*vec->elSize],vec->elSize);
                        }
                    }
                }
            }
        }
    }
    GHOST_INSTR_STOP(spmv_haloexchange_assemblebuffers)
    goto out;
err:

out:
    return ret;
#else
    UNUSED(vec);
    UNUSED(permutation);
    return GHOST_ERR_NOT_IMPLEMENTED;
#endif
}

ghost_error_t ghost_spmv_haloexchange_initiate(ghost_densemat_t *vec, ghost_permutation_t *permutation, bool assembled)
{
#ifdef GHOST_HAVE_MPI
    int nprocs;
    int me; 
    int i, from_PE, to_PE;
    ghost_nnz_t max_wishes;
    ghost_error_t ret = GHOST_SUCCESS;
    
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, vec->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_rank(&me, vec->context->mpicomm),err,ret);
    
    msgcount = 0;
    GHOST_CALL_GOTO(ghost_malloc((void **)&request,sizeof(MPI_Request)*2*nprocs),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&status,sizeof(MPI_Status)*2*nprocs),err,ret);

    max_wishes = 0;
    for (i=0;i<nprocs;i++) {
        if (vec->context->dues[i]>max_dues) {
            max_dues = vec->context->dues[i];
        }
        if (vec->context->wishes[i]>max_wishes) {
            max_wishes = vec->context->wishes[i];
        }
    }
    ghost_malloc((void **)&tmprecv_mem,vec->traits.ncols*vec->elSize*max_wishes*nprocs);
    ghost_malloc((void **)&tmprecv,nprocs*sizeof(char *));
    
    for (from_PE=0; from_PE<nprocs; from_PE++){
        tmprecv[from_PE] = &tmprecv_mem[from_PE*vec->traits.ncols*vec->elSize*max_wishes];
    }

    

    msgcount = 0;
    for (i=0;i<2*nprocs;i++) {
        request[i] = MPI_REQUEST_NULL;
    }
    
    char *recv;

    for (from_PE=0; from_PE<nprocs; from_PE++){
        if (vec->context->wishes[from_PE]>0){
            if (vec->traits.storage == GHOST_DENSEMAT_ROWMAJOR) {
                recv = vec->val[vec->context->hput_pos[from_PE]];
            } else {
                recv = tmprecv[from_PE];
            }
#ifdef GHOST_HAVE_INSTR_DATA
            INFO_LOG("from %d: %zu bytes",from_PE,vec->context->wishes[from_PE]*vec->elSize);
#endif
            MPI_CALL_GOTO(MPI_Irecv(recv, vec->traits.ncols*vec->elSize*vec->context->wishes[from_PE],MPI_CHAR, from_PE, from_PE, vec->context->mpicomm,&request[msgcount]),err,ret);
            msgcount++;
#ifdef GHOST_HAVE_INSTR_DATA
            recvBytes += vec->context->wishes[from_PE]*vec->elSize*vec->traits.ncols;
            recvMsgs++;
#endif
        }
    }
   
    if (!assembled) { 
        GHOST_CALL_GOTO(ghost_spmv_haloexchange_assemble(vec,permutation),err,ret);
    }
    
    for (to_PE=0 ; to_PE<nprocs ; to_PE++){
        if (vec->context->dues[to_PE]>0){
            MPI_CALL_GOTO(MPI_Isend( work + to_PE*max_dues*vec->elSize*vec->traits.ncols, vec->context->dues[to_PE]*vec->elSize*vec->traits.ncols, MPI_CHAR, to_PE, me, vec->context->mpicomm, &request[msgcount]),err,ret);
            msgcount++;
        }
    ;}


    goto out;
err:

out:
    return ret;
#else
    UNUSED(vec);
    UNUSED(permutation);
    UNUSED(assembled);
    return GHOST_ERR_NOT_IMPLEMENTED;
#endif
} 


ghost_error_t ghost_spmv_haloexchange_finalize(ghost_densemat_t *vec)
{
#ifdef GHOST_HAVE_MPI
    ghost_error_t ret = GHOST_SUCCESS;
    int nprocs;
    int i, from_PE;
    ghost_idx_t c;
    
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, vec->context->mpicomm),err,ret);

    GHOST_INSTR_START(spmv_haloexchange_waitall);
    MPI_CALL_GOTO(MPI_Waitall(msgcount, request, status),err,ret);
    GHOST_INSTR_STOP(spmv_haloexchange_waitall);
    
    if (vec->traits.storage == GHOST_DENSEMAT_COLMAJOR) {
        for (from_PE=0; from_PE<nprocs; from_PE++){
            for (i=0; i<vec->context->wishes[from_PE]; i++){
                for (c=0; c<vec->traits.ncols; c++) {
                    memcpy(&vec->val[c][(vec->context->hput_pos[from_PE]+i)*vec->elSize],&tmprecv[from_PE][(i*vec->traits.ncols+c)*vec->elSize],vec->elSize);
                }
            }
        }
    }    
    GHOST_CALL_GOTO(vec->uploadHalo(vec),err,ret);
    
    free(work); work = NULL;
    free(tmprecv_mem); tmprecv_mem = NULL;
    free(tmprecv); tmprecv = NULL;
    free(request); request = NULL;
    free(status); status = NULL;

    goto out;
err:

out:
    return ret;

#else
    return GHOST_ERR_NOT_IMPLEMENTED;
#endif
}
