#include "ghost/spmv.h"
#include "ghost/instr.h"
#include "ghost/error.h"
#include "ghost/util.h"
#include "ghost/sparsemat.h"
#include "ghost/context.h"
#include "ghost/locality.h"
#include "ghost/densemat_cm.h"
#include "ghost/datatransfers.h"


// one of the three options below has to be commented out
#define CUDA_COMMUNICATION_ASSEMBLY_KERNEL // use a CUDA kernel for assembly (preferred!)
//#define CUDA_COMMUNICATION_ASSEMBLY_DL // download the CUDA vector and do the assembly on the host
//#define CUDA_COMMUNICATION_ASSEMBLY_MEMCPY // do single-element memcpys for assembly (probably very slow)

#ifdef GHOST_HAVE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef GHOST_HAVE_MPI
#include <mpi.h>


static int msgcount;
static MPI_Request *request = NULL;
static MPI_Status  *status = NULL;
static char **tmprecv = NULL;
static char *tmprecv_mem = NULL;
static char *work = NULL;
static ghost_lidx_t *dueptr = NULL;
static ghost_lidx_t *wishptr = NULL;
static ghost_lidx_t acc_dues = 0;
static ghost_lidx_t acc_wishes = 0;
#ifdef GHOST_HAVE_CUDA || !defined(CUDA_COMMUNICATION_ASSEMBLY_DL)
static void *cu_work;
#endif

#endif

ghost_error_t ghost_spmv_haloexchange_assemble(ghost_densemat_t *vec, ghost_permutation_t *permutation)
{
#ifdef GHOST_HAVE_MPI
    ghost_error_t ret = GHOST_SUCCESS;
    int nprocs;
    //max_dues = 0;
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, vec->context->mpicomm),err,ret);
    int i, to_PE;

    if (nprocs == 1) {
        return GHOST_SUCCESS;
    }
    GHOST_INSTR_START(spmv_haloexchange_assemblebuffers)

    GHOST_CALL_RETURN(ghost_malloc((void **)&dueptr,(nprocs+1)*sizeof(ghost_lidx_t)));
    
    dueptr[0] = 0;
    for (i=0;i<nprocs;i++) {
        dueptr[i+1] = dueptr[i]+vec->context->dues[i];
    }
    acc_dues = dueptr[nprocs];
    
    GHOST_CALL_RETURN(ghost_malloc((void **)&work,vec->traits.ncols*acc_dues*vec->elSize));
    
#ifdef GHOST_HAVE_CUDA
#ifdef CUDA_COMMUNICATION_ASSEMBLY_DL
    ghost_datatransfer_register("spmv_halo",GHOST_DATATRANSFER_IN,GHOST_DATATRANSFER_RANK_GPU,vec->traits.ncols*vec->traits.nrows*vec->elSize);
    GHOST_INSTR_START(spmv_haloexchange_download)
    vec->downloadNonHalo(vec);
    GHOST_INSTR_STOP(spmv_haloexchange_download)
#else
    if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
        GHOST_CALL_GOTO(ghost_cu_malloc(&cu_work,vec->traits.ncols*acc_dues*vec->elSize),err,ret);
    }
#endif
#endif
    if (vec->traits.storage == GHOST_DENSEMAT_ROWMAJOR) {
        if (permutation && permutation->scope == GHOST_PERMUTATION_LOCAL) {
#pragma omp parallel private(to_PE,i)
            for (to_PE=0 ; to_PE<nprocs ; to_PE++){
#pragma omp for 
                for (i=0; i<vec->context->dues[to_PE]; i++){
                    memcpy(work + (dueptr[to_PE]+i)*vec->elSize*vec->traits.ncols,vec->val[permutation->perm[vec->context->duelist[to_PE][i]]],vec->elSize*vec->traits.ncols);
                }
            }
        } else {
#pragma omp parallel private(to_PE,i)
            for (to_PE=0 ; to_PE<nprocs ; to_PE++){
#pragma omp for 
                for (i=0; i<vec->context->dues[to_PE]; i++){
                    memcpy(work + (dueptr[to_PE]+i)*vec->elSize*vec->traits.ncols,vec->val[vec->context->duelist[to_PE][i]],vec->elSize*vec->traits.ncols);
                }
            }
        }
    } else if (vec->traits.storage == GHOST_DENSEMAT_COLMAJOR) {
        if (permutation && permutation->scope == GHOST_PERMUTATION_LOCAL) {
#ifdef GHOST_HAVE_CUDA
            if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
#ifdef CUDA_COMMUNICATION_ASSEMBLY_KERNEL
                ghost_densemat_cm_cu_communicationassembly(cu_work,dueptr,vec,(ghost_lidx_t *)permutation->cu_perm);
#elif defined(CUDA_COMMUNICATION_ASSEMBLY_MEMCPY)
#pragma omp parallel private(to_PE,i,c)
                for (to_PE=0 ; to_PE<nprocs ; to_PE++) {
#pragma omp for 
                    for (i=0; i<vec->context->dues[to_PE]; i++){
                        for (c=0; c<vec->traits.ncols; c++) {
                            cudaMemcpy(cu_work + (dueptr[to_PE]+i)*vec->elSize*vec->traits.ncols + c*vec->elSize,CUVECVAL_CM(vec,vec->cu_val,c,permutation->perm[vec->context->duelist[to_PE][i]]),vec->elSize,cudaMemcpyDeviceToDevice);
                        }
                    }
                }
#endif
            }
#endif
            if (vec->traits.flags & GHOST_DENSEMAT_HOST) {
                ghost_gidx_t c;
#pragma omp parallel private(to_PE,i,c)
                for (to_PE=0 ; to_PE<nprocs ; to_PE++){
#pragma omp for 
                    for (i=0; i<vec->context->dues[to_PE]; i++){
                        for (c=0; c<vec->traits.ncols; c++) {
                            memcpy(work + (dueptr[to_PE]+i)*vec->elSize*vec->traits.ncols + c*vec->elSize,&vec->val[c][permutation->perm[vec->context->duelist[to_PE][i]]*vec->elSize],vec->elSize);
                        }
                    }
                }
            }
        } else {
#ifdef GHOST_HAVE_CUDA
            if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
#ifdef CUDA_COMMUNICATION_ASSEMBLY_KERNEL
                ghost_densemat_cm_cu_communicationassembly(cu_work,dueptr,vec,NULL);
#elif defined(CUDA_COMMUNICATION_ASSEMBLY_MEMCPY)
                ghost_gidx_t c;
#pragma omp parallel private(to_PE,i,c)
                for (to_PE=0 ; to_PE<nprocs ; to_PE++) {
#pragma omp for 
                    for (i=0; i<vec->context->dues[to_PE]; i++){
                        for (c=0; c<vec->traits.ncols; c++) {
                            cudaMemcpy(cu_work + (dueptr[to_PE]+i)*vec->elSize*vec->traits.ncols + c*vec->elSize,CUVECVAL_CM(vec,vec->cu_val,c,vec->context->duelist[to_PE][i]),vec->elSize,cudaMemcpyDeviceToDevice);
                        }
                    }
                }
#endif
            }
#else
            if (vec->traits.flags & GHOST_DENSEMAT_HOST) {
                ghost_gidx_t c;
#pragma omp parallel private(to_PE,i,c)
                for (to_PE=0 ; to_PE<nprocs ; to_PE++){
#pragma omp for 
                    for (i=0; i<vec->context->dues[to_PE]; i++){
                        for (c=0; c<vec->traits.ncols; c++) {
                                memcpy(work + (dueptr[to_PE]+i)*vec->elSize*vec->traits.ncols + c*vec->elSize,&vec->val[c][vec->context->duelist[to_PE][i]*vec->elSize],vec->elSize);
                        }
                    }
                }
            }
#endif
        }
#ifdef GHOST_HAVE_CUDA
#ifndef CUDA_COMMUNICATION_ASSEMBLY_DL
        if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
#ifdef GHOST_HAVE_TRACK_DATATRANSFERS
            ghost_datatransfer_register("spmv_halo",GHOST_DATATRANSFER_IN,GHOST_DATATRANSFER_RANK_GPU,vec->traits.ncols*acc_dues*vec->elSize);
#endif
            ghost_cu_download(work,cu_work,vec->traits.ncols*acc_dues*vec->elSize);
        }
#endif
#endif
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
    GHOST_INSTR_START(spmv_haloexchange_initiate)
    int nprocs;
    int me; 
    int i, from_PE, to_PE;
    ghost_error_t ret = GHOST_SUCCESS;
    
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, vec->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_rank(&me, vec->context->mpicomm),err,ret);
    
    msgcount = 0;
    GHOST_CALL_GOTO(ghost_malloc((void **)&request,sizeof(MPI_Request)*2*nprocs),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&status,sizeof(MPI_Status)*2*nprocs),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&wishptr,(nprocs+1)*sizeof(ghost_lidx_t)),err,ret);

    wishptr[0] = 0;
    for (i=0;i<nprocs;i++) {
        wishptr[i+1] = wishptr[i]+vec->context->wishes[i];
    }
    acc_wishes = wishptr[nprocs];
    
    if (vec->traits.storage == GHOST_DENSEMAT_COLMAJOR) {
        GHOST_CALL_GOTO(ghost_malloc((void **)&tmprecv_mem,vec->traits.ncols*vec->elSize*acc_wishes),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&tmprecv,nprocs*sizeof(char *)),err,ret);
        
        for (from_PE=0; from_PE<nprocs; from_PE++){
            tmprecv[from_PE] = &tmprecv_mem[wishptr[from_PE]*vec->traits.ncols*vec->elSize];
        }
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
#ifdef GHOST_HAVE_TRACK_DATATRANSFERS
            ghost_datatransfer_register("spmv_halo",GHOST_DATATRANSFER_IN,from_PE,vec->context->wishes[from_PE]*vec->elSize*vec->traits.ncols);
#endif
            MPI_CALL_GOTO(MPI_Irecv(recv, vec->traits.ncols*vec->elSize*vec->context->wishes[from_PE],MPI_CHAR, from_PE, from_PE, vec->context->mpicomm,&request[msgcount]),err,ret);
            msgcount++;
        }
    }
   
    if (!assembled) { 
        GHOST_CALL_GOTO(ghost_spmv_haloexchange_assemble(vec,permutation),err,ret);
    }
    
    for (to_PE=0 ; to_PE<nprocs ; to_PE++){
        if (vec->context->dues[to_PE]>0){
#ifdef GHOST_HAVE_TRACK_DATATRANSFERS
            ghost_datatransfer_register("spmv_halo",GHOST_DATATRANSFER_OUT,to_PE,vec->context->dues[to_PE]*vec->elSize*vec->traits.ncols);
#endif
            MPI_CALL_GOTO(MPI_Isend( work + dueptr[to_PE]*vec->elSize*vec->traits.ncols, vec->context->dues[to_PE]*vec->elSize*vec->traits.ncols, MPI_CHAR, to_PE, me, vec->context->mpicomm, &request[msgcount]),err,ret);
            msgcount++;
        }
    ;}


    goto out;
err:

out:
    GHOST_INSTR_STOP(spmv_haloexchange_initiate)
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
    GHOST_INSTR_START(spmv_haloexchange_finalize)
    ghost_error_t ret = GHOST_SUCCESS;
    int nprocs;
    int i, from_PE;
    ghost_gidx_t c;
    
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
#ifdef GHOST_HAVE_CUDA 
#ifdef GHOST_HAVE_TRACK_DATATRANSFERS
    ghost_datatransfer_register("spmv_halo",GHOST_DATATRANSFER_OUT,GHOST_DATATRANSFER_RANK_GPU,vec->context->halo_elements*vec->traits.ncols*vec->elSize);
#endif
    GHOST_CALL_GOTO(vec->uploadHalo(vec),err,ret);
#endif
    
#ifdef GHOST_HAVE_CUDA || !defined(CUDA_COMMUNICATION_ASSEMBLY_DL)
    if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
        ghost_cu_free(cu_work);
    }
#endif
    free(work); work = NULL;
    free(tmprecv_mem); tmprecv_mem = NULL;
    free(tmprecv); tmprecv = NULL;
    free(request); request = NULL;
    free(status); status = NULL;
    free(dueptr); dueptr = NULL;
    free(wishptr); wishptr = NULL;


    goto out;
err:

out:
    GHOST_INSTR_STOP(spmv_haloexchange_finalize)
    return ret;

#else
    UNUSED(vec);
    return GHOST_ERR_NOT_IMPLEMENTED;
#endif
}
