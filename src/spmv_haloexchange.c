#include "ghost/instr.h"
#include "ghost/error.h"
#include "ghost/util.h"
#include "ghost/sparsemat.h"
#include "ghost/context.h"
#include "ghost/locality.h"
#include "ghost/datatransfers.h"
#ifdef GHOST_HAVE_CUDA
#include "ghost/cu_densemat_rm.h"
#include "ghost/cu_densemat_cm.h"
#endif

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
#ifdef GHOST_HAVE_CUDA
static void *cu_work;
#endif

#endif

ghost_error_t ghost_spmv_haloexchange_assemble(ghost_densemat_t *vec)
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
    ghost_error_t ret = GHOST_SUCCESS;
    int nprocs;
    ghost_permutation_t *permutation = vec->context->permutation;
    //max_dues = 0;
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, vec->context->mpicomm),err,ret);
    int i, to_PE;

    if (nprocs == 1) {
        goto out;
    }

    GHOST_CALL_RETURN(ghost_malloc((void **)&dueptr,(nprocs+1)*sizeof(ghost_lidx_t)));

    dueptr[0] = 0;
    for (i=0;i<nprocs;i++) {
        dueptr[i+1] = dueptr[i]+vec->context->dues[i];
    }
    acc_dues = dueptr[nprocs];

    if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
#ifdef GHOST_HAVE_CUDA
        CUDA_CALL_RETURN(cudaHostAlloc((void **)&work,(size_t)vec->traits.ncols*acc_dues*vec->elSize,cudaHostAllocDefault));
#endif
    } else {
        GHOST_CALL_RETURN(ghost_malloc((void **)&work,(size_t)vec->traits.ncols*acc_dues*vec->elSize));
    }

#ifdef GHOST_HAVE_CUDA
    if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
        GHOST_CALL_GOTO(ghost_cu_malloc(&cu_work,vec->traits.ncols*acc_dues*vec->elSize),err,ret);
    }
#endif
    if (vec->traits.storage == GHOST_DENSEMAT_ROWMAJOR) {
        if (permutation && permutation->scope == GHOST_PERMUTATION_LOCAL) {
#ifdef GHOST_HAVE_CUDA
            if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
                ghost_densemat_rm_cu_communicationassembly(cu_work,dueptr,vec,(ghost_lidx_t *)permutation->cu_perm);
            } else 
#endif
                if (vec->traits.flags & GHOST_DENSEMAT_HOST) {
#pragma omp parallel private(to_PE,i)
                    for (to_PE=0 ; to_PE<nprocs ; to_PE++){
#pragma omp for 
                        for (i=0; i<vec->context->dues[to_PE]; i++){
                            memcpy(work + (dueptr[to_PE]+i)*vec->elSize*vec->traits.ncols,vec->val[permutation->perm[vec->context->duelist[to_PE][i]]],vec->elSize*vec->traits.ncols);
                        }
                    }
                }
        } else {
#ifdef GHOST_HAVE_CUDA
            if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
                ghost_densemat_rm_cu_communicationassembly(cu_work,dueptr,vec,NULL);
            } else 
#endif
                if (vec->traits.flags & GHOST_DENSEMAT_HOST) {
#pragma omp parallel private(to_PE,i)
                    for (to_PE=0 ; to_PE<nprocs ; to_PE++){
#pragma omp for 
                        for (i=0; i<vec->context->dues[to_PE]; i++){
                            memcpy(work + (dueptr[to_PE]+i)*vec->elSize*vec->traits.ncols,&(vec->val[vec->context->duelist[to_PE][i]][ghost_bitmap_first(vec->ldmask)*vec->elSize]),vec->elSize*vec->traits.ncols);
                        }
                    }
                }
        }
    } else if (vec->traits.storage == GHOST_DENSEMAT_COLMAJOR) {
        if (permutation && permutation->scope == GHOST_PERMUTATION_LOCAL) {
#ifdef GHOST_HAVE_CUDA
            if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
                ghost_densemat_cm_cu_communicationassembly(cu_work,dueptr,vec,(ghost_lidx_t *)permutation->cu_perm);
            } else
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
                ghost_densemat_cm_cu_communicationassembly(cu_work,dueptr,vec,NULL);
            } else
#endif
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
        }
    }

#ifdef GHOST_HAVE_CUDA
    if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
        GHOST_INSTR_START("downloadwork");
#ifdef GHOST_HAVE_TRACK_DATATRANSFERS
        ghost_datatransfer_register("spmv_halo",GHOST_DATATRANSFER_IN,GHOST_DATATRANSFER_RANK_GPU,vec->traits.ncols*acc_dues*vec->elSize);

#endif
        ghost_cu_download(work,cu_work,vec->traits.ncols*acc_dues*vec->elSize);
        GHOST_INSTR_STOP("downloadwork");
    }
#endif

    goto out;
err:

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_COMMUNICATION);
    return ret;
#else
    UNUSED(vec);
    return GHOST_ERR_NOT_IMPLEMENTED;
#endif
}

ghost_error_t ghost_spmv_haloexchange_initiate(ghost_densemat_t *vec, bool assembled)
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION)
        int nprocs;
    int me; 
    int i, from_PE, to_PE;
    ghost_error_t ret = GHOST_SUCCESS;
    int rowsize;
    MPI_CALL_GOTO(MPI_Type_size(vec->row_mpidt,&rowsize),err,ret);

    GHOST_CALL_GOTO(ghost_nrank(&nprocs, vec->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_rank(&me, vec->context->mpicomm),err,ret);

    msgcount = 0;
    GHOST_CALL_GOTO(ghost_malloc((void **)&wishptr,(nprocs+1)*sizeof(ghost_lidx_t)),err,ret);

    int nMsgsOverall = 0;

    wishptr[0] = 0;
    for (i=0;i<nprocs;i++) {
        wishptr[i+1] = wishptr[i]+vec->context->wishes[i];
        if (vec->context->wishes[i]) {
            nMsgsOverall += ((size_t)rowsize*vec->context->wishes[i])/INT_MAX + 1;
        }
    }
    acc_wishes = wishptr[nprocs];
    nMsgsOverall *= 2; // we need to send _and_ receive

    GHOST_CALL_GOTO(ghost_malloc((void **)&request,sizeof(MPI_Request)*nMsgsOverall),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&status,sizeof(MPI_Status)*nMsgsOverall),err,ret);

    if ((vec->traits.storage == GHOST_DENSEMAT_COLMAJOR && vec->traits.ncols > 1) || (vec->traits.storage == GHOST_DENSEMAT_ROWMAJOR && vec->traits.ncols != vec->traits.ncolspadded)) {
        GHOST_CALL_GOTO(ghost_malloc((void **)&tmprecv_mem,vec->traits.ncols*vec->elSize*acc_wishes),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&tmprecv,nprocs*sizeof(char *)),err,ret);

        for (from_PE=0; from_PE<nprocs; from_PE++){
            tmprecv[from_PE] = &tmprecv_mem[wishptr[from_PE]*vec->traits.ncols*vec->elSize];
        }
    }

    msgcount = 0;
    for (i=0;i<nMsgsOverall;i++) {
        request[i] = MPI_REQUEST_NULL;
    }

    char *recv;

    for (from_PE=0; from_PE<nprocs; from_PE++){
        if (vec->context->wishes[from_PE]>0) {
            if (vec->traits.storage == GHOST_DENSEMAT_COLMAJOR) {
                if (vec->traits.ncols > 1) {
                    recv = tmprecv[from_PE];
                } else {
                    recv = &vec->val[0][vec->context->hput_pos[from_PE]*vec->elSize];
                }
            } else {
                if (vec->traits.ncols != vec->traits.ncolspadded) {
                    recv = tmprecv[from_PE];
                } else {
                    recv = vec->val[vec->context->hput_pos[from_PE]];
                }
            }
#ifdef GHOST_HAVE_TRACK_DATATRANSFERS
            ghost_datatransfer_register("spmv_halo",GHOST_DATATRANSFER_IN,from_PE,vec->context->wishes[from_PE]*vec->elSize*vec->traits.ncols);
#endif
            int msg;
            int nmsgs = (size_t)rowsize*vec->context->wishes[from_PE]/INT_MAX + 1;
            size_t msgSizeRows = vec->context->wishes[from_PE]/nmsgs;

            for (msg = 0; msg < nmsgs-1; msg++) {
                MPI_CALL_GOTO(MPI_Irecv(recv + msg*msgSizeRows*rowsize, msgSizeRows, vec->row_mpidt, from_PE, from_PE, vec->context->mpicomm,&request[msgcount]),err,ret);
                msgcount++;
            }

            // remainder
            MPI_CALL_GOTO(MPI_Irecv(recv + msg*msgSizeRows*rowsize, vec->context->wishes[from_PE] - msg*msgSizeRows, vec->row_mpidt, from_PE, from_PE, vec->context->mpicomm,&request[msgcount]),err,ret);
            msgcount++;
        }
    }

    if (!assembled) { 
        GHOST_CALL_GOTO(ghost_spmv_haloexchange_assemble(vec),err,ret);
    }

    for (to_PE=0 ; to_PE<nprocs ; to_PE++){
        if (vec->context->dues[to_PE]>0){
#ifdef GHOST_HAVE_TRACK_DATATRANSFERS
            ghost_datatransfer_register("spmv_halo",GHOST_DATATRANSFER_OUT,to_PE,vec->context->dues[to_PE]*vec->elSize*vec->traits.ncols);
#endif
            int msg;
            int nmsgs = (size_t)rowsize*vec->context->dues[to_PE]/INT_MAX + 1;
            size_t msgSizeRows = vec->context->dues[to_PE]/nmsgs;

            for (msg = 0; msg < nmsgs-1; msg++) {
                MPI_CALL_GOTO(MPI_Isend(work + dueptr[to_PE]*vec->elSize*vec->traits.ncols+msg*msgSizeRows*rowsize, msgSizeRows, vec->row_mpidt, to_PE, me, vec->context->mpicomm, &request[msgcount]),err,ret);
                msgcount++;
            }

            // remainder
            MPI_CALL_GOTO(MPI_Isend(work + dueptr[to_PE]*vec->elSize*vec->traits.ncols+msg*msgSizeRows*rowsize, vec->context->dues[to_PE] - msg*msgSizeRows, vec->row_mpidt, to_PE, me, vec->context->mpicomm, &request[msgcount]),err,ret);
            msgcount++;
        }
        ;}


    goto out;
err:

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_COMMUNICATION);
    return ret;
#else
    UNUSED(vec);
    UNUSED(assembled);
    return GHOST_ERR_NOT_IMPLEMENTED;
#endif
} 


ghost_error_t ghost_spmv_haloexchange_finalize(ghost_densemat_t *vec)
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
    ghost_error_t ret = GHOST_SUCCESS;
    int nprocs;
    int i, from_PE;
    ghost_gidx_t c;
    if (!request) {
        ERROR_LOG("The request array is NULL!");
        ret = GHOST_ERR_UNKNOWN;
        goto err;
    }
    if (!status) {
        ERROR_LOG("The status array is NULL!");
        ret = GHOST_ERR_UNKNOWN;
        goto err;
    }

    GHOST_CALL_GOTO(ghost_nrank(&nprocs, vec->context->mpicomm),err,ret);

    GHOST_INSTR_START("waitall");
    MPI_CALL_GOTO(MPI_Waitall(msgcount, request, status),err,ret);
    GHOST_INSTR_STOP("waitall");

    if (vec->traits.storage == GHOST_DENSEMAT_COLMAJOR && vec->traits.ncols > 1) {
        GHOST_INSTR_START("re-order from col-major");
        for (from_PE=0; from_PE<nprocs; from_PE++){
            for (i=0; i<vec->context->wishes[from_PE]; i++){
                for (c=0; c<vec->traits.ncols; c++) {
                    memcpy(&vec->val[c][(vec->context->hput_pos[from_PE]+i)*vec->elSize],&tmprecv[from_PE][(i*vec->traits.ncols+c)*vec->elSize],vec->elSize);
                }
            }
        }
        GHOST_INSTR_STOP("re-order from col-major");
    }   
    if (vec->traits.storage == GHOST_DENSEMAT_ROWMAJOR && vec->traits.ncols != vec->traits.ncolspadded) {
        GHOST_INSTR_START("Assemble row-major view");
        for (from_PE=0; from_PE<nprocs; from_PE++){
            for (i=0; i<vec->context->wishes[from_PE]; i++){
                memcpy(&(vec->val[(vec->context->hput_pos[from_PE]+i)][ghost_bitmap_first(vec->ldmask)*vec->elSize]),&tmprecv[from_PE][(i*vec->traits.ncols)*vec->elSize],vec->elSize*vec->traits.ncols);
            }
        }
        GHOST_INSTR_STOP("Assemble row-major view");
    }

#ifdef GHOST_HAVE_CUDA 
    GHOST_INSTR_START("upload")
#ifdef GHOST_HAVE_TRACK_DATATRANSFERS
        if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
            ghost_datatransfer_register("spmv_halo",GHOST_DATATRANSFER_OUT,GHOST_DATATRANSFER_RANK_GPU,vec->context->halo_elements*vec->traits.ncols*vec->elSize);
        }
#endif
    GHOST_CALL_GOTO(vec->uploadHalo(vec),err,ret);
    GHOST_INSTR_STOP("upload")
#endif

        if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
#ifdef GHOST_HAVE_CUDA
            ghost_cu_free(cu_work);
            cudaFreeHost(work); work = NULL;
#endif
        } else {
            free(work); work = NULL;
        }
    free(tmprecv_mem); tmprecv_mem = NULL;
    free(tmprecv); tmprecv = NULL;
    free(request); request = NULL;
    free(status); status = NULL;
    free(dueptr); dueptr = NULL;
    free(wishptr); wishptr = NULL;


    goto out;
err:

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_COMMUNICATION);
    return ret;

#else
    UNUSED(vec);
    return GHOST_ERR_NOT_IMPLEMENTED;
#endif
}
