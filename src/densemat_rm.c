#define _XOPEN_SOURCE 500 
#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/core.h"
#include "ghost/datatransfers.h"
#include "ghost/densemat_rm.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include "ghost/context.h"
#include "ghost/instr.h"
#include "ghost/machine.h"
#include "ghost/log.h"
#include "ghost/bindensemat.h"
#include "ghost/densemat_cm.h"
#include "ghost/constants.h"

#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#ifdef GHOST_HAVE_CUDA
#include <cuda_runtime.h>
#endif

#define ROWMAJOR
#include "ghost/densemat_iter_macros.h"
#include "ghost/densemat_common.c.def"

ghost_error ghost_densemat_rm_distributeVector(ghost_densemat *vec, ghost_densemat *nodeVec, ghost_context *ctx)
{
    DEBUG_LOG(1,"Distributing vector");
    int me;
    int nprocs;
    GHOST_CALL_RETURN(ghost_rank(&me, ctx->mpicomm));
    GHOST_CALL_RETURN(ghost_nrank(&nprocs, ctx->mpicomm));
    
    bool uniformstorage;
    GHOST_CALL_RETURN(ghost_densemat_uniformstorage(&uniformstorage,vec,ctx->mpicomm));
    if (!uniformstorage) {
        ERROR_LOG("Cannot collect vectors of different storage order");
        return GHOST_ERR_INVALID_ARG;
    }

    ghost_lidx c;
#ifdef GHOST_HAVE_MPI
    DEBUG_LOG(2,"Scattering global vector to local vectors");

    ghost_mpi_datatype mpidt;
    GHOST_CALL_RETURN(ghost_mpi_datatype_get(&mpidt,vec->traits.datatype));

    int i;

    MPI_Request req[vec->traits.ncols*2*(nprocs-1)];
    MPI_Status stat[vec->traits.ncols*2*(nprocs-1)];
    int msgcount = 0;

    for (i=0;i<vec->traits.ncols*2*(nprocs-1);i++) 
        req[i] = MPI_REQUEST_NULL;

    if (me != 0) {
        for (c=0; c<vec->traits.ncols; c++) {
            MPI_CALL_RETURN(MPI_Irecv(DENSEMAT_VALPTR(nodeVec,0,c),ctx->row_map->ldim[me],mpidt,0,me,ctx->mpicomm,&req[msgcount]));
            msgcount++;
        }
    } else {
        for (c=0; c<vec->traits.ncols; c++) {
            memcpy(DENSEMAT_VALPTR(nodeVec,0,c),DENSEMAT_VALPTR(vec,0,c),vec->elSize*ctx->row_map->ldim[0]);
            for (i=1;i<nprocs;i++) {
                MPI_CALL_RETURN(MPI_Isend(DENSEMAT_VALPTR(vec,c,ctx->row_map->goffs[i]),ctx->row_map->ldim[i],mpidt,i,i,ctx->mpicomm,&req[msgcount]));
                msgcount++;
            }
        }
    }
    MPI_CALL_RETURN(MPI_Waitall(msgcount,req,stat));
#else

    for (c=0; c<vec->traits.ncols; c++) {
        memcpy(DENSEMAT_VALPTR(nodeVec,0,c),DENSEMAT_VALPTR(vec,0,c),DM_NROWS(vec)*vec->elSize);
    }
    //    *nodeVec = vec->clone(vec);
#endif

    ghost_densemat_upload(nodeVec);

    DEBUG_LOG(1,"Vector distributed successfully");

    return GHOST_SUCCESS;
}

ghost_error ghost_densemat_rm_collectVectors(ghost_densemat *vec, ghost_densemat *totalVec, ghost_context *ctx) 
{
#ifdef GHOST_HAVE_MPI
    ghost_lidx c;
    int me;
    int nprocs;
    ghost_mpi_datatype mpidt;
    GHOST_CALL_RETURN(ghost_rank(&me, ctx->mpicomm));
    GHOST_CALL_RETURN(ghost_nrank(&nprocs, ctx->mpicomm));
    GHOST_CALL_RETURN(ghost_mpi_datatype_get(&mpidt,vec->traits.datatype));
    
    bool uniformstorage;
    GHOST_CALL_RETURN(ghost_densemat_uniformstorage(&uniformstorage,vec,ctx->mpicomm));
    if (!uniformstorage) {
        ERROR_LOG("Cannot collect vectors of different storage order");
        return GHOST_ERR_INVALID_ARG;
    }

//    if (ctx != NULL)
//        vec->permute(vec,ctx->invRowPerm); 

    int i;

    MPI_Request req[vec->traits.ncols*2*(nprocs-1)];
    MPI_Status stat[vec->traits.ncols*2*(nprocs-1)];
    int msgcount = 0;

    for (i=0;i<vec->traits.ncols*2*(nprocs-1);i++) 
        req[i] = MPI_REQUEST_NULL;

    if (me != 0) {
        for (c=0; c<vec->traits.ncols; c++) {
            MPI_CALL_RETURN(MPI_Isend(DENSEMAT_VALPTR(vec,0,c),ctx->row_map->ldim[me],mpidt,0,me,ctx->mpicomm,&req[msgcount]));
            msgcount++;
        }
    } else {
        for (c=0; c<vec->traits.ncols; c++) {
            memcpy(DENSEMAT_VALPTR(totalVec,0,c),DENSEMAT_VALPTR(vec,0,c),vec->elSize*ctx->row_map->ldim[0]);
            for (i=1;i<nprocs;i++) {
                MPI_CALL_RETURN(MPI_Irecv(DENSEMAT_VALPTR(totalVec,c,ctx->row_map->goffs[i]),ctx->row_map->ldim[i],mpidt,i,i,ctx->mpicomm,&req[msgcount]));
                msgcount++;
            }
        }
    }
    MPI_CALL_RETURN(MPI_Waitall(msgcount,req,stat));
#else
    UNUSED(ctx);
    ghost_densemat_init_densemat(totalVec,vec,0,0);
#endif

    return GHOST_SUCCESS;

}


ghost_error ghost_densemat_rm_compress(ghost_densemat *vec)
{
    if (!(vec->traits.flags & GHOST_DENSEMAT_SCATTERED)) {
        return GHOST_SUCCESS;
    }

    if (vec->traits.location & GHOST_LOCATION_HOST) {
        ghost_lidx v,i;

        char *val = NULL;
        if (vec->traits.location & GHOST_LOCATION_DEVICE) {
            GHOST_CALL_RETURN(ghost_malloc_pinned((void **)&val,
                        (size_t)vec->traits.ncolspadded*DM_NROWSPAD(vec)*
                        vec->elSize));
        } else {
            GHOST_CALL_RETURN(ghost_malloc_align((void **)&val,
                        (size_t)vec->traits.ncolspadded*DM_NROWSPAD(vec)*
                        vec->elSize,GHOST_DATA_ALIGNMENT));
        }

#pragma omp parallel for schedule(runtime) private(v)
        for (i=0; i<DM_NROWSPAD(vec); i++) {
            for (v=0; v<vec->traits.ncolspadded; v++) {
                val[(v*DM_NROWSPAD(vec)+i)*vec->elSize] = 0;
            }
        }
        
        DENSEMAT_ITER(vec,memcpy(&val[(row*vec->traits.ncolspadded+col)*vec->elSize],
                    valptr,vec->elSize));
        
        vec->val = val;
      
    }
    if (vec->traits.location & GHOST_LOCATION_DEVICE) {
#ifdef GHOST_HAVE_CUDA

        char *cu_val;
        GHOST_CALL_RETURN(ghost_cu_malloc((void **)&cu_val,DM_NROWSPAD(vec)*vec->traits.ncolspadded*vec->elSize));
        
        DENSEMAT_ITER(vec,ghost_cu_memcpy(&cu_val[(row*vec->traits.ncolspadded+col)*vec->elSize],
                    DENSEMAT_CUVALPTR(vec,memrow,memcol),vec->elSize));

        if (!(vec->traits.flags & GHOST_DENSEMAT_VIEW)) {
            GHOST_CALL_RETURN(ghost_cu_free(vec->cu_val));
        }
        vec->cu_val = cu_val;
#endif 
    }
    ghost_bitmap_set_range(vec->colmask,0,vec->traits.ncols-1);
    ghost_bitmap_set_range(vec->rowmask,0,DM_NROWS(vec)-1);
    vec->traits.flags &= ~(ghost_densemat_flags)GHOST_DENSEMAT_VIEW;
    vec->traits.flags &= ~(ghost_densemat_flags)GHOST_DENSEMAT_SCATTERED;
    vec->stride = vec->traits.ncolspadded;
    vec->src = vec;

    return GHOST_SUCCESS;
}

ghost_error ghost_densemat_rm_halocommInit(ghost_densemat *vec, ghost_context *ctx, ghost_densemat_halo_comm *comm)
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
    ghost_error ret = GHOST_SUCCESS;
    int i, to_PE, from_PE, partner;
    int nprocs;
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, ctx->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_densemat_halocommInit_common(vec,ctx,comm),err,ret);
        
    GHOST_CALL_GOTO(ghost_malloc((void **)&comm->tmprecv,nprocs*sizeof(char *)),err,ret);
    
    if (vec->stride == vec->traits.ncols) {
        if (vec->traits.location & GHOST_LOCATION_DEVICE) {
#ifdef GHOST_HAVE_GPUDIRECT
            for (from_PE=0; from_PE<nprocs; from_PE++) {
                comm->tmprecv[from_PE] = DENSEMAT_CUVALPTR(vec,ctx->hput_pos[from_PE],0);
            }
#else 
            GHOST_INSTR_START("hostAlloc")
            GHOST_CALL_GOTO(ghost_cu_malloc_pinned((void **)&comm->work,(size_t)vec->traits.ncols*comm->acc_dues*vec->elSize),err,ret);
            GHOST_INSTR_STOP("hostAlloc")
            GHOST_CALL_GOTO(ghost_malloc((void **)&comm->tmprecv_mem,vec->traits.ncols*vec->elSize*comm->acc_wishes),err,ret);

            for (from_PE=0; from_PE<nprocs; from_PE++){
                comm->tmprecv[from_PE] = &comm->tmprecv_mem[comm->wishptr[from_PE]*vec->traits.ncols*vec->elSize];
            }
#endif
            GHOST_INSTR_START("deviceAlloc")
            GHOST_CALL_GOTO(ghost_cu_malloc(&comm->cu_work,vec->traits.ncols*comm->acc_dues*vec->elSize),err,ret);
            GHOST_INSTR_STOP("deviceAlloc")
        } else {
            for (from_PE=0; from_PE<nprocs; from_PE++) {
                comm->tmprecv[from_PE] = DENSEMAT_VALPTR(vec,ctx->hput_pos[from_PE],0);
            }
            GHOST_CALL_GOTO(ghost_malloc((void **)&comm->work,(size_t)vec->traits.ncols*comm->acc_dues*vec->elSize),err,ret);
        }
    } else {
        GHOST_CALL_GOTO(ghost_malloc((void **)&comm->tmprecv_mem,vec->traits.ncols*vec->elSize*comm->acc_wishes),err,ret);

        for (from_PE=0; from_PE<nprocs; from_PE++){
            comm->tmprecv[from_PE] = &comm->tmprecv_mem[comm->wishptr[from_PE]*vec->traits.ncols*vec->elSize];
        }
        if (vec->traits.location & GHOST_LOCATION_DEVICE) {
            GHOST_INSTR_START("hostAlloc")
            GHOST_CALL_GOTO(ghost_cu_malloc_pinned((void **)&comm->work,(size_t)vec->traits.ncols*comm->acc_dues*vec->elSize),err,ret);
            GHOST_INSTR_STOP("hostAlloc")
            GHOST_INSTR_START("deviceAlloc")
            GHOST_CALL_GOTO(ghost_cu_malloc(&comm->cu_work,vec->traits.ncols*comm->acc_dues*vec->elSize),err,ret);
            GHOST_INSTR_STOP("deviceAlloc")
        } else {
            GHOST_CALL_GOTO(ghost_malloc((void **)&comm->work,(size_t)vec->traits.ncols*comm->acc_dues*vec->elSize),err,ret);
        }
    }
       
    GHOST_INSTR_START("assemble_buf"); 
    if (ctx->col_map->loc_perm) {
#ifdef GHOST_HAVE_CUDA
        if (vec->traits.location & GHOST_LOCATION_DEVICE) {
            ghost_densemat_cu_rm_communicationassembly(comm->cu_work,comm->dueptr,comm->acc_dues,vec,ctx,ctx->col_map->cu_loc_perm);
        } else 
#endif
            if (vec->traits.location & GHOST_LOCATION_HOST) {
                for (partner = 0; partner<ctx->nduepartners; partner++) {
                    to_PE = ctx->duepartners[partner];
#pragma omp parallel for 
                    for (i=0; i<ctx->dues[to_PE]; i++){
                        memcpy(comm->work + (comm->dueptr[to_PE]+i)*vec->elSize*vec->traits.ncols,DENSEMAT_VALPTR(vec,ctx->col_map->loc_perm[ctx->duelist[to_PE][i]],0),vec->elSize*vec->traits.ncols);
                    }
                }
            }
    } else {
#ifdef GHOST_HAVE_CUDA
        if (vec->traits.location & GHOST_LOCATION_DEVICE) {
            ghost_densemat_cu_rm_communicationassembly(comm->cu_work,comm->dueptr,comm->acc_dues,vec,ctx,NULL);
        } else 
#endif
            if (vec->traits.location & GHOST_LOCATION_HOST) {
                for (partner = 0; partner<ctx->nduepartners; partner++) {
                    to_PE = ctx->duepartners[partner];
#pragma omp parallel for 
                    for (i=0; i<ctx->dues[to_PE]; i++){
                        memcpy(comm->work + (comm->dueptr[to_PE]+i)*vec->elSize*vec->traits.ncols,DENSEMAT_VALPTR(vec,ctx->duelist[to_PE][i],0),vec->elSize*vec->traits.ncols);
                    }
                }
            }
    }
    GHOST_INSTR_STOP("assemble_buf");
#ifdef GHOST_HAVE_CUDA
    if (vec->traits.location & GHOST_LOCATION_DEVICE) {
        GHOST_INSTR_START("download_buf");
#ifdef GHOST_TRACK_DATATRANSFERS
        ghost_datatransfer_register("spmv_halo",GHOST_DATATRANSFER_IN,GHOST_DATATRANSFER_RANK_GPU,vec->traits.ncols*comm->acc_dues*vec->elSize);

#endif
#ifdef GHOST_HAVE_GPUDIRECT
        if (vec->traits.ncols != vec->stride)
#endif 
        {
            GHOST_CALL_GOTO(ghost_cu_download(comm->work,comm->cu_work,vec->traits.ncols*comm->acc_dues*vec->elSize),err,ret);
        }
        GHOST_INSTR_STOP("download_buf");
    }
#endif
    

    goto out;
err:

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_COMMUNICATION);
    return ret;
#else
    UNUSED(vec);
    UNUSED(comm);
    UNUSED(ctx);
    return GHOST_ERR_NOT_IMPLEMENTED;
#endif


}

ghost_error ghost_densemat_rm_halocommFinalize(ghost_densemat *vec, ghost_context *ctx, ghost_densemat_halo_comm *comm)
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
    ghost_error ret = GHOST_SUCCESS;
    
    int nprocs;
    int i, from_PE, partner;
    
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, ctx->mpicomm),err,ret);

    GHOST_CALL_GOTO(ghost_densemat_halocommFinalize_common(comm),err,ret);
    if ((vec->stride != vec->traits.ncols) && (vec->traits.location == GHOST_LOCATION_HOST)) {
        GHOST_INSTR_START("Assemble row-major view");
        for (partner=0; partner<ctx->nwishpartners; partner++){
            from_PE = ctx->wishpartners[partner];
            for (i=0; i<ctx->wishes[from_PE]; i++){
                memcpy(DENSEMAT_VALPTR(vec,ctx->hput_pos[from_PE]+i,0),&comm->tmprecv[from_PE][(i*vec->traits.ncols)*vec->elSize],vec->elSize*vec->traits.ncols);
            }
        }
        GHOST_INSTR_STOP("Assemble row-major view");
    }

#ifdef GHOST_HAVE_CUDA 
    GHOST_INSTR_START("upload")
    if (vec->traits.location & GHOST_LOCATION_DEVICE) {
#ifdef GHOST_TRACK_DATATRANSFERS
        ghost_datatransfer_register("spmv_halo",GHOST_DATATRANSFER_OUT,GHOST_DATATRANSFER_RANK_GPU,ctx->col_map->nhalo*vec->traits.ncols*vec->elSize);
#endif
#ifdef GHOST_HAVE_GPUDIRECT
        if (vec->traits.ncols != vec->stride)
#endif 
        {
            GHOST_CALL_GOTO(ghost_cu_upload2d(DENSEMAT_CUVALPTR(vec,vec->map->dimhalo-vec->map->nhalo,0),vec->stride*vec->elSize,comm->tmprecv_mem,vec->traits.ncols*vec->elSize,vec->traits.ncols*vec->elSize,comm->acc_wishes),err,ret);
        }
    }
    GHOST_INSTR_STOP("upload");
#endif


    if (vec->traits.location & GHOST_LOCATION_DEVICE) {
        GHOST_CALL_GOTO(ghost_cu_free(comm->cu_work),err,ret); comm->cu_work = NULL;
        GHOST_CALL_GOTO(ghost_cu_free_host(comm->work),err,ret); comm->work = NULL;
    } else {
        free(comm->work); comm->work = NULL;
    }
    free(comm->tmprecv_mem); comm->tmprecv_mem = NULL;
    free(comm->tmprecv); comm->tmprecv = NULL;
    free(comm->request); comm->request = NULL;
    free(comm->status); comm->status = NULL;
    free(comm->dueptr); comm->dueptr = NULL;
    free(comm->wishptr); comm->wishptr = NULL;
    
    goto out;
err:

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_COMMUNICATION);
    return ret;

#else
    UNUSED(vec);
    UNUSED(comm);
    UNUSED(ctx);
    return GHOST_ERR_NOT_IMPLEMENTED;
#endif
}
