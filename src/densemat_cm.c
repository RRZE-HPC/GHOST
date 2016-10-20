#define _XOPEN_SOURCE 500 
#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/core.h"
#include "ghost/datatransfers.h"
#include "ghost/densemat_cm.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include "ghost/context.h"
#include "ghost/instr.h"
#include "ghost/machine.h"
#include "ghost/log.h"
#include "ghost/bindensemat.h"
#include "ghost/densemat_rm.h"
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

#define COLMAJOR
#include "ghost/densemat_iter_macros.h"
#include "ghost/densemat_common.c.def"

ghost_error ghost_densemat_cm_distributeVector(ghost_densemat *vec, ghost_densemat *nodeVec, ghost_context *ctx)
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
                MPI_CALL_RETURN(MPI_Isend(DENSEMAT_VALPTR(vec,ctx->row_map->goffs[i],c),ctx->row_map->ldim[i],mpidt,i,i,ctx->mpicomm,&req[msgcount]));
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

ghost_error ghost_densemat_cm_collectVectors(ghost_densemat *vec, ghost_densemat *totalVec, ghost_context *ctx) 
{
    ghost_lidx c;
#ifdef GHOST_HAVE_MPI
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

    for (i=0;i<vec->traits.ncols*2*(nprocs-1);i++) {
        req[i] = MPI_REQUEST_NULL;
    }

    if (me != 0) {
        for (c=0; c<vec->traits.ncols; c++) {
            MPI_CALL_RETURN(MPI_Isend(DENSEMAT_VALPTR(vec,0,c),ctx->row_map->ldim[me],mpidt,0,me,ctx->mpicomm,&req[msgcount]));
            msgcount++;
        }
    } else {
        for (c=0; c<vec->traits.ncols; c++) {
            memcpy(DENSEMAT_VALPTR(totalVec,0,c),DENSEMAT_VALPTR(vec,0,c),vec->elSize*ctx->row_map->ldim[0]);
            for (i=1;i<nprocs;i++) {
                MPI_CALL_RETURN(MPI_Irecv(DENSEMAT_VALPTR(totalVec,ctx->row_map->goffs[i],c),ctx->row_map->ldim[i],mpidt,i,i,ctx->mpicomm,&req[msgcount]));
                msgcount++;
            }
        }
    }
    MPI_CALL_RETURN(MPI_Waitall(msgcount,req,stat));
#else
    if (ctx != NULL) {
//        vec->permute(vec,ctx->invRowPerm);
        for (c=0; c<vec->traits.ncols; c++) {
DM_NROWS(            memcpy(DENSEMAT_VALPTR(totalVec,0,c),DENSEMAT_VALPTR(vec,0,c),totalVec)*vec->elSize);
        }
    }
#endif

    return GHOST_SUCCESS;

}

ghost_error ghost_densemat_cm_compress(ghost_densemat *vec)
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
        for (i=0; i<DM_NROWSPAD(vec); i++)
        {
            for (v=0; v<vec->traits.ncols; v++)
            {
                val[(v*DM_NROWSPAD(vec)+i)*vec->elSize] = 0;
            }
        }

        
        DENSEMAT_ITER(vec,memcpy(&val[((col)*DM_NROWSPAD(vec)+(row))*vec->elSize],valptr,vec->elSize));

        vec->val = val;
        
/*        for (v=0; v<vec->traits.ncols; v++)
        {
            memcpy(&val[(v*DM_NROWS(vec)padded)*vec->elSize],
                    DENSEMAT_VALPTR(vec,0,v),DM_NROWS(vec)padded*vec->elSize);

            if (!(vec->traits.flags & GHOST_DENSEMAT_VIEW)) {
                free(vec->val[v]);
            }
            vec->val[v] = &val[(v*DM_NROWS(vec)padded)*vec->elSize];
        }*/
    }
    if (vec->traits.location & GHOST_LOCATION_DEVICE) {
#ifdef GHOST_HAVE_CUDA

        char *cu_val;
        GHOST_CALL_RETURN(ghost_cu_malloc((void **)&cu_val,DM_NROWSPAD(vec)*vec->traits.ncols*vec->elSize));
        
        DENSEMAT_ITER(vec,ghost_cu_memcpy(&cu_val[(col*DM_NROWSPAD(vec)+col)*vec->elSize],
                    DENSEMAT_CUVALPTR(vec,memrow,memcol),vec->elSize));

        if (!(vec->traits.flags & GHOST_DENSEMAT_VIEW)) {
            GHOST_CALL_RETURN(ghost_cu_free(vec->cu_val));
        }
        vec->cu_val = cu_val;
#endif 
    }

    ghost_bitmap_free(vec->rowmask); vec->rowmask = NULL;
    ghost_bitmap_free(vec->colmask); vec->colmask = NULL;
    vec->traits.flags &= ~(ghost_densemat_flags)GHOST_DENSEMAT_VIEW;
    vec->traits.flags &= ~(ghost_densemat_flags)GHOST_DENSEMAT_SCATTERED;
    vec->stride = DM_NROWSPAD(vec);
    vec->src = vec;

    return GHOST_SUCCESS;
}
    
ghost_error ghost_densemat_cm_halocommInit(ghost_densemat *vec, ghost_context *ctx, ghost_densemat_halo_comm *comm)
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
    ghost_error ret = GHOST_SUCCESS;
    int i, to_PE, from_PE, partner;
    int nprocs, me;

    GHOST_CALL_GOTO(ghost_rank(&me, ctx->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, ctx->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_densemat_halocommInit_common(vec,ctx,comm),err,ret);
    
    GHOST_CALL_GOTO(ghost_malloc((void **)&comm->tmprecv,nprocs*sizeof(char *)),err,ret);

    GHOST_CALL_GOTO(ghost_malloc((void **)&comm->tmprecv_mem,vec->traits.ncols*vec->elSize*comm->acc_wishes),err,ret);

    for (from_PE=0; from_PE<nprocs; from_PE++){
        comm->tmprecv[from_PE] = &comm->tmprecv_mem[comm->wishptr[from_PE]*vec->traits.ncols*vec->elSize];
    }
        
    
    if (ctx->col_map->loc_perm) {
#ifdef GHOST_HAVE_CUDA
        if (vec->traits.location & GHOST_LOCATION_DEVICE) {
            ghost_densemat_cu_cm_communicationassembly(comm->cu_work,comm->dueptr,comm->acc_dues,vec,ctx,ctx->col_map->cu_loc_perm);
        } else
#endif
            if (vec->traits.location & GHOST_LOCATION_HOST) {
                ghost_gidx c;
                for (partner = 0; partner<ctx->nduepartners; partner++) {
                    to_PE = ctx->duepartners[partner];
#pragma omp parallel for private(c) 
                    for (i=0; i<ctx->dues[to_PE]; i++){
                        for (c=0; c<vec->traits.ncols; c++) {
                            memcpy(comm->work + (c*ctx->dues[to_PE]+comm->dueptr[to_PE]*vec->traits.ncols+i)*vec->elSize,
                                    DENSEMAT_VALPTR(vec,ctx->col_map->loc_perm[ctx->duelist[to_PE][i]],c),vec->elSize);//change to colPerm
                        }
                    }
                }
            }
    } else {
#ifdef GHOST_HAVE_CUDA
        if (vec->traits.location & GHOST_LOCATION_DEVICE) {
            ghost_densemat_cu_cm_communicationassembly(comm->cu_work,comm->dueptr,comm->acc_dues,vec,ctx,NULL);
        } else
#endif
            if (vec->traits.location & GHOST_LOCATION_HOST) {
                ghost_gidx c;
                for (partner = 0; partner<ctx->nduepartners; partner++) {
                    to_PE = ctx->duepartners[partner];
#pragma omp parallel for private(c) 
                    for (i=0; i<ctx->dues[to_PE]; i++){
                        for (c=0; c<vec->traits.ncols; c++) {
                            memcpy(comm->work + (c*ctx->dues[to_PE]+comm->dueptr[to_PE]*vec->traits.ncols+i)*vec->elSize,
                                    DENSEMAT_VALPTR(vec,ctx->duelist[to_PE][i],c),vec->elSize);
                        }
                    }
                }
            }
    }
      
#ifdef GHOST_HAVE_CUDA
    if (vec->traits.location & GHOST_LOCATION_DEVICE) {
        GHOST_INSTR_START("downloadcomm->work");
#ifdef GHOST_TRACK_DATATRANSFERS
        ghost_datatransfer_register("spmv_halo",GHOST_DATATRANSFER_IN,GHOST_DATATRANSFER_RANK_GPU,vec->traits.ncols*comm->acc_dues*vec->elSize);

#endif
        ghost_cu_download(comm->work,comm->cu_work,vec->traits.ncols*comm->acc_dues*vec->elSize);
        GHOST_INSTR_STOP("downloadcomm->work");
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
    return GHOST_ERR_NOT_IMPLEMENTED;
#endif


}

ghost_error ghost_densemat_cm_halocommFinalize(ghost_densemat *vec, ghost_context *ctx, ghost_densemat_halo_comm *comm)
{
#ifdef GHOST_HAVE_MPI
GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
ghost_error ret = GHOST_SUCCESS;

int nprocs;
int i, from_PE, partner;

GHOST_CALL_GOTO(ghost_nrank(&nprocs, ctx->mpicomm),err,ret);

            
ghost_densemat_halocommFinalize_common(comm);
    
if (vec->traits.location & GHOST_LOCATION_HOST) {
    GHOST_INSTR_START("Assemble row-major view");
    for (partner=0; partner<ctx->nwishpartners; partner++){
        from_PE = ctx->wishpartners[partner];
/*        if( (ctx->perm_local) && (ctx->flags & GHOST_PERM_NO_DISTINCTION) ){
        //copy to permuted position
            for (i=0; i<ctx->wishes[from_PE]; i++){
                for (c=0; c<vec->traits.ncols; c++) {
                       memcpy(DENSEMAT_VALPTR(vec,ctx->perm_local->colPerm[ctx->hput_pos[from_PE]+i],c),
                                &comm->tmprecv[from_PE][(c*ctx->wishes[from_PE]+i)*vec->elSize],vec->elSize);
                }
            }
        } else { */
     for (i=0; i<vec->traits.ncols; i++){
                memcpy(DENSEMAT_VALPTR(vec,ctx->hput_pos[from_PE],i),&comm->tmprecv[from_PE][(i*ctx->wishes[from_PE])*vec->elSize],vec->elSize*ctx->wishes[from_PE]);
 //           }
        }
    }
   
    GHOST_INSTR_STOP("Assemble row-major view");
}

#ifdef GHOST_HAVE_CUDA 
GHOST_INSTR_START("upload")
if (vec->traits.location & GHOST_LOCATION_DEVICE) {
#ifdef GHOST_TRACK_DATATRANSFERS
    ghost_datatransfer_register("spmv_halo",GHOST_DATATRANSFER_OUT,GHOST_DATATRANSFER_RANK_GPU,ctx->halo_elements*vec->traits.ncols*vec->elSize);
#endif
    for (from_PE=0; from_PE<nprocs; from_PE++){
        ghost_cu_upload2d(DENSEMAT_CUVALPTR(vec,ctx->hput_pos[from_PE],0),vec->stride*vec->elSize,comm->tmprecv[from_PE],ctx->wishes[from_PE]*vec->elSize,ctx->wishes[from_PE]*vec->elSize,vec->traits.ncols);
    }
}
GHOST_INSTR_STOP("upload");
#endif

    if (vec->traits.location & GHOST_LOCATION_DEVICE) {
#ifdef GHOST_HAVE_CUDA
        GHOST_CALL_GOTO(ghost_cu_free(comm->cu_work),err,ret);
        cudaFreeHost(comm->work); comm->work = NULL;
#endif
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
return GHOST_ERR_NOT_IMPLEMENTED;
#endif


}
