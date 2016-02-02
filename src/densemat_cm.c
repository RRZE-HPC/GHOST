#define _XOPEN_SOURCE 500 
#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/core.h"
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

static ghost_error vec_cm_fromFunc(ghost_densemat *vec, int (*fp)(ghost_gidx, ghost_lidx, void *, void *), void *arg);
static ghost_error ghost_distributeVector(ghost_densemat *vec, ghost_densemat *nodeVec);
static ghost_error ghost_collectVectors(ghost_densemat *vec, ghost_densemat *totalVec); 
static ghost_error ghost_cloneVector(ghost_densemat *src, ghost_densemat **new, ghost_lidx nr, ghost_lidx roffs, ghost_lidx nc, ghost_lidx coffs);
static ghost_error vec_cm_compress(ghost_densemat *vec);
static ghost_error densemat_cm_halocommInit(ghost_densemat *vec, ghost_densemat_halo_comm *comm);
static ghost_error densemat_cm_halocommFinalize(ghost_densemat *vec, ghost_densemat_halo_comm *comm);

ghost_error ghost_densemat_cm_setfuncs(ghost_densemat *vec)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_SETUP);
    ghost_error ret = GHOST_SUCCESS;

    if (vec->traits.location & GHOST_LOCATION_DEVICE)
    {
#ifdef GHOST_HAVE_CUDA
        vec->localdot_vanilla = &ghost_densemat_cu_cm_dotprod;
        vec->vaxpy = &ghost_densemat_cu_cm_vaxpy;
        vec->vaxpby = &ghost_densemat_cu_cm_vaxpby;
        vec->axpy = &ghost_densemat_cu_cm_axpy;
        vec->axpby = &ghost_densemat_cu_cm_axpby;
        vec->axpbypcz = &ghost_densemat_cu_cm_axpbypcz;
        vec->vaxpbypcz = &ghost_densemat_cu_cm_vaxpbypcz;
        vec->scale = &ghost_densemat_cu_cm_scale;
        vec->vscale = &ghost_densemat_cu_cm_vscale;
        vec->fromScalar = &ghost_densemat_cu_cm_fromScalar;
        vec->fromRand = &ghost_densemat_cu_cm_fromRand;
        vec->conj = &ghost_densemat_cu_cm_conj;
#endif
    }
    else 
    {
        vec->norm = &ghost_densemat_cm_norm_selector;
        vec->localdot_vanilla = &ghost_densemat_cm_dotprod_selector;
        vec->vaxpy = &ghost_densemat_cm_vaxpy_selector;
        vec->vaxpby = &ghost_densemat_cm_vaxpby_selector;
        vec->axpy = &ghost_densemat_cm_axpy;
        vec->axpby = &ghost_densemat_cm_axpby;
        vec->axpbypcz = &ghost_densemat_cm_axpbypcz;
        vec->vaxpbypcz = &ghost_densemat_cm_vaxpbypcz_selector;
        vec->scale = &ghost_densemat_cm_scale;
        vec->vscale = &ghost_densemat_cm_vscale_selector;
        vec->fromScalar = &ghost_densemat_cm_fromScalar_selector;
        vec->fromRand = &ghost_densemat_cm_fromRand_selector;
        vec->conj = &ghost_densemat_cm_conj_selector;
    }

    vec->reduce = &ghost_densemat_cm_reduce;
    vec->compress = &vec_cm_compress;
    vec->string = &ghost_densemat_cm_string_selector;
    vec->fromFunc = &vec_cm_fromFunc;
    vec->fromVec = &ghost_densemat_cm_fromVec_selector;
    vec->fromFile = &ghost_densemat_cm_fromFile;
    vec->toFile = &ghost_densemat_cm_toFile;
    vec->distribute = &ghost_distributeVector;
    vec->collect = &ghost_collectVectors;
    vec->normalize = &ghost_densemat_cm_normalize_selector;
    vec->destroy = &ghost_densemat_destroy;
    vec->permute = &ghost_densemat_cm_permute_selector;
    vec->clone = &ghost_cloneVector;
    vec->entry = &ghost_densemat_cm_entry;
    vec->viewVec = &ghost_densemat_cm_view;
    vec->viewPlain = &ghost_densemat_cm_viewPlain;
    vec->viewScatteredVec = &ghost_densemat_cm_viewScatteredVec;
    vec->viewScatteredCols = &ghost_densemat_cm_viewScatteredCols;
    vec->viewCols = &ghost_densemat_cm_viewCols;
    vec->syncValues = &ghost_densemat_cm_syncValues;
    vec->halocommInit = &densemat_cm_halocommInit;
    vec->halocommFinalize = &densemat_cm_halocommFinalize;
    vec->halocommStart = &ghost_densemat_halocommStart_common;

    vec->averageHalo = &ghost_densemat_cm_averagehalo_selector;

    vec->upload = &ghost_densemat_cm_upload;
    vec->download = &ghost_densemat_cm_download;

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_SETUP);
    return ret;
}


static ghost_error vec_cm_fromFunc(ghost_densemat *vec, int (*fp)(ghost_gidx, ghost_lidx, void *, void *), void *arg)
{
    int rank;
    ghost_gidx offset;
    if (vec->context) {
        GHOST_CALL_RETURN(ghost_rank(&rank, vec->context->mpicomm));
        offset = vec->context->lfRow[rank];
    } else {
        rank = 0;
        offset = 0;
    }
    int needInit = 0;
    GHOST_CALL_RETURN(ghost_densemat_cm_malloc(vec,&needInit));
    DEBUG_LOG(1,"Filling vector via function");

    if (vec->traits.location & GHOST_LOCATION_HOST) { // vector is stored on host
        if( needInit ) {
          DENSEMAT_ITER_INIT(vec,fp(offset+row,col,valptr,arg));
        } else {
          DENSEMAT_ITER(vec,fp(offset+row,col,valptr,arg));
        }
        
        // host+device case: uploading will be done in fromVec()
        if (vec->traits.location & GHOST_LOCATION_DEVICE) {
            vec->upload(vec);
        }
    } else {
        INFO_LOG("Need to create dummy HOST densemat!");
        ghost_densemat *hostVec;
        ghost_densemat_traits htraits = vec->traits;
        htraits.location = GHOST_LOCATION_HOST;
        htraits.flags &= (ghost_densemat_flags)~GHOST_DENSEMAT_VIEW;
        GHOST_CALL_RETURN(ghost_densemat_create(&hostVec,vec->context,htraits));
        GHOST_CALL_RETURN(hostVec->fromFunc(hostVec,fp,arg));
        GHOST_CALL_RETURN(vec->fromVec(vec,hostVec,0,0));
        hostVec->destroy(hostVec);
    }

    return GHOST_SUCCESS;
}

static ghost_error ghost_distributeVector(ghost_densemat *vec, ghost_densemat *nodeVec)
{
    DEBUG_LOG(1,"Distributing vector");
    int me;
    int nprocs;
    GHOST_CALL_RETURN(ghost_rank(&me, nodeVec->context->mpicomm));
    GHOST_CALL_RETURN(ghost_nrank(&nprocs, nodeVec->context->mpicomm));
    
    bool uniformstorage;
    GHOST_CALL_RETURN(ghost_densemat_uniformstorage(&uniformstorage,vec));
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
            MPI_CALL_RETURN(MPI_Irecv(DENSEMAT_VALPTR(nodeVec,0,c),nodeVec->context->lnrows[me],mpidt,0,me,nodeVec->context->mpicomm,&req[msgcount]));
            msgcount++;
        }
    } else {
        for (c=0; c<vec->traits.ncols; c++) {
            memcpy(DENSEMAT_VALPTR(nodeVec,0,c),DENSEMAT_VALPTR(vec,0,c),vec->elSize*nodeVec->context->lnrows[0]);
            for (i=1;i<nprocs;i++) {
                MPI_CALL_RETURN(MPI_Isend(DENSEMAT_VALPTR(vec,nodeVec->context->lfRow[i],c),nodeVec->context->lnrows[i],mpidt,i,i,nodeVec->context->mpicomm,&req[msgcount]));
                msgcount++;
            }
        }
    }
    MPI_CALL_RETURN(MPI_Waitall(msgcount,req,stat));
#else

    for (c=0; c<vec->traits.ncols; c++) {
        memcpy(DENSEMAT_VALPTR(nodeVec,0,c),DENSEMAT_VALPTR(vec,0,c),vec->traits.nrows*vec->elSize);
    }
    //    *nodeVec = vec->clone(vec);
#endif

    nodeVec->upload(nodeVec);

    DEBUG_LOG(1,"Vector distributed successfully");

    return GHOST_SUCCESS;
}

static ghost_error ghost_collectVectors(ghost_densemat *vec, ghost_densemat *totalVec) 
{
    ghost_lidx c;
#ifdef GHOST_HAVE_MPI
    int me;
    int nprocs;
    ghost_mpi_datatype mpidt;
    GHOST_CALL_RETURN(ghost_rank(&me, vec->context->mpicomm));
    GHOST_CALL_RETURN(ghost_nrank(&nprocs, vec->context->mpicomm));
    GHOST_CALL_RETURN(ghost_mpi_datatype_get(&mpidt,vec->traits.datatype));

    bool uniformstorage;
    GHOST_CALL_RETURN(ghost_densemat_uniformstorage(&uniformstorage,vec));
    if (!uniformstorage) {
        ERROR_LOG("Cannot collect vectors of different storage order");
        return GHOST_ERR_INVALID_ARG;
    }
//    if (vec->context != NULL)
//        vec->permute(vec,vec->context->invRowPerm); 

    int i;

    MPI_Request req[vec->traits.ncols*2*(nprocs-1)];
    MPI_Status stat[vec->traits.ncols*2*(nprocs-1)];
    int msgcount = 0;

    for (i=0;i<vec->traits.ncols*2*(nprocs-1);i++) {
        req[i] = MPI_REQUEST_NULL;
    }

    if (me != 0) {
        for (c=0; c<vec->traits.ncols; c++) {
            MPI_CALL_RETURN(MPI_Isend(DENSEMAT_VALPTR(vec,0,c),vec->context->lnrows[me],mpidt,0,me,vec->context->mpicomm,&req[msgcount]));
            msgcount++;
        }
    } else {
        for (c=0; c<vec->traits.ncols; c++) {
            memcpy(DENSEMAT_VALPTR(totalVec,0,c),DENSEMAT_VALPTR(vec,0,c),vec->elSize*vec->context->lnrows[0]);
            for (i=1;i<nprocs;i++) {
                MPI_CALL_RETURN(MPI_Irecv(DENSEMAT_VALPTR(totalVec,vec->context->lfRow[i],c),vec->context->lnrows[i],mpidt,i,i,vec->context->mpicomm,&req[msgcount]));
                msgcount++;
            }
        }
    }
    MPI_CALL_RETURN(MPI_Waitall(msgcount,req,stat));
#else
    if (vec->context != NULL) {
//        vec->permute(vec,vec->context->invRowPerm);
        for (c=0; c<vec->traits.ncols; c++) {
            memcpy(DENSEMAT_VALPTR(totalVec,0,c),DENSEMAT_VALPTR(vec,0,c),totalVec->traits.nrows*vec->elSize);
        }
    }
#endif

    return GHOST_SUCCESS;

}

static ghost_error ghost_cloneVector(ghost_densemat *src, ghost_densemat **new, ghost_lidx nr, ghost_lidx roffs, ghost_lidx nc, ghost_lidx coffs)
{
    ghost_densemat_traits newTraits = src->traits;
    newTraits.ncols = nc;
    newTraits.ncolsorig = nc;
    newTraits.nrows = nr;
    newTraits.nrowsorig = nr;
    newTraits.flags &= ~(ghost_densemat_flags)GHOST_DENSEMAT_VIEW;
    newTraits.flags &= ~(ghost_densemat_flags)GHOST_DENSEMAT_SCATTERED;
    ghost_densemat_create(new,src->context,newTraits);

    (*new)->fromVec(*new,src,roffs,coffs);
    return GHOST_SUCCESS;
}

static ghost_error vec_cm_compress(ghost_densemat *vec)
{
    if (!(vec->traits.flags & GHOST_DENSEMAT_SCATTERED)) {
        return GHOST_SUCCESS;
    }

    if (vec->traits.location & GHOST_LOCATION_HOST) {
        ghost_lidx v,i;

        char *val = NULL;
        if (vec->traits.location & GHOST_LOCATION_DEVICE) {
            GHOST_CALL_RETURN(ghost_malloc_pinned((void **)&val,
                        (size_t)vec->traits.ncolspadded*vec->traits.nrowspadded*
                        vec->elSize));
        } else {
            GHOST_CALL_RETURN(ghost_malloc_align((void **)&val,
                        (size_t)vec->traits.ncolspadded*vec->traits.nrowspadded*
                        vec->elSize,GHOST_DATA_ALIGNMENT));
        }

#pragma omp parallel for schedule(runtime) private(v)
        for (i=0; i<vec->traits.nrowspadded; i++)
        {
            for (v=0; v<vec->traits.ncols; v++)
            {
                val[(v*vec->traits.nrowspadded+i)*vec->elSize] = 0;
            }
        }

        
        DENSEMAT_ITER(vec,memcpy(&val[((col)*vec->traits.nrowspadded+(row))*vec->elSize],valptr,vec->elSize));

        vec->val = val;
        
/*        for (v=0; v<vec->traits.ncols; v++)
        {
            memcpy(&val[(v*vec->traits.nrowspadded)*vec->elSize],
                    DENSEMAT_VALPTR(vec,0,v),vec->traits.nrowspadded*vec->elSize);

            if (!(vec->traits.flags & GHOST_DENSEMAT_VIEW)) {
                free(vec->val[v]);
            }
            vec->val[v] = &val[(v*vec->traits.nrowspadded)*vec->elSize];
        }*/
    }
    if (vec->traits.location & GHOST_LOCATION_DEVICE) {
#ifdef GHOST_HAVE_CUDA

        char *cu_val;
        GHOST_CALL_RETURN(ghost_cu_malloc((void **)&cu_val,vec->traits.nrowspadded*vec->traits.ncols*vec->elSize));
        
        DENSEMAT_ITER(vec,ghost_cu_memcpy(&cu_val[(col*vec->traits.nrowspadded+col)*vec->elSize],
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
    vec->traits.ncolsorig = vec->traits.ncols;
    vec->traits.nrowsorig = vec->traits.nrows;
    vec->stride = vec->traits.nrowspadded;

    return GHOST_SUCCESS;
}
    
static ghost_error densemat_cm_halocommInit(ghost_densemat *vec, ghost_densemat_halo_comm *comm)
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
    ghost_error ret = GHOST_SUCCESS;
    int i, to_PE, from_PE, partner;
    int nprocs, me;

    GHOST_CALL_GOTO(ghost_rank(&me, vec->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, vec->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_densemat_halocommInit_common(vec,comm),err,ret);
    
    GHOST_CALL_GOTO(ghost_malloc((void **)&comm->tmprecv,nprocs*sizeof(char *)),err,ret);

    GHOST_CALL_GOTO(ghost_malloc((void **)&comm->tmprecv_mem,vec->traits.ncols*vec->elSize*comm->acc_wishes),err,ret);

    for (from_PE=0; from_PE<nprocs; from_PE++){
        comm->tmprecv[from_PE] = &comm->tmprecv_mem[comm->wishptr[from_PE]*vec->traits.ncols*vec->elSize];
    }
        
    
    if (vec->context->perm_local) {
#ifdef GHOST_HAVE_CUDA
        if (vec->traits.location & GHOST_LOCATION_DEVICE) {
            ghost_densemat_cu_cm_communicationassembly(comm->cu_work,comm->dueptr,comm->acc_dues,vec,vec->context->perm_local->cu_perm);
        } else
#endif
            if (vec->traits.location & GHOST_LOCATION_HOST) {
                ghost_gidx c;
                for (partner = 0; partner<vec->context->nduepartners; partner++) {
                    to_PE = vec->context->duepartners[partner];
#pragma omp parallel for private(c) 
                    for (i=0; i<vec->context->dues[to_PE]; i++){
                        for (c=0; c<vec->traits.ncols; c++) {
                            memcpy(comm->work + (c*vec->context->dues[to_PE]+comm->dueptr[to_PE]*vec->traits.ncols+i)*vec->elSize,
                                    DENSEMAT_VALPTR(vec,vec->context->perm_local->perm[vec->context->duelist[to_PE][i]],c),vec->elSize);
                        }
                    }
                }
            }
    } else {
#ifdef GHOST_HAVE_CUDA
        if (vec->traits.location & GHOST_LOCATION_DEVICE) {
            ghost_densemat_cu_cm_communicationassembly(comm->cu_work,comm->dueptr,comm->acc_dues,vec,NULL);
        } else
#endif
            if (vec->traits.location & GHOST_LOCATION_HOST) {
                ghost_gidx c;
                for (partner = 0; partner<vec->context->nduepartners; partner++) {
                    to_PE = vec->context->duepartners[partner];
#pragma omp parallel for private(c) 
                    for (i=0; i<vec->context->dues[to_PE]; i++){
                        for (c=0; c<vec->traits.ncols; c++) {
                            memcpy(comm->work + (c*vec->context->dues[to_PE]+comm->dueptr[to_PE]*vec->traits.ncols+i)*vec->elSize,
                                    DENSEMAT_VALPTR(vec,vec->context->duelist[to_PE][i],c),vec->elSize);
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

static ghost_error densemat_cm_halocommFinalize(ghost_densemat *vec, ghost_densemat_halo_comm *comm)
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
    ghost_error ret = GHOST_SUCCESS;
    
    int nprocs;
    int i, from_PE;
    
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, vec->context->mpicomm),err,ret);

    ghost_densemat_halocommFinalize_common(comm);
    if (vec->traits.location & GHOST_LOCATION_HOST) {
        GHOST_INSTR_START("Assemble row-major view");
        for (from_PE=0; from_PE<nprocs; from_PE++){
            for (i=0; i<vec->traits.ncols; i++){
                memcpy(DENSEMAT_VALPTR(vec,vec->context->hput_pos[from_PE],i),&comm->tmprecv[from_PE][(i*vec->context->wishes[from_PE])*vec->elSize],vec->elSize*vec->context->wishes[from_PE]);
            }
        }
        GHOST_INSTR_STOP("Assemble row-major view");
    }

#ifdef GHOST_HAVE_CUDA 
    GHOST_INSTR_START("upload")
    if (vec->traits.location & GHOST_LOCATION_DEVICE) {
#ifdef GHOST_TRACK_DATATRANSFERS
        ghost_datatransfer_register("spmv_halo",GHOST_DATATRANSFER_OUT,GHOST_DATATRANSFER_RANK_GPU,vec->context->halo_elements*vec->traits.ncols*vec->elSize);
#endif
        for (from_PE=0; from_PE<nprocs; from_PE++){
            ghost_cu_upload2d(DENSEMAT_CUVALPTR(vec,vec->context->hput_pos[from_PE],0),vec->stride*vec->elSize,comm->tmprecv[from_PE],vec->context->wishes[from_PE]*vec->elSize,vec->context->wishes[from_PE]*vec->elSize,vec->traits.ncols);
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
