#define _XOPEN_SOURCE 500 
#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/core.h"
#include "ghost/densemat.h"
#include "ghost/densemat_cm.h"
#include "ghost/densemat_rm.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include "ghost/context.h"
#include "ghost/instr.h"
#include "ghost/machine.h"
#include "ghost/log.h"
#include "ghost/bindensemat.h"
#include "ghost/sell.h"

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

const ghost_densemat_traits_t GHOST_DENSEMAT_TRAITS_INITIALIZER = {
    .nrows = 0,
    .nrowsorig = 0,
    .nrowshalo = 0,
    .nrowspadded = 0,
    .ncols = 1,
    .ncolsorig = 0,
    .ncolspadded = 0,
    .flags = GHOST_DENSEMAT_DEFAULT,
    .storage = GHOST_DENSEMAT_COLMAJOR,
    .datatype = (ghost_datatype_t)(GHOST_DT_DOUBLE|GHOST_DT_REAL)
};

const ghost_densemat_halo_comm_t GHOST_DENSEMAT_HALO_COMM_INITIALIZER = {
#ifdef GHOST_HAVE_MPI
    .msgcount = 0,
    .request = NULL,
    .status = NULL,
    .tmprecv = NULL,
    .tmprecv_mem = NULL,
    .work = NULL,
    .dueptr = NULL,
    .wishptr = NULL,
    .acc_dues = 0,
    .acc_wishes = 0,
    .cu_work = NULL
#endif
};


static ghost_error_t getNrowsFromContext(ghost_densemat_t *vec);

ghost_error_t ghost_densemat_create(ghost_densemat_t **vec, ghost_context_t *ctx, ghost_densemat_traits_t traits)
{
    ghost_error_t ret = GHOST_SUCCESS;
    GHOST_CALL_GOTO(ghost_malloc((void **)vec,sizeof(ghost_densemat_t)),err,ret);
    (*vec)->context = ctx;
    (*vec)->traits = traits;
    (*vec)->ldmask = ghost_bitmap_alloc();
    (*vec)->trmask = ghost_bitmap_alloc();
    (*vec)->val = NULL;
    (*vec)->cu_val = NULL;
    (*vec)->viewing = NULL;
    if (!(*vec)->ldmask) {
        ERROR_LOG("Could not create dense matrix mask!");
        goto err;
    }

    GHOST_CALL_GOTO(ghost_datatype_size(&(*vec)->elSize,(*vec)->traits.datatype),err,ret);
    getNrowsFromContext((*vec));

    DEBUG_LOG(1,"Initializing vector");

    if (!((*vec)->traits.flags & (GHOST_DENSEMAT_HOST | GHOST_DENSEMAT_DEVICE)))
    { // no placement specified
        DEBUG_LOG(2,"Setting vector placement");
        (*vec)->traits.flags |= (ghost_densemat_flags_t)GHOST_DENSEMAT_HOST;
        ghost_type_t ghost_type;
        GHOST_CALL_RETURN(ghost_type_get(&ghost_type));
        if (ghost_type == GHOST_TYPE_CUDA) {
            (*vec)->traits.flags |= (ghost_densemat_flags_t)GHOST_DENSEMAT_DEVICE;
        }
    }

    if ((*vec)->traits.storage == GHOST_DENSEMAT_ROWMAJOR) {
        ghost_bitmap_set_range((*vec)->ldmask,0,(*vec)->traits.ncolsorig-1);
        ghost_bitmap_set_range((*vec)->trmask,0,(*vec)->traits.nrowsorig-1);
        ghost_densemat_rm_setfuncs(*vec);
        (*vec)->stride = &(*vec)->traits.ncolspadded;
    } else {
        ghost_bitmap_set_range((*vec)->ldmask,0,(*vec)->traits.nrowsorig-1);
        ghost_bitmap_set_range((*vec)->trmask,0,(*vec)->traits.ncolsorig-1);
        ghost_densemat_cm_setfuncs(*vec);
        (*vec)->stride = &(*vec)->traits.nrowspadded;
    }
#ifdef GHOST_HAVE_MPI
    ghost_mpi_datatype_t dt;
    ghost_mpi_datatype(&dt,(*vec)->traits.datatype);

    MPI_CALL_RETURN(MPI_Type_contiguous((*vec)->traits.ncols,dt,&(*vec)->row_mpidt));
    MPI_CALL_RETURN(MPI_Type_commit(&(*vec)->row_mpidt));
#else
    (*vec)->row_mpidt = MPI_DATATYPE_NULL;
#endif


    goto out;
err:
    free(*vec); *vec = NULL;

out:
    return ret;
}

static ghost_error_t getNrowsFromContext(ghost_densemat_t *vec)
{
    DEBUG_LOG(1,"Computing the number of vector rows from the context");

    if (vec->context != NULL) {
        if (vec->traits.nrows == 0) {
            DEBUG_LOG(2,"nrows for vector not given. determining it from the context");
            if ((vec->context->flags & GHOST_CONTEXT_REDUNDANT) || (vec->traits.flags & GHOST_DENSEMAT_GLOBAL))
            {
                if (vec->traits.flags & GHOST_DENSEMAT_NO_HALO) {
                    vec->traits.nrows = vec->context->gnrows;
                } else {
                    vec->traits.nrows = vec->context->gncols;
                }

            } 
            else 
            {
                int rank;
                GHOST_CALL_RETURN(ghost_rank(&rank, vec->context->mpicomm));
                vec->traits.nrows = vec->context->lnrows[rank];
            }
        }
        if (vec->traits.nrowshalo == 0) {
            DEBUG_LOG(2,"nrowshalo for vector not given. determining it from the context");
            if ((vec->context->flags & GHOST_CONTEXT_REDUNDANT) || (vec->traits.flags & GHOST_DENSEMAT_GLOBAL))
            {
                vec->traits.nrowshalo = vec->traits.nrows;
            } 
            else 
            {
                if (!(vec->traits.flags & GHOST_DENSEMAT_GLOBAL) && !(vec->traits.flags & GHOST_DENSEMAT_NO_HALO)) {
                    if (vec->context->halo_elements == -1) {
                        ERROR_LOG("You have to make sure to read in the matrix _before_ creating the right hand side vector in a distributed context! This is because we have to know the number of halo elements of the vector.");
                        return GHOST_ERR_UNKNOWN;
                    }
                    vec->traits.nrowshalo = vec->traits.nrows+vec->context->halo_elements+1;
                } else {
                    // context->hput_pos[0] = nrows if only one process, so we need a dummy element 
                    vec->traits.nrowshalo = vec->traits.nrows+1; 
                }
            }    
        }
    } else {
        // the case context==NULL is allowed - the vector is local.
        DEBUG_LOG(1,"The vector's context is NULL.");
    }

    if (vec->traits.nrowspadded == 0) {
        if (vec->traits.flags & GHOST_DENSEMAT_VIEW) {
            INFO_LOG("No padding for view!");
            vec->traits.nrowspadded = vec->traits.nrows;
        } else {
            DEBUG_LOG(2,"nrowspadded for vector not given. determining it from the context");
            ghost_lidx_t padding = vec->elSize;
            if (vec->traits.nrows > 1) {
#ifdef GHOST_HAVE_MIC
                padding = 64; // 64 byte padding
#elif defined(GHOST_HAVE_AVX)
                padding = 32; // 32 byte padding
                if (vec->traits.nrows <= 2) {
                    PERFWARNING_LOG("Force SSE over AVX vor densemat with less than 2 rows!");
                    padding = 16; // SSE in this case: only 16 byte alignment required
                }
#elif defined (GHOST_HAVE_SSE)
                padding = 16; // 16 byte padding
#endif
            }
            
            padding /= vec->elSize;
            padding = MAX(padding,ghost_sell_max_cfg_chunkheight());
            
            vec->traits.nrowspadded = PAD(MAX(vec->traits.nrowshalo,vec->traits.nrows),padding);
        }
    }
    if (vec->traits.ncolspadded == 0) {
        if (vec->traits.flags & GHOST_DENSEMAT_VIEW) {
            INFO_LOG("No padding for view!");
            vec->traits.ncolspadded = vec->traits.ncols;
        } else {
            DEBUG_LOG(2,"ncolspadded for vector not given. determining it from the context");
            ghost_lidx_t padding = vec->elSize;
            if (vec->traits.ncols > 1) {
#ifdef GHOST_HAVE_MIC
                padding = 64; // 64 byte padding
#elif defined(GHOST_HAVE_AVX)
                padding = 32; // 32 byte padding
                if (vec->traits.ncols <= 2) {
                    PERFWARNING_LOG("Force SSE over AVX vor densemat with less than 2 columns!");
                    padding = 16; // SSE in this case: only 16 byte alignment required
                }
#elif defined (GHOST_HAVE_SSE)
                padding = 16; // 16 byte padding
#endif
            }
            padding /= vec->elSize;
            vec->traits.ncolspadded = PAD(vec->traits.ncols,padding);
            if (vec->traits.ncols % padding) {
                INFO_LOG("Cols will be padded to a multiple of %"PRLIDX" to %"PRLIDX,padding,vec->traits.ncolspadded);
            }
        }
    }
    if (vec->traits.ncolsorig == 0) {
        vec->traits.ncolsorig = vec->traits.ncols;
    }
    if (vec->traits.nrowsorig == 0) {
        vec->traits.nrowsorig = vec->traits.nrows;
    }

    DEBUG_LOG(1,"The vector has %"PRLIDX" w/ %"PRLIDX" halo elements (padded: %"PRLIDX") rows",
            vec->traits.nrows,vec->traits.nrowshalo-vec->traits.nrows,vec->traits.nrowspadded);
    return GHOST_SUCCESS; 
}

ghost_error_t ghost_densemat_valptr(ghost_densemat_t *vec, void **ptr)
{
    if (!ptr) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }

    if (vec->traits.nrows < 1) {
        ERROR_LOG("No rows");
        return GHOST_ERR_INVALID_ARG;
    }
    if (vec->traits.ncols < 1) {
        ERROR_LOG("No columns");
        return GHOST_ERR_INVALID_ARG;
    }

    if (ghost_bitmap_iszero(vec->ldmask)) {
        ERROR_LOG("Everything masked out. This is a zero-view.");
        return GHOST_ERR_INVALID_ARG;
    }

    *ptr = &vec->val[0][ghost_bitmap_first(vec->ldmask)*vec->elSize];

    return GHOST_SUCCESS;


}

ghost_error_t ghost_densemat_cu_valptr(ghost_densemat_t *vec, void **ptr)
{
    if (!ptr) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }

    if (vec->traits.nrows < 1) {
        ERROR_LOG("No rows");
        return GHOST_ERR_INVALID_ARG;
    }
    if (vec->traits.ncols < 1) {
        ERROR_LOG("No columns");
        return GHOST_ERR_INVALID_ARG;
    }

    if (ghost_bitmap_iszero(vec->ldmask)) {
        ERROR_LOG("Everything masked out. This is a zero-view.");
        return GHOST_ERR_INVALID_ARG;
    }

#ifdef GHOST_HAVE_CUDA
    *ptr = &vec->cu_val[(ghost_bitmap_first(vec->trmask)*(*(vec->stride))+ghost_bitmap_first(vec->ldmask))*vec->elSize];
#else
    *ptr = NULL;
#endif

    return GHOST_SUCCESS;
}

ghost_error_t ghost_densemat_mask2charfield(ghost_bitmap_t mask, ghost_lidx_t len, char *charfield)
{
    unsigned int i;
    memset(charfield,0,len);
    for (i=0; i<(unsigned int)len; i++) {
        if(ghost_bitmap_isset(mask,i)) {
            charfield[i] = 1;
        }
    }

    return GHOST_SUCCESS;
}

bool array_strictly_ascending (ghost_lidx_t *coffs, ghost_lidx_t nc)
{
    ghost_lidx_t i;

    for (i=1; i<nc; i++) {
        if (coffs[i] <= coffs[i-1]) {
            return 0;
        }
    }
    return 1;
}

ghost_error_t ghost_densemat_uniformstorage(bool *uniform, ghost_densemat_t *vec)
{
#ifndef GHOST_HAVE_MPI
    UNUSED(vec);
    *uniform = true;
#else
    int nprocs;
    int allstorages = (int)vec->traits.storage;
    GHOST_CALL_RETURN(ghost_nrank(&nprocs, vec->context->mpicomm));
    
    MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE,&allstorages,1,MPI_INT,MPI_SUM,vec->context->mpicomm));
    *uniform = ((int)vec->traits.storage * nprocs == allstorages);
#endif
    return GHOST_SUCCESS;;
}

char * ghost_densemat_storage_string(ghost_densemat_t *densemat)
{
    switch(densemat->traits.storage) {
        case GHOST_DENSEMAT_ROWMAJOR:
            return "Row-major";
        case GHOST_DENSEMAT_COLMAJOR:
            return "Col-major";
        default:
            return "Invalid";
    }
}
   
static void charfield2string(char *str, char *cf, int len) {
    int i;
    for (i=0; i<len; i++) {
        if (cf[i]) {
            str[i] = 'x';
        } else {
            str[i] = '.';
        }
    }
    str[len]='\0';
}

ghost_error_t ghost_densemat_info_string(char **str, ghost_densemat_t *densemat)
{
    int myrank;
    int mynoderank;
    ghost_mpi_comm_t nodecomm;
    
    GHOST_CALL_RETURN(ghost_nodecomm_get(&nodecomm));
    GHOST_CALL_RETURN(ghost_rank(&myrank, MPI_COMM_WORLD));
    GHOST_CALL_RETURN(ghost_rank(&mynoderank, nodecomm));
    GHOST_CALL_RETURN(ghost_malloc((void **)str,1));
    memset(*str,'\0',1);
    
    ghost_header_string(str,"Dense matrix @ local rank %d (glob %d)",mynoderank,myrank);
    ghost_line_string(str,"Dimension",NULL,"%"PRLIDX"x%"PRLIDX,densemat->traits.nrows,densemat->traits.ncols);
    ghost_line_string(str,"Padded dimension",NULL,"%"PRLIDX"x%"PRLIDX,densemat->traits.nrowspadded,densemat->traits.ncolspadded);
    ghost_line_string(str,"View",NULL,"%s",densemat->traits.flags&GHOST_DENSEMAT_VIEW?"Yes":"No");
    if (densemat->traits.flags&GHOST_DENSEMAT_VIEW) {
        ghost_line_string(str,"Dimension of viewed densemat",NULL,"%"PRLIDX"x%"PRLIDX,densemat->traits.nrowsorig,densemat->traits.ncolsorig);
        char colmask[densemat->traits.ncolsorig];
        char colmaskstr[densemat->traits.ncolsorig+1];
        ghost_densemat_mask2charfield((densemat->traits.storage==GHOST_DENSEMAT_ROWMAJOR)?densemat->ldmask:densemat->trmask,densemat->traits.ncolsorig,colmask);
        charfield2string(colmaskstr,colmask,densemat->traits.ncolsorig);
        ghost_line_string(str,"Viewed columns",NULL,"%s",colmaskstr);
        char rowmask[densemat->traits.nrowsorig];
        char rowmaskstr[densemat->traits.nrowsorig+1];
        ghost_densemat_mask2charfield(densemat->traits.storage==GHOST_DENSEMAT_ROWMAJOR?densemat->trmask:densemat->ldmask,densemat->traits.nrowsorig,rowmask);
        charfield2string(rowmaskstr,rowmask,densemat->traits.nrowsorig);
        ghost_line_string(str,"Viewed rows",NULL,"%s",rowmaskstr);

    }
   
    ghost_line_string(str,"Location",NULL,"%s",densemat->traits.storage&GHOST_DENSEMAT_DEVICE?densemat->traits.flags&GHOST_DENSEMAT_HOST?"Device+Host":"Device":"Host");
    ghost_line_string(str,"Storage order",NULL,"%s",ghost_densemat_storage_string(densemat));
    ghost_footer_string(str);
    
    return GHOST_SUCCESS;

}

ghost_error_t ghost_densemat_halocommInit_common(ghost_densemat_t *vec, ghost_densemat_halo_comm_t *comm) 
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
    int nprocs;
    int me; 
    int i;
    ghost_error_t ret = GHOST_SUCCESS;
    int rowsize;

    if (vec->traits.flags & GHOST_DENSEMAT_NO_HALO) {
        ERROR_LOG("The densemat has no halo buffer!");
        return GHOST_ERR_INVALID_ARG;
    }
    if (vec->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        ERROR_LOG("Halo communication for scattered densemats not yet supported!");
        ret = GHOST_ERR_NOT_IMPLEMENTED;
        goto err;
    }


    GHOST_CALL_GOTO(ghost_rank(&me, vec->context->mpicomm),err,ret);
    MPI_CALL_GOTO(MPI_Type_size(vec->row_mpidt,&rowsize),err,ret);
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, vec->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_rank(&me, vec->context->mpicomm),err,ret);
    
    comm->msgcount = 0;
    GHOST_CALL_GOTO(ghost_malloc((void **)&comm->wishptr,(nprocs+1)*sizeof(ghost_lidx_t)),err,ret);

    int nMsgsOverall = 0;

    comm->wishptr[0] = 0;
    for (i=0;i<nprocs;i++) {
        comm->wishptr[i+1] = comm->wishptr[i]+vec->context->wishes[i];
        if (vec->context->wishes[i]) {
            nMsgsOverall += ((size_t)rowsize*vec->context->wishes[i])/INT_MAX + 1;
        }
        if (vec->context->dues[i]) {
            nMsgsOverall += ((size_t)rowsize*vec->context->dues[i])/INT_MAX + 1;
        }
    }
    comm->acc_wishes = comm->wishptr[nprocs];

    GHOST_CALL_GOTO(ghost_malloc((void **)&comm->request,sizeof(MPI_Request)*nMsgsOverall),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&comm->status,sizeof(MPI_Status)*nMsgsOverall),err,ret);

    for (i=0;i<nMsgsOverall;i++) {
        comm->request[i] = MPI_REQUEST_NULL;
    }
    

    GHOST_CALL_RETURN(ghost_malloc((void **)&comm->dueptr,(nprocs+1)*sizeof(ghost_lidx_t)));

    comm->dueptr[0] = 0;
    for (i=0;i<nprocs;i++) {
        comm->dueptr[i+1] = comm->dueptr[i]+vec->context->dues[i];
    }
    comm->acc_dues = comm->dueptr[nprocs];

    if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
#ifdef GHOST_HAVE_CUDA
        CUDA_CALL_RETURN(cudaHostAlloc((void **)&comm->work,(size_t)vec->traits.ncols*comm->acc_dues*vec->elSize,cudaHostAllocDefault));
#endif
    } else {
        GHOST_CALL_RETURN(ghost_malloc((void **)&comm->work,(size_t)vec->traits.ncols*comm->acc_dues*vec->elSize));
    }

#ifdef GHOST_HAVE_CUDA
    if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
        GHOST_CALL_GOTO(ghost_cu_malloc(&comm->cu_work,vec->traits.ncols*comm->acc_dues*vec->elSize),err,ret);
    }
#endif

#ifdef GHOST_HAVE_CUDA
    if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
        GHOST_INSTR_START("downloadcomm->work");
#ifdef GHOST_HAVE_TRACK_DATATRANSFERS
        ghost_datatransfer_register("spmv_halo",GHOST_DATATRANSFER_IN,GHOST_DATATRANSFER_RANK_GPU,vec->traits.ncols*comm->acc_dues*vec->elSize);

#endif
        ghost_cu_download(comm->work,comm->cu_work,vec->traits.ncols*comm->acc_dues*vec->elSize);
        GHOST_INSTR_STOP("downloadcomm->work");
    }
#endif

    goto out;
err:
    ERROR_LOG("Error in function!");
    return GHOST_ERR_UNKNOWN;

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_COMMUNICATION);
    return ret;
#else
    UNUSED(vec);
    UNUSED(comm);
    return GHOST_ERR_NOT_IMPLEMENTED;
#endif


}

ghost_error_t ghost_densemat_halocommStart_common(ghost_densemat_t *vec, ghost_densemat_halo_comm_t *comm)
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION)
    ghost_error_t ret = GHOST_SUCCESS;
    char *recv;
    int from_PE, to_PE;
    int nprocs;
    int rowsize;
    int me; 
    GHOST_CALL_GOTO(ghost_rank(&me, vec->context->mpicomm),err,ret);
    MPI_CALL_GOTO(MPI_Type_size(vec->row_mpidt,&rowsize),err,ret);
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, vec->context->mpicomm),err,ret);

    for (from_PE=0; from_PE<nprocs; from_PE++){
        if (vec->context->wishes[from_PE]>0) {
            recv = comm->tmprecv[from_PE];

#ifdef GHOST_HAVE_TRACK_DATATRANSFERS
            ghost_datatransfer_register("spmv_halo",GHOST_DATATRANSFER_IN,from_PE,vec->context->wishes[from_PE]*vec->elSize*vec->traits.ncols);
#endif
            int msg;
            int nmsgs = (size_t)rowsize*vec->context->wishes[from_PE]/INT_MAX + 1;
            size_t msgSizeRows = vec->context->wishes[from_PE]/nmsgs;

            for (msg = 0; msg < nmsgs-1; msg++) {
                MPI_CALL_GOTO(MPI_Irecv(recv + msg*msgSizeRows*rowsize, msgSizeRows, vec->row_mpidt, from_PE, from_PE, vec->context->mpicomm,&comm->request[comm->msgcount]),err,ret);
                comm->msgcount++;
            }

            // remainder
            MPI_CALL_GOTO(MPI_Irecv(recv + msg*msgSizeRows*rowsize, vec->context->wishes[from_PE] - msg*msgSizeRows, vec->row_mpidt, from_PE, from_PE, vec->context->mpicomm,&comm->request[comm->msgcount]),err,ret);
            comm->msgcount++;
        }
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
                MPI_CALL_GOTO(MPI_Isend(comm->work + comm->dueptr[to_PE]*vec->elSize*vec->traits.ncols+msg*msgSizeRows*rowsize, msgSizeRows, vec->row_mpidt, to_PE, me, vec->context->mpicomm, &comm->request[comm->msgcount]),err,ret);
                comm->msgcount++;
            }

            // remainder
            MPI_CALL_GOTO(MPI_Isend(comm->work + comm->dueptr[to_PE]*vec->elSize*vec->traits.ncols+msg*msgSizeRows*rowsize, vec->context->dues[to_PE] - msg*msgSizeRows, vec->row_mpidt, to_PE, me, vec->context->mpicomm, &comm->request[comm->msgcount]),err,ret);
            comm->msgcount++;
        }
    }


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

ghost_error_t ghost_densemat_halocommFinalize_common(ghost_densemat_halo_comm_t *comm)
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
    ghost_error_t ret = GHOST_SUCCESS;

    GHOST_INSTR_START("waitall");
    MPI_CALL_GOTO(MPI_Waitall(comm->msgcount, comm->request, comm->status),err,ret);
    GHOST_INSTR_STOP("waitall");

    goto out;
err:

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_COMMUNICATION);
    return ret;

#else
    UNUSED(comm);
    return GHOST_ERR_NOT_IMPLEMENTED;
#endif
}
