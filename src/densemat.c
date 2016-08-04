#define _XOPEN_SOURCE 500 
#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/core.h"
#include "ghost/datatransfers.h"
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
#include "ghost/constants.h"
#include "ghost/datatransfers.h"

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

const ghost_densemat_traits GHOST_DENSEMAT_TRAITS_INITIALIZER = {
    .nrows = 0,
    .nrowsorig = 0,
    .nrowshalo = 0,
    .nrowspadded = 0,
    .gnrows = 0,
    .goffs = 0,
    .ncols = 1,
    .ncolsorig = 0,
    .ncolspadded = 0,
    .flags = GHOST_DENSEMAT_DEFAULT,
    .storage = GHOST_DENSEMAT_STORAGE_DEFAULT,
    .location = GHOST_LOCATION_DEFAULT,
    .datatype = (ghost_datatype)(GHOST_DT_DOUBLE|GHOST_DT_REAL),
    .permutemethod = NONE
};

const ghost_densemat_halo_comm GHOST_DENSEMAT_HALO_COMM_INITIALIZER = {
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


static ghost_error getNrowsFromContext(ghost_densemat *vec, ghost_context *ctx);

ghost_error ghost_densemat_create(ghost_densemat **vec, ghost_context *ctx, ghost_densemat_traits traits)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_SETUP);
    ghost_error ret = GHOST_SUCCESS;
    GHOST_CALL_GOTO(ghost_malloc((void **)vec,sizeof(ghost_densemat)),err,ret);
    (*vec)->traits = traits;
    (*vec)->context = ctx;
    (*vec)->colmask = NULL;
    (*vec)->rowmask = NULL;
    (*vec)->val = NULL;
    (*vec)->cu_val = NULL;
   
    if (ctx) {
        if (ctx->perm_global || ctx->perm_local) {
           // (*vec)->traits.flags |= (ghost_densemat_flags)GHOST_DENSEMAT_PERMUTED;//why this??
        }
        if ((*vec)->traits.gnrows == 0) {
            (*vec)->traits.gnrows = ctx->gnrows;
        }
        if ((*vec)->traits.goffs == 0) {
            int me;
            ghost_rank(&me,ctx->mpicomm);
            (*vec)->traits.goffs = ctx->lfRow[me];
        }
    }

    if (!((*vec)->traits.flags & GHOST_DENSEMAT_VIEW)) {
        (*vec)->src = *vec;
    } else {
        (*vec)->src = NULL;
    }


   if(ctx==NULL || ctx->perm_local == NULL) {
		      (*vec)->perm_local = NULL;
    } else if((*vec)->traits.permutemethod == COLUMN) {
	        GHOST_CALL_GOTO(ghost_malloc((void **)&((*vec)->perm_local),sizeof(ghost_densemat_permutation)),err,ret); 
      		(*vec)->perm_local->perm    = ctx->perm_local->colPerm;
		      (*vec)->perm_local->invPerm = ctx->perm_local->colInvPerm;
    } else {
	        GHOST_CALL_GOTO(ghost_malloc((void **)&((*vec)->perm_local),sizeof(ghost_densemat_permutation)),err,ret); 
		      (*vec)->perm_local->perm    = ctx->perm_local->perm;
		      (*vec)->perm_local->invPerm = ctx->perm_local->invPerm;
    }

    //Right now there are no Global row and column permutation, once there, modify this
    if(ctx==NULL || ctx->perm_global == NULL) {
		      (*vec)->perm_global = NULL;
    } else {
	        GHOST_CALL_GOTO(ghost_malloc((void **)&((*vec)->perm_global),sizeof(ghost_densemat_permutation)),err,ret); 
		      (*vec)->perm_global->perm    = ctx->perm_global->colPerm;
		      (*vec)->perm_global->invPerm = ctx->perm_global->colInvPerm;
    }


    GHOST_CALL_GOTO(ghost_datatype_size(&(*vec)->elSize,(*vec)->traits.datatype),err,ret);
    getNrowsFromContext((*vec),ctx);

    DEBUG_LOG(1,"Initializing vector");

    if ((*vec)->traits.location == GHOST_LOCATION_DEFAULT) { // no placement specified
        ghost_type ghost_type;
        GHOST_CALL_RETURN(ghost_type_get(&ghost_type));
        if (ghost_type == GHOST_TYPE_CUDA) {
            DEBUG_LOG(1,"Auto-place on device");
            (*vec)->traits.location = GHOST_LOCATION_DEVICE;
        } else {
            DEBUG_LOG(1,"Auto-place on host");
            (*vec)->traits.location = GHOST_LOCATION_HOST;
        }

    } else {
        DEBUG_LOG(1,"Placement given: %s",ghost_location_string((*vec)->traits.location));
    }

    if ((*vec)->traits.storage == GHOST_DENSEMAT_STORAGE_DEFAULT) {
        if ((*vec)->traits.ncols > 1) {
            DEBUG_LOG(1,"Setting densemat storage to row-major!");
            (*vec)->traits.storage = GHOST_DENSEMAT_ROWMAJOR;
        } else {
            DEBUG_LOG(1,"Setting densemat storage to col-major!");
            (*vec)->traits.storage = GHOST_DENSEMAT_COLMAJOR;
        }
    }
    if ((*vec)->traits.storage == GHOST_DENSEMAT_ROWMAJOR) {
        ghost_densemat_rm_setfuncs(*vec);
        (*vec)->stride = (*vec)->traits.ncolspadded;
        (*vec)->nblock = (*vec)->traits.nrows;
        (*vec)->blocklen = (*vec)->traits.ncols;
    } else {
        ghost_densemat_cm_setfuncs(*vec);
        (*vec)->stride = (*vec)->traits.nrowshalopadded;
        (*vec)->nblock = (*vec)->traits.ncols;
        (*vec)->blocklen = (*vec)->traits.nrows;
    }
#ifdef GHOST_HAVE_MPI
    GHOST_CALL_RETURN(ghost_mpi_datatype_get(&(*vec)->mpidt,(*vec)->traits.datatype));
    MPI_CALL_RETURN(MPI_Type_vector((*vec)->nblock,(*vec)->blocklen,(*vec)->stride,(*vec)->mpidt,&((*vec)->fullmpidt)));
    MPI_CALL_RETURN(MPI_Type_commit(&((*vec)->fullmpidt)));
#else
    (*vec)->mpidt = MPI_DATATYPE_NULL;
    (*vec)->fullmpidt = MPI_DATATYPE_NULL;
#endif

//    char *str;
//    ghost_densemat_info_string(&str,*vec);
//    printf("%s\n",str);
    goto out;
err:
    free(*vec); *vec = NULL;

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_SETUP);
    return ret;
}

static ghost_error getNrowsFromContext(ghost_densemat *vec, ghost_context *ctx)
{
    DEBUG_LOG(1,"Computing the number of vector rows from the context");
    
    if (ctx != NULL) {
        int rank;
        GHOST_CALL_RETURN(ghost_rank(&rank, ctx->mpicomm));
        //make distinction between Left and Right side vectors (since now we have rectangular matrix)
        if(ctx->flags & GHOST_PERM_NO_DISTINCTION && vec->traits.permutemethod==COLUMN) {
		      vec->traits.nrows = ctx->nrowspadded; 
        } else if(ctx->flags & GHOST_PERM_NO_DISTINCTION && vec->traits.permutemethod==ROW) {
          vec->traits.nrows = ctx->lnrows[rank];
        } else {
        	vec->traits.nrows = ctx->lnrows[rank];
	      }
    }

    if (vec->traits.flags & GHOST_DENSEMAT_VIEW) {
        //INFO_LOG("No padding for view!");
        // already copied!
        //vec->traits.nrowspadded = vec->traits.nrows;
        //vec->traits.ncolspadded = vec->traits.ncols;
    } else {
        if (vec->traits.flags & GHOST_DENSEMAT_PAD_COLS) {
            ghost_lidx padding = vec->elSize;
            if (vec->traits.nrows > 1) {
#ifdef GHOST_BUILD_MIC
                padding = 64; // 64 byte padding
#elif defined(GHOST_BUILD_AVX) || defined(GHOST_BUILD_AVX2)
                padding = 32; // 32 byte padding
                if (vec->traits.ncols == 2) {
                    padding = 16; // SSE in this case: only 16 byte alignment required
                }
                if (vec->traits.ncols == 1) {
                    padding = vec->elSize; // (pseudo-) row-major: no padding
                }
#elif defined (GHOST_BUILD_SSE)
                padding = 16; // 16 byte padding
                if (vec->traits.ncols == 1) {
                    padding = vec->elSize; // (pseudo-) row-major: no padding
                }
#endif
            }
           
            padding /= vec->elSize;
            

            vec->traits.ncolspadded = PAD(vec->traits.ncols,padding);
        } else {
            vec->traits.ncolspadded = vec->traits.ncols;
        }
      
        ghost_lidx padding = ghost_densemat_row_padding(); 

#ifdef GHOST_BUILD_MIC
        //WARNING_LOG("Extremely large row padding because the performance for TSMM and a large dimension power of two is very bad. This has to be fixed!");
       // padding=500000; 
#endif
 
        vec->traits.nrowspadded = PAD(vec->traits.nrows,padding);
    }
      
    if (vec->traits.ncolsorig == 0) {
        vec->traits.ncolsorig = vec->traits.ncols;
    }
    if (vec->traits.nrowsorig == 0) {
        vec->traits.nrowsorig = vec->traits.nrows;
    }

    if (ctx != NULL) {
/*        int rank;
        GHOST_CALL_RETURN(ghost_rank(&rank, ctx->mpicomm)); 
        if(ctx->flags & GHOST_PERM_NO_DISTINCTION) {
	      	vec->traits.nrows = ctx->nrowspadded;
      	} else {
        	vec->traits.nrows = ctx->lnrows[rank];
	      }
*/
	 if (!(vec->traits.flags & GHOST_DENSEMAT_NO_HALO)) {
            if (ctx->halo_elements == -1) {
                ERROR_LOG("You have to make sure to read in the matrix _before_ creating the right hand side vector in a distributed context! This is because we have to know the number of halo elements of the vector.");
                return GHOST_ERR_UNKNOWN;
            }
            vec->traits.nrowshalo = vec->traits.nrowspadded+ctx->halo_elements;
        } else {
            vec->traits.nrowshalo = vec->traits.nrowspadded;
        }
    } else {
        // context->hput_pos[0] = nrows if only one process, so we need a dummy element 
        vec->traits.nrowshalo = vec->traits.nrowspadded; 
    }
 
    vec->traits.nrowshalopadded = PAD(vec->traits.nrowshalo,ghost_machine_simd_width()/4);
    
    DEBUG_LOG(1,"The vector has %"PRLIDX" w/ %"PRLIDX" halo elements (padded: %"PRLIDX") rows",
            vec->traits.nrows,vec->traits.nrowshalo-vec->traits.nrows,vec->traits.nrowspadded);
    return GHOST_SUCCESS; 
}

ghost_error ghost_densemat_mask2charfield(ghost_bitmap mask, ghost_lidx len, char *charfield)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    unsigned int i;
    memset(charfield,0,len);
    for (i=0; i<(unsigned int)len; i++) {
        if(ghost_bitmap_isset(mask,i)) {
            charfield[i] = 1;
        }
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}

bool array_strictly_ascending (ghost_lidx *coffs, ghost_lidx nc)
{
    ghost_lidx i;

    for (i=1; i<nc; i++) {
        if (coffs[i] <= coffs[i-1]) {
            return 0;
        }
    }
    return 1;
}

ghost_error ghost_densemat_uniformstorage(bool *uniform, ghost_densemat *vec, ghost_mpi_comm mpicomm)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
#ifndef GHOST_HAVE_MPI
    UNUSED(vec);
    *uniform = true;
#else
    int nprocs;
    int allstorages = (int)vec->traits.storage;
    GHOST_CALL_RETURN(ghost_nrank(&nprocs, mpicomm));
    
    MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE,&allstorages,1,MPI_INT,MPI_SUM,mpicomm));
    *uniform = ((int)vec->traits.storage * nprocs == allstorages);
#endif
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;;
}

char * ghost_densemat_storage_string(ghost_densemat_storage storage)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    char *ret;
    switch(storage) {
        case GHOST_DENSEMAT_ROWMAJOR:
            ret = "Row-major";
            break;
        case GHOST_DENSEMAT_COLMAJOR:
            ret = "Col-major";
            break;
        default:
            ret = "Invalid";
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return ret;
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

ghost_error ghost_densemat_info_string(char **str, ghost_densemat *densemat)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    int myrank;
    int mynoderank;
    ghost_mpi_comm nodecomm;
    GHOST_CALL_RETURN(ghost_nodecomm_get(&nodecomm));
    GHOST_CALL_RETURN(ghost_rank(&mynoderank, nodecomm));
    GHOST_CALL_RETURN(ghost_rank(&myrank, MPI_COMM_WORLD));

    GHOST_CALL_RETURN(ghost_malloc((void **)str,1));
    memset(*str,'\0',1);
    
    ghost_header_string(str,"Dense matrix @ local rank %d (glob %d)",mynoderank,myrank);
    ghost_line_string(str,"Dimension",NULL,"%"PRLIDX"x%"PRLIDX,densemat->traits.nrows,densemat->traits.ncols);
    ghost_line_string(str,"Dimension w/ halo",NULL,"%"PRLIDX"x%"PRLIDX,densemat->traits.nrowshalo,densemat->traits.ncols);
    ghost_line_string(str,"Padded dimension",NULL,"%"PRLIDX"x%"PRLIDX,densemat->traits.nrowspadded,densemat->traits.ncolspadded);
    ghost_line_string(str,"Number of blocks",NULL,"%"PRLIDX,densemat->nblock);
    ghost_line_string(str,"Stride between blocks",NULL,"%"PRLIDX,densemat->stride);
    ghost_line_string(str,"View",NULL,"%s",densemat->traits.flags&GHOST_DENSEMAT_VIEW?"Yes":"No");
    ghost_line_string(str,"Scattered",NULL,"%s",densemat->traits.flags&GHOST_DENSEMAT_SCATTERED?"Yes":"No");
    if (densemat->traits.flags&GHOST_DENSEMAT_VIEW) {
        ghost_line_string(str,"Dimension of viewed densemat",NULL,"%"PRLIDX"x%"PRLIDX,densemat->traits.nrowsorig,densemat->traits.ncolsorig);
        if (densemat->traits.flags & GHOST_DENSEMAT_SCATTERED) {
            char colmask[densemat->traits.ncolsorig];
            char colmaskstr[densemat->traits.ncolsorig+1];
            ghost_densemat_mask2charfield(densemat->colmask,densemat->traits.ncolsorig,colmask);
            charfield2string(colmaskstr,colmask,densemat->traits.ncolsorig);
            ghost_line_string(str,"Viewed columns",NULL,"%s",colmaskstr);
            char rowmask[densemat->traits.nrowsorig];
            char rowmaskstr[densemat->traits.nrowsorig+1];
            ghost_densemat_mask2charfield(densemat->rowmask,densemat->traits.nrowsorig,rowmask);
            charfield2string(rowmaskstr,rowmask,densemat->traits.nrowsorig);
            ghost_line_string(str,"Viewed rows",NULL,"%s",rowmaskstr);
        }

    }
   
    ghost_line_string(str,"Location",NULL,"%s",ghost_location_string(densemat->traits.location));
    ghost_line_string(str,"Storage order",NULL,"%s",ghost_densemat_storage_string(densemat->traits.storage));
    ghost_footer_string(str);
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;

}

ghost_error ghost_densemat_halocommInit_common(ghost_densemat *vec, ghost_context *ctx, ghost_densemat_halo_comm *comm) 
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
    int nprocs;
    int me; 
    int i;
    ghost_error ret = GHOST_SUCCESS;
    int rowsize = vec->traits.ncols*vec->elSize;

    if (vec->traits.flags & GHOST_DENSEMAT_NO_HALO) {
        ERROR_LOG("The densemat has no halo buffer!");
        return GHOST_ERR_INVALID_ARG;
    }
    if (vec->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        ERROR_LOG("Halo communication for scattered densemats not yet supported!");
        ret = GHOST_ERR_NOT_IMPLEMENTED;
        goto err;
    }


    GHOST_CALL_GOTO(ghost_rank(&me, ctx->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, ctx->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_rank(&me, ctx->mpicomm),err,ret);
    
    comm->msgcount = 0;
    GHOST_CALL_GOTO(ghost_malloc((void **)&comm->wishptr,(nprocs+1)*sizeof(ghost_lidx)),err,ret);

    int nMsgsOverall = 0;

    comm->wishptr[0] = 0;
    for (i=0;i<nprocs;i++) {
        comm->wishptr[i+1] = comm->wishptr[i]+ctx->wishes[i];
        if (ctx->wishes[i]) {
            nMsgsOverall += ((size_t)rowsize*ctx->wishes[i])/INT_MAX + 1;
        }
        if (ctx->dues[i]) {
            nMsgsOverall += ((size_t)rowsize*ctx->dues[i])/INT_MAX + 1;
        }
    }
    comm->acc_wishes = comm->wishptr[nprocs];

    GHOST_CALL_GOTO(ghost_malloc((void **)&comm->request,sizeof(MPI_Request)*nMsgsOverall),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&comm->status,sizeof(MPI_Status)*nMsgsOverall),err,ret);

    for (i=0;i<nMsgsOverall;i++) {
        comm->request[i] = MPI_REQUEST_NULL;
    }
    

    GHOST_CALL_RETURN(ghost_malloc((void **)&comm->dueptr,(nprocs+1)*sizeof(ghost_lidx)));

    comm->dueptr[0] = 0;
    for (i=0;i<nprocs;i++) {
        comm->dueptr[i+1] = comm->dueptr[i]+ctx->dues[i];
    }
    comm->acc_dues = comm->dueptr[nprocs];

    if (vec->traits.location & GHOST_LOCATION_DEVICE) {
#ifdef GHOST_HAVE_CUDA
        CUDA_CALL_RETURN(cudaHostAlloc((void **)&comm->work,(size_t)vec->traits.ncols*comm->acc_dues*vec->elSize,cudaHostAllocDefault));
        GHOST_CALL_GOTO(ghost_cu_malloc(&comm->cu_work,vec->traits.ncols*comm->acc_dues*vec->elSize),err,ret);
#endif
    } else {
        GHOST_CALL_RETURN(ghost_malloc((void **)&comm->work,(size_t)vec->traits.ncols*comm->acc_dues*vec->elSize));
    }

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

ghost_error ghost_densemat_halocommStart_common(ghost_densemat *vec, ghost_context *ctx, ghost_densemat_halo_comm *comm)
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION)
    ghost_error ret = GHOST_SUCCESS;
    char *recv;
    int from_PE, to_PE;
    int nprocs;
    int rowsize = vec->traits.ncols*vec->elSize;
    int me; 
    GHOST_CALL_GOTO(ghost_rank(&me, ctx->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, ctx->mpicomm),err,ret);

    for (from_PE=0; from_PE<nprocs; from_PE++){
        if (ctx->wishes[from_PE]>0) {
            recv = comm->tmprecv[from_PE];

#ifdef GHOST_TRACK_DATATRANSFERS
            ghost_datatransfer_register("spmv_halo",GHOST_DATATRANSFER_IN,from_PE,ctx->wishes[from_PE]*vec->elSize*vec->traits.ncols);
#endif
            int msg;
            int nmsgs = (size_t)rowsize*ctx->wishes[from_PE]/INT_MAX + 1;
            size_t msgSizeRows = ctx->wishes[from_PE]/nmsgs;
            size_t msgSizeEls = ctx->wishes[from_PE]/nmsgs*vec->traits.ncols;

            for (msg = 0; msg < nmsgs-1; msg++) {
                MPI_CALL_GOTO(MPI_Irecv(recv + msg*msgSizeRows*rowsize, msgSizeEls, vec->mpidt, from_PE, from_PE, ctx->mpicomm,&comm->request[comm->msgcount]),err,ret);
                comm->msgcount++;
            }

            // remainder
            MPI_CALL_GOTO(MPI_Irecv(recv + msg*msgSizeRows*rowsize, ctx->wishes[from_PE]*vec->traits.ncols - msg*msgSizeEls, vec->mpidt, from_PE, from_PE, ctx->mpicomm,&comm->request[comm->msgcount]),err,ret);
            comm->msgcount++;
        }
    }
    for (to_PE=0 ; to_PE<nprocs ; to_PE++){
        if (ctx->dues[to_PE]>0){
#ifdef GHOST_TRACK_DATATRANSFERS
            ghost_datatransfer_register("spmv_halo",GHOST_DATATRANSFER_OUT,to_PE,ctx->dues[to_PE]*vec->elSize*vec->traits.ncols);
#endif
            int msg;
            int nmsgs = (size_t)rowsize*ctx->dues[to_PE]/INT_MAX + 1;
            size_t msgSizeRows = ctx->dues[to_PE]/nmsgs;
            size_t msgSizeEls = ctx->dues[to_PE]/nmsgs*vec->traits.ncols;

            for (msg = 0; msg < nmsgs-1; msg++) {
                MPI_CALL_GOTO(MPI_Isend(comm->work + comm->dueptr[to_PE]*vec->elSize*vec->traits.ncols+msg*msgSizeRows*rowsize, msgSizeEls, vec->mpidt, to_PE, me, ctx->mpicomm, &comm->request[comm->msgcount]),err,ret);
                comm->msgcount++;
            }

            // remainder
            MPI_CALL_GOTO(MPI_Isend(comm->work + comm->dueptr[to_PE]*vec->elSize*vec->traits.ncols+msg*msgSizeRows*rowsize, ctx->dues[to_PE]*vec->traits.ncols - msg*msgSizeEls, vec->mpidt, to_PE, me, ctx->mpicomm, &comm->request[comm->msgcount]),err,ret);
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

ghost_error ghost_densemat_halocommFinalize_common(ghost_densemat_halo_comm *comm)
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
    ghost_error ret = GHOST_SUCCESS;

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

void ghost_densemat_destroy( ghost_densemat* vec ) 
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_TEARDOWN);
    if (vec) {
        if (!(vec->traits.flags & GHOST_DENSEMAT_VIEW)) {
            if (vec->traits.location & GHOST_LOCATION_DEVICE) {
                ghost_cu_free(vec->cu_val); vec->cu_val = NULL;
            } 
            if (vec->traits.location & GHOST_LOCATION_HOST) {
                ghost_type mytype;
                ghost_type_get(&mytype);
                if (mytype == GHOST_TYPE_CUDA) {
                    ghost_cu_free_host(vec->val); vec->val = NULL;
                } else {
                    free(vec->val); vec->val = NULL;
                }
            }
        }
        ghost_bitmap_free(vec->rowmask); vec->rowmask = NULL;
        ghost_bitmap_free(vec->colmask); vec->colmask = NULL;
#ifdef GHOST_HAVE_MPI
        MPI_Type_free(&(vec->fullmpidt));
#endif
        /* free permutation objects - but not the actual index arrays. They 
           belong to the context of the matrix which defines the permutation
           applied.
         */
        if (vec->perm_local) {free(vec->perm_local); vec->perm_local=NULL;}
        if (vec->perm_global){free(vec->perm_global); vec->perm_global=NULL;}
        free(vec);
    }
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_TEARDOWN);
}

ghost_lidx ghost_densemat_row_padding()
{
    // pad for SELL SpMV
    ghost_lidx padding = ghost_sell_max_cfg_chunkheight();  
    // pad for unrolled densemat kernels, assume worst case: SP data with 4 bytes
    padding = MAX(padding,ghost_machine_simd_width()/4 * GHOST_MAX_ROWS_UNROLL);
    return padding;
}

