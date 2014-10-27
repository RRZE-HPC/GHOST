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

#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

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
        (*vec)->traits.flags |= GHOST_DENSEMAT_HOST;
        ghost_type_t ghost_type;
        GHOST_CALL_RETURN(ghost_type_get(&ghost_type));
        if (ghost_type == GHOST_TYPE_CUDA) {
            (*vec)->traits.flags |= GHOST_DENSEMAT_DEVICE;
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
                    vec->traits.nrowshalo = vec->traits.nrows+vec->context->halo_elements;
                } else {
                    vec->traits.nrowshalo = vec->traits.nrows;
                }
            }    
        }
    } else {
        // the case context==NULL is allowed - the vector is local.
        DEBUG_LOG(1,"The vector's context is NULL.");
    }


    if (vec->traits.nrowspadded == 0) {
        DEBUG_LOG(2,"nrowspadded for vector not given. determining it from the context");
        vec->traits.nrowspadded = PAD(MAX(vec->traits.nrowshalo,vec->traits.nrows),GHOST_PAD_MAX); // TODO needed?
    }
    if (vec->traits.ncolspadded == 0) {
        DEBUG_LOG(2,"ncolspadded for vector not given. determining it from the context");
        ghost_lidx_t padding = vec->elSize;
        if (vec->traits.ncols > 1) {
#ifdef GHOST_HAVE_MIC
            padding = 64; // 64 byte padding
#elif defined(GHOST_HAVE_AVX)
            padding = 32; // 32 byte padding
            if (vec->traits.ncols <= 2) {
                padding = 16; // SSE in this case: only 16 byte alignment required
            }
#elif defined (GHOST_HAVE_SSE)
            padding = 16; // 16 byte padding
#endif
        }
        padding /= vec->elSize;
        if (vec->traits.ncols % padding) {
            INFO_LOG("Cols will be padded to a multiple of %"PRLIDX,padding);
        }
        vec->traits.ncolspadded = PAD(vec->traits.ncols,padding);
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
    ghost_line_string(str,"View",NULL,"%s",densemat->traits.flags&GHOST_DENSEMAT_VIEW?"Yes":"No");
    if (densemat->traits.flags&GHOST_DENSEMAT_VIEW) {
        ghost_line_string(str,"Dimension of viewed densemat",NULL,"%"PRLIDX"x%"PRLIDX,densemat->traits.nrowsorig,densemat->traits.ncolsorig);
        char colmask[densemat->traits.ncolsorig];
        char colmaskstr[densemat->traits.ncolsorig+1];
        ghost_densemat_mask2charfield((densemat->traits.flags&GHOST_DENSEMAT_ROWMAJOR)?densemat->trmask:densemat->ldmask,densemat->traits.ncolsorig,colmask);
        charfield2string(colmaskstr,colmask,densemat->traits.ncolsorig);
        ghost_line_string(str,"Viewed columns",NULL,"%s",colmaskstr);
        char rowmask[densemat->traits.nrowsorig];
        char rowmaskstr[densemat->traits.nrowsorig+1];
        ghost_densemat_mask2charfield(densemat->traits.flags&GHOST_DENSEMAT_ROWMAJOR?densemat->ldmask:densemat->trmask,densemat->traits.nrowsorig,rowmask);
        charfield2string(rowmaskstr,rowmask,densemat->traits.nrowsorig);
        ghost_line_string(str,"Viewed rows",NULL,"%s",rowmaskstr);

    }
   
     ghost_line_string(str,"Location",NULL,"%s",densemat->traits.flags&GHOST_DENSEMAT_DEVICE?densemat->traits.flags&GHOST_DENSEMAT_HOST?"Device+Host":"Device":"Host");
    ghost_line_string(str,"Storage order",NULL,"%s",ghost_densemat_storage_string(densemat));
    ghost_footer_string(str);
    
    return GHOST_SUCCESS;

}
