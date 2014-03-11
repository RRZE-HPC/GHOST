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

static ghost_error_t getNrowsFromContext(ghost_densemat_t *vec);

ghost_error_t ghost_densemat_create(ghost_densemat_t **vec, ghost_context_t *ctx, ghost_densemat_traits_t traits)
{
    ghost_error_t ret = GHOST_SUCCESS;
    GHOST_CALL_GOTO(ghost_malloc((void **)vec,sizeof(ghost_densemat_t)),err,ret);
    (*vec)->context = ctx;
    (*vec)->traits = traits;
    (*vec)->mask = hwloc_bitmap_alloc();
    if (!(*vec)->mask) {
        ERROR_LOG("Could not create dense matrix mask!");
        goto err;
    }
    hwloc_bitmap_fill((*vec)->mask);

    getNrowsFromContext((*vec));
    GHOST_CALL_GOTO(ghost_datatype_size(&(*vec)->elSize,(*vec)->traits.datatype),err,ret);

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
        ghost_densemat_rm_create(*vec);
    } else {
        ghost_densemat_cm_create(*vec);
    }

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
            if (vec->traits.flags & GHOST_DENSEMAT_DUMMY) {
                vec->traits.nrows = 0;
            } else if ((vec->context->flags & GHOST_CONTEXT_REDUNDANT) || (vec->traits.flags & GHOST_DENSEMAT_GLOBAL))
            {
                if (vec->traits.flags & GHOST_DENSEMAT_LHS) {
                    vec->traits.nrows = vec->context->gnrows;
                } else if (vec->traits.flags & GHOST_DENSEMAT_RHS) {
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
            if (vec->traits.flags & GHOST_DENSEMAT_DUMMY) {
                vec->traits.nrowshalo = 0;
            } else if ((vec->context->flags & GHOST_CONTEXT_REDUNDANT) || (vec->traits.flags & GHOST_DENSEMAT_GLOBAL))
            {
                vec->traits.nrowshalo = vec->traits.nrows;
            } 
            else 
            {
                if (!(vec->traits.flags & GHOST_DENSEMAT_GLOBAL) && vec->traits.flags & GHOST_DENSEMAT_RHS) {
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
        vec->traits.ncolspadded = PAD(vec->traits.ncols,GHOST_PAD_MAX); // TODO needed?
    }
    if (vec->traits.ncolsorig == 0) {
        vec->traits.ncolsorig = vec->traits.ncols;
    }

    DEBUG_LOG(1,"The vector has %"PRIDX" w/ %"PRIDX" halo elements (padded: %"PRIDX") rows",
            vec->traits.nrows,vec->traits.nrowshalo-vec->traits.nrows,vec->traits.nrowspadded);
    return GHOST_SUCCESS; 
}

