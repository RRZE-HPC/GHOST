#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/locality.h"
#include "ghost/densemat.h"
#include "ghost/util.h"
#include "ghost/instr.h"
#include "ghost/sparsemat.h"
#include "ghost/spmv_solvers.h"

#ifdef GHOST_HAVE_MPI
#include <mpi.h>
#endif
#include <sys/types.h>
#include <string.h>

#ifdef GHOST_HAVE_OPENMP
#include <omp.h>
#endif

ghost_error_t ghost_spmv_goodfaith(ghost_densemat_t* res, ghost_sparsemat_t* mat, ghost_densemat_t* invec, ghost_spmv_flags_t flags, va_list argp)
{
#ifndef GHOST_HAVE_MPI
    UNUSED(res);
    UNUSED(mat);
    UNUSED(invec);
    UNUSED(flags);
    UNUSED(argp);
    ERROR_LOG("Cannot execute this spMV solver without MPI");
    return GHOST_ERR_UNKNOWN;
#else
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret = GHOST_SUCCESS;

    ghost_spmv_flags_t localopts = (ghost_spmv_flags_t)(flags|(ghost_spmv_flags_t)GHOST_SPMV_LOCAL);
    ghost_spmv_flags_t remoteopts = (ghost_spmv_flags_t)(flags|(ghost_spmv_flags_t)GHOST_SPMV_REMOTE);
    ghost_densemat_halo_comm_t comm;

    va_list remote_argp;
    va_copy(remote_argp,argp);

    GHOST_CALL_RETURN(invec->halocommInit(invec,&comm));
    GHOST_CALL_RETURN(invec->halocommStart(invec,&comm));
    
    GHOST_INSTR_START("local");
    GHOST_CALL_GOTO(mat->localPart->spmv(mat->localPart,res,invec,localopts,argp),err,ret);
    GHOST_INSTR_STOP("local");

    GHOST_CALL_RETURN(invec->halocommFinalize(invec,&comm));
    
    GHOST_INSTR_START("remote");
    GHOST_CALL_GOTO(mat->remotePart->spmv(mat->remotePart,res,invec,remoteopts,remote_argp),err,ret);
    GHOST_INSTR_STOP("remote");

    goto out;
err:

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
#endif
}
