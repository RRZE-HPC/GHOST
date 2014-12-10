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
#include <stdio.h>
#include <string.h>

#ifdef GHOST_HAVE_OPENMP
#include <omp.h>
#endif

ghost_error_t ghost_spmv_vectormode(ghost_densemat_t* res, ghost_sparsemat_t* mat, ghost_densemat_t* invec, ghost_spmv_flags_t flags,va_list argp)
{
#ifndef GHOST_HAVE_MPI
    UNUSED(mat->context);
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
    if (mat->context == NULL) {
        ERROR_LOG("The mat->context is NULL");
        ret = GHOST_ERR_INVALID_ARG;
        goto err;
    }

    
    GHOST_INSTR_START("comm");
    GHOST_CALL_GOTO(ghost_spmv_haloexchange_initiate(invec,false),err,ret);
    GHOST_CALL_GOTO(ghost_spmv_haloexchange_finalize(invec),err,ret);
    GHOST_INSTR_STOP("comm");

    GHOST_INSTR_START("comp");
    GHOST_CALL_GOTO(mat->spmv(mat,res,invec,flags,argp),err,ret);    
    GHOST_INSTR_STOP("comp");

    goto out;
err:

out:

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);
    return ret;
#endif
}

