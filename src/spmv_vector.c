#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/locality.h"
#include "ghost/densemat.h"
#include "ghost/util.h"
#include "ghost/instr.h"
#include "ghost/sparsemat.h"
#include "ghost/spmv_solvers.h"
#include "ghost/math.h"

#ifdef GHOST_HAVE_MPI
#include <mpi.h>
#endif
#include <stdio.h>
#include <string.h>

#ifdef GHOST_HAVE_OPENMP
#include <omp.h>
#endif

ghost_error ghost_spmv_vectormode(ghost_densemat* res, ghost_sparsemat* mat, ghost_densemat* invec, ghost_spmv_opts traits)
{
#ifndef GHOST_HAVE_MPI
    UNUSED(mat->context);
    UNUSED(res);
    UNUSED(mat);
    UNUSED(invec);
    UNUSED(traits);
    GHOST_ERROR_LOG("Cannot execute this spMV solver without MPI");
    return GHOST_ERR_UNKNOWN;
#else
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error ret = GHOST_SUCCESS;
    if (mat->context == NULL) {
        GHOST_ERROR_LOG("The mat->context is NULL");
        ret = GHOST_ERR_INVALID_ARG;
        goto err;
    }

    
    GHOST_INSTR_START("comm");
    ghost_densemat_halo_comm comm = GHOST_DENSEMAT_HALO_COMM_INITIALIZER;
    GHOST_CALL_GOTO(ghost_densemat_halocomm_init(invec,mat->context,&comm),err,ret);
    GHOST_CALL_GOTO(ghost_densemat_halocomm_start(invec,mat->context,&comm),err,ret);
    GHOST_CALL_GOTO(ghost_densemat_halocomm_finalize(invec,mat->context,&comm),err,ret);
    GHOST_INSTR_STOP("comm");

    GHOST_INSTR_START("comp");
    GHOST_CALL_GOTO(ghost_spmv_nocomm(res,mat,invec,traits),err,ret);    
    GHOST_INSTR_STOP("comp");

    goto out;
err:

out:

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);
    return ret;
#endif
}

