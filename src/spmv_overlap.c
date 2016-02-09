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

ghost_error ghost_spmv_goodfaith(ghost_densemat* res, ghost_sparsemat* mat, ghost_densemat* invec, ghost_spmv_opts traits)
{
#ifndef GHOST_HAVE_MPI
    UNUSED(res);
    UNUSED(mat);
    UNUSED(invec);
    UNUSED(traits);
    ERROR_LOG("Cannot execute this spMV solver without MPI");
    return GHOST_ERR_UNKNOWN;
#else
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error ret = GHOST_SUCCESS;

    ghost_spmv_opts localtraits = traits, remotetraits = traits;
    localtraits.flags = (ghost_spmv_flags)(localtraits.flags|(ghost_spmv_flags)GHOST_SPMV_LOCAL);
    remotetraits.flags = (ghost_spmv_flags)(remotetraits.flags|(ghost_spmv_flags)GHOST_SPMV_REMOTE);
    
    ghost_densemat_halo_comm comm = GHOST_DENSEMAT_HALO_COMM_INITIALIZER;

    GHOST_CALL_RETURN(invec->halocommInit(invec,&comm));

    GHOST_INSTR_START("comm+localcomp");
    GHOST_CALL_RETURN(invec->halocommStart(invec,&comm));
    
    GHOST_INSTR_START("local");
    GHOST_CALL_GOTO(mat->localPart->spmv(res,mat->localPart,invec,localtraits),err,ret);
    GHOST_INSTR_STOP("local");

    GHOST_CALL_RETURN(invec->halocommFinalize(invec,&comm));
    GHOST_INSTR_STOP("comm+localcomp");
    
    GHOST_INSTR_START("remote");
    GHOST_CALL_GOTO(mat->remotePart->spmv(res,mat->remotePart,invec,remotetraits),err,ret);
    GHOST_INSTR_STOP("remote");

    goto out;
err:

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
#endif
}
