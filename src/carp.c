#include "ghost/carp.h"
#include "ghost/util.h"
#include "ghost/locality.h"

ghost_error ghost_carp(ghost_sparsemat *mat, ghost_densemat *x, ghost_densemat *b, void *omega)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_SOLVER);

    // 1. communicate local x entries to remote halo
    // 2. perform kacz on full matrix
    // 3. communicate remote halo x entries to local
    //    if column is present on more procs: average!

    ghost_densemat_halo_comm comm = GHOST_DENSEMAT_HALO_COMM_INITIALIZER;
    GHOST_CALL_RETURN(x->halocommInit(x,&comm));
    GHOST_CALL_RETURN(x->halocommStart(x,&comm));
    GHOST_CALL_RETURN(x->halocommFinalize(x,&comm));
    
    GHOST_CALL_RETURN(mat->kacz(mat,x,b,omega,1));
    GHOST_CALL_RETURN(x->averageHalo(x));
    GHOST_CALL_RETURN(mat->kacz(mat,x,b,omega,0));

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_SOLVER);
    return GHOST_SUCCESS;

}
