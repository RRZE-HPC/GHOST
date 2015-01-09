#include "ghost/carp.h"
#include "ghost/util.h"
#include "ghost/locality.h"

ghost_error_t ghost_carp(ghost_sparsemat_t *mat, ghost_densemat_t *x, ghost_densemat_t *b, void *omega)
{

    // 1. communicate local x entries to remote halo
    // 2. perform kacz on full matrix
    // 3. communicate remote halo x entries to local
    //    if column is present on more procs: average!

    ghost_densemat_halo_comm_t comm;
    GHOST_CALL_RETURN(x->halocommInit(x,&comm));
    GHOST_CALL_RETURN(x->halocommStart(x,&comm));
    GHOST_CALL_RETURN(x->halocommFinalize(x,&comm));
    
    GHOST_CALL_RETURN(mat->kacz(mat,x,b,omega,1));
    GHOST_CALL_RETURN(x->averageHalo(x));
    GHOST_CALL_RETURN(mat->kacz(mat,x,b,omega,0));

    return GHOST_SUCCESS;

}
