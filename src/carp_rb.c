#include "ghost/carp.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include "ghost/sell_kacz_rb.h"

ghost_error ghost_carp_rb(ghost_sparsemat *mat, ghost_densemat *x, ghost_densemat *b, void *omega, int flag_rb)
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
   
    ghost_kacz_opts opts = GHOST_KACZ_OPTS_INITIALIZER;
    opts.omega = omega;
    opts.direction = GHOST_KACZ_DIRECTION_FORWARD;

    if(flag_rb!=0){
   	 GHOST_CALL_RETURN(ghost_kacz_rb(x,mat,b,opts));
   	 GHOST_CALL_RETURN(x->averageHalo(x));
   	 opts.direction = GHOST_KACZ_DIRECTION_BACKWARD;
   	 GHOST_CALL_RETURN(ghost_kacz_rb(x,mat,b,opts));
    } else {
         GHOST_CALL_RETURN(ghost_kacz(x,mat,b,opts));
         GHOST_CALL_RETURN(x->averageHalo(x));
         opts.direction = GHOST_KACZ_DIRECTION_BACKWARD;
         GHOST_CALL_RETURN(ghost_kacz(x,mat,b,opts));
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_SOLVER);
    return GHOST_SUCCESS;

}