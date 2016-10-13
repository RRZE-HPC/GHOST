#include "ghost/carp.h"
#include "ghost/util.h"
#include "ghost/locality.h"

ghost_error ghost_carp(ghost_sparsemat *mat, ghost_densemat *x, ghost_densemat *b, ghost_carp_opts carp_opts)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_SOLVER);

    // 1. communicate local x entries to remote halo
    // 2. perform kacz on full matrix
    // 3. communicate remote halo x entries to local
    //    if column is present on more procs: average!

    ghost_densemat_halo_comm comm = GHOST_DENSEMAT_HALO_COMM_INITIALIZER;
    ghost_kacz_opts opts = GHOST_KACZ_OPTS_INITIALIZER;
    opts.omega = carp_opts.omega;
    opts.shift = carp_opts.shift;
    opts.num_shifts = carp_opts.num_shifts;
    opts.best_block_size = carp_opts.best_block_size;
    opts.normalize = carp_opts.normalize;
    opts.scale = carp_opts.scale;
    opts.initialized = carp_opts.initialized;

    opts.direction = GHOST_KACZ_DIRECTION_FORWARD;
    GHOST_CALL_RETURN(ghost_densemat_halocomm_init(x,mat->context,&comm));
    GHOST_CALL_RETURN(ghost_densemat_halocomm_start(x,mat->context,&comm));
    GHOST_CALL_RETURN(ghost_densemat_halocomm_finalize(x,mat->context,&comm));
    ghost_kacz(x,mat,b,opts);    
    GHOST_CALL_RETURN(ghost_densemat_halo_avg(x,mat->context));

    opts.direction = GHOST_KACZ_DIRECTION_BACKWARD;
    GHOST_CALL_RETURN(ghost_densemat_halocomm_init(x,mat->context,&comm));
    GHOST_CALL_RETURN(ghost_densemat_halocomm_start(x,mat->context,&comm));
    GHOST_CALL_RETURN(ghost_densemat_halocomm_finalize(x,mat->context,&comm));
    ghost_kacz(x,mat,b,opts);
    GHOST_CALL_RETURN(ghost_densemat_halo_avg(x,mat->context));

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_SOLVER);

    return GHOST_SUCCESS;
}

