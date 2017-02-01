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
  
    ghost_kacz_opts opts = GHOST_KACZ_OPTS_INITIALIZER;
    opts.omega = omega;

    opts.direction = GHOST_KACZ_DIRECTION_FORWARD;


    GHOST_CALL_RETURN(ghost_densemat_halocomm_init(x,mat->context,&comm));
    GHOST_CALL_RETURN(ghost_densemat_halocomm_start(x,mat->context,&comm));
    GHOST_CALL_RETURN(ghost_densemat_halocomm_finalize(x,mat->context,&comm));
    
    if(flag_rb!= 0){
 	GHOST_CALL_RETURN(ghost_kacz_rb(x,mat,b,opts));
     } else {
       	GHOST_CALL_RETURN(ghost_kacz_mc(x,mat,b,opts)); 
     }
#ifdef GHOST_HAVE_MPI 
     MPI_CALL_RETURN(MPI_Barrier(mat->context->mpicomm));
#endif
    
     GHOST_CALL_RETURN(ghost_densemat_halo_avg(x,mat->context));

     opts.direction = GHOST_KACZ_DIRECTION_BACKWARD;

     GHOST_CALL_RETURN(ghost_densemat_halocomm_init(x,mat->context,&comm));
     GHOST_CALL_RETURN(ghost_densemat_halocomm_start(x,mat->context,&comm));
     GHOST_CALL_RETURN(ghost_densemat_halocomm_finalize(x,mat->context,&comm));

    
     if(flag_rb!=0){
       	GHOST_CALL_RETURN(ghost_kacz_rb(x,mat,b,opts));
     } else {
       	GHOST_CALL_RETURN(ghost_kacz_mc(x,mat,b,opts));   
     }

    GHOST_CALL_RETURN(ghost_densemat_halo_avg(x,mat->context));

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_SOLVER);
    return GHOST_SUCCESS;

}
