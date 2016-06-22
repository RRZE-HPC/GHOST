#include "ghost/sparsemat.h"
#include "ghost/densemat.h"
#include "ghost/sell_kacz_rb.h"
#include "ghost/sell_kacz_bmc.h"
#include "ghost/sell_kacz_mc.h"

ghost_error ghost_sell_kacz_selector(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts)
{
    ghost_error ret = GHOST_SUCCESS;
    if(!(mat->traits.flags & GHOST_SPARSEMAT_COLOR)) {
    	if(!(mat->traits.flags & GHOST_SPARSEMAT_BLOCKCOLOR) && (mat->kaczRatio >= 2*mat->kacz_setting.active_threads)) {
		INFO_LOG("BMC KACZ without transition called");
		ret = ghost_kacz_rb(x,mat,b,opts);
    	}
    	else {
		INFO_LOG("BMC KACZ with transition called");
		ret = ghost_kacz_bmc(x,mat,b,opts);  
    	}
    } else {
	INFO_LOG("Using unoptimal kernel KAVZ with MC");
	ret = ghost_kacz_mc(x,mat,b,opts);
    }
return ret;
}
		
ghost_error ghost_sell_kacz_shift_selector(ghost_densemat *x_real, ghost_densemat *x_imag, ghost_sparsemat *mat, ghost_densemat *b, double sigma_r, double sigma_i, ghost_kacz_opts opts)
{
    ghost_error ret = GHOST_SUCCESS;
/*    if(!(mat->traits.flags & GHOST_SPARSEMAT_COLOR)) {
    	if(!(mat->traits.flags & GHOST_SPARSEMAT_BLOCKCOLOR) && (mat->kaczRatio >= 2*mat->kacz_setting.active_threads)) {
		INFO_LOG("BMC KACZ without transition called");
		ret = ghost_kacz_rb(x,mat,b,opts);
    	}
    	else {*/
		INFO_LOG("BMC KACZ with transition called");
		ret = ghost_kacz_shift_bmc(x_real,x_imag,mat,b,sigma_r,sigma_i,opts);  
/*    	}
    } else {
	INFO_LOG("Using unoptimal kernel KAVZ with MC");
	ret = ghost_kacz_mc(x,mat,b,opts);
    }*/
return ret;
}  
