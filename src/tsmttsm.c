#include "ghost/tsmttsm.h"
#include "ghost/math.h"
    
ghost_error_t ghost_tsmttsm(ghost_densemat_t *x, ghost_densemat_t *v, ghost_densemat_t *w, void *alpha, void *beta)
{

    ghost_tsmttsm_parameters_t par = {.dt = x->traits.datatype, .blocksz = x->traits.ncols};
    tsmttsm_kernel kernel = ghost_tsmttsm_kernel(par);

    if (!kernel) {
        INFO_LOG("Could not find TSMTTSM kernel with %d %d. Fallback to GEMM",par.dt,par.blocksz);
        return ghost_gemm(x,v,"T",w,"N",alpha,beta,GHOST_GEMM_ALL_REDUCE);
    }


    return kernel(x,v,w,alpha,beta);

}
