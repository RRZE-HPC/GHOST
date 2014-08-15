#include "ghost/tsmm_inplace.h"
#include "ghost/math.h"
    
ghost_error_t ghost_tsmm_inplace(ghost_densemat_t *x, ghost_densemat_t *w, void *alpha)
{

    ghost_tsmm_inplace_parameters_t par = {.dt = x->traits.datatype, .blocksz = x->traits.ncols};
    ghost_tsmm_inplace_kernel_t kernel = ghost_tsmm_inplace_kernel(par,x,x,w,GHOST_GEMM_NO_REDUCE);

    if (!kernel) {
        INFO_LOG("Could not find in-place TSMM kernel with %d %d. Fallback to GEMM.",par.dt,par.blocksz);
        char zero[x->elSize];
        memset(zero,0,x->elSize);

        return ghost_gemm(x,x,"N",w,"N",alpha,zero,GHOST_GEMM_NO_REDUCE);
    }


    return kernel(x,w,alpha);

}
