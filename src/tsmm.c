#include "ghost/tsmm.h"
#include "ghost/math.h"
    
ghost_error_t ghost_tsmm(ghost_densemat_t *x, ghost_densemat_t *v, ghost_densemat_t *w, void *alpha, void *beta)
{

    ghost_tsmm_parameters_t par = {.dt = x->traits.datatype, .blocksz1 = x->traits.ncols, .blocksz2 = v->traits.ncols};
    ghost_tsmm_kernel_t kernel = ghost_tsmm_kernel(par,x,v,w,GHOST_GEMM_NO_REDUCE);

    if (!kernel) {
        INFO_LOG("Could not find TSMM kernel with %d %d %d. Fallback to GEMM.",par.dt,par.blocksz1,par.blocksz2);
        return ghost_gemm(x,v,"N",w,"N",alpha,beta,GHOST_GEMM_NO_REDUCE);
    }


    return kernel(x,v,w,alpha,beta);

}
