#include "ghost/tsmttsm.h"
#include "ghost/math.h"
    
ghost_error_t ghost_tsmttsm(ghost_densemat_t *x, ghost_densemat_t *v, ghost_densemat_t *w, void *alpha, void *beta)
{
    ghost_tsmttsm_parameters_t par = {.dt = x->traits.datatype, .blocksz1 = x->traits.ncols, .blocksz2 = x->traits.nrows};
    ghost_tsmttsm_kernel_t kernel = ghost_tsmttsm_kernel(par,x,v,w,GHOST_GEMM_ALL_REDUCE);

    if (!kernel) {
        INFO_LOG("Could not find TSMTTSM kernel with %d %d %d. Fallback to GEMM",par.dt,par.blocksz1,par.blocksz2);
        
        if (x->traits.datatype & GHOST_DT_REAL) {
            return ghost_gemm(x,v,"T",w,"N",alpha,beta,GHOST_GEMM_ALL_REDUCE);
        } else {
            return ghost_gemm(x,v,"C",w,"N",alpha,beta,GHOST_GEMM_ALL_REDUCE);
        }
    }


    return kernel(x,v,w,alpha,beta);
}
