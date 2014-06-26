#include "ghost/tsmttsm.h"
#include "ghost/math.h"
    
ghost_error_t ghost_tsmttsm(ghost_densemat_t *x, ghost_densemat_t *v, ghost_densemat_t *w, void *alpha, void *beta)
{
    if (x->traits.datatype != v->traits.datatype || x->traits.datatype != w->traits.datatype) {
        ERROR_LOG("Mixed data types not supported!");
        return GHOST_ERR_INVALID_ARG;
    }

    ghost_tsmttsm_parameters_t par = {.dt = x->traits.datatype, .blocksz = x->traits.ncols};
    tsmttsm_kernel kernel = ghost_tsmttsm_kernel(par);

    if (!kernel) {
        INFO_LOG("Could not find TSMTTSM kernel with %d %d. Fallback to GEMM",par.dt,par.blocksz);
        
        if (x->traits.datatype & GHOST_DT_REAL) {
            return ghost_gemm(x,v,"T",w,"N",alpha,beta,GHOST_GEMM_ALL_REDUCE);
        } else {
            return ghost_gemm(x,v,"C",w,"N",alpha,beta,GHOST_GEMM_ALL_REDUCE);
        }
    }


    return kernel(x,v,w,alpha,beta);

}
