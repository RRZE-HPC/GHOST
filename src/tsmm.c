#include "ghost/tsmm.h"
#include "ghost/math.h"
    
ghost_error_t ghost_tsmm(ghost_densemat_t *x, ghost_densemat_t *v, ghost_densemat_t *w, void *alpha, void *beta)
{
    if (x->traits.datatype != v->traits.datatype || x->traits.datatype != w->traits.datatype) {
        ERROR_LOG("Mixed data types not supported!");
        return GHOST_ERR_INVALID_ARG;
    }

    ghost_tsmm_parameters_t par = {.dt = x->traits.datatype, .blocksz1 = x->traits.ncols, .blocksz2 = v->traits.ncols};
    tsmm_kernel kernel = ghost_tsmm_kernel(par);

    if (!kernel || (x->traits.flags & GHOST_DENSEMAT_SCATTERED)|| (v->traits.flags & GHOST_DENSEMAT_SCATTERED) || (w->traits.flags & GHOST_DENSEMAT_SCATTERED) || !ghost_bitmap_iscompact(x->ldmask) || !ghost_bitmap_iscompact(v->ldmask) || !ghost_bitmap_iscompact(w->ldmask)) {
        INFO_LOG("Could not find TSMM kernel with %d %d %d. Fallback to GEMM.",par.dt,par.blocksz1,par.blocksz2);
        return ghost_gemm(x,v,"N",w,"N",alpha,beta,GHOST_GEMM_NO_REDUCE);
    }


    return kernel(x,v,w,alpha,beta);

}
