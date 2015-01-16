#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/densemat.h"
#include "ghost/util.h"
#include "ghost/tsmm_inplace.h"
#include "ghost/tsmm_inplace_gen.h"
#include "ghost/tsmm_inplace.h"
#include "ghost/math.h"
#include <map>

using namespace std;

bool operator<(const ghost_tsmm_inplace_parameters_t &a, const ghost_tsmm_inplace_parameters_t &b) 
{ 
    return ghost_hash(a.dt,a.blocksz,0) < ghost_hash(b.dt,b.blocksz,0); 
}


static map<ghost_tsmm_inplace_parameters_t, ghost_tsmm_inplace_kernel_t> ghost_tsmm_inplace_kernels;

ghost_error_t ghost_tsmm_inplace(ghost_densemat_t *x, ghost_densemat_t *w, void *alpha)
{
    if (x->traits.datatype != w->traits.datatype) {
        ERROR_LOG("Different data types!");
        return GHOST_ERR_INVALID_ARG;
    }
    if (w->traits.storage != GHOST_DENSEMAT_COLMAJOR) {
        ERROR_LOG("w must be stored col-major!");
        return GHOST_ERR_INVALID_ARG;
    }
    if (x->traits.storage != GHOST_DENSEMAT_ROWMAJOR) {
        ERROR_LOG("x must be stored row-major!");
        return GHOST_ERR_INVALID_ARG;
    }
    if ((x->traits.flags & GHOST_DENSEMAT_SCATTERED) || (w->traits.flags & GHOST_DENSEMAT_SCATTERED)) {
        ERROR_LOG("Scattered views not supported!");
        return GHOST_ERR_INVALID_ARG;
    }
    
    if (ghost_tsmm_inplace_kernels.empty()) {
#include "tsmm_inplace.def"
    }
    
    ghost_tsmm_inplace_parameters_t p;

#ifdef GHOST_HAVE_MIC
    p.impl = GHOST_IMPLEMENTATION_MIC;
#elif defined(GHOST_HAVE_AVX)
    p.impl = GHOST_IMPLEMENTATION_AVX;
#elif defined(GHOST_HAVE_SSE)
    p.impl = GHOST_IMPLEMENTATION_SSE;
#endif

    p.dt = x->traits.datatype;
    p.blocksz = x->traits.ncolspadded;
    

    ghost_tsmm_inplace_kernel_t kernel = ghost_tsmm_inplace_kernels[p];
    if (!kernel) {
        PERFWARNING_LOG("Try kernel with arbitrary block size");
        p.blocksz = -1;
    }
    kernel = ghost_tsmm_inplace_kernels[p];

    if (!kernel) {
        INFO_LOG("Could not find in-place TSMM kernel with %d %d. Fallback to GEMM.",p.dt,p.blocksz);
        char zero[x->elSize];
        memset(zero,0,x->elSize);

        return ghost_gemm(x,x,"N",w,"N",alpha,zero,GHOST_GEMM_NO_REDUCE);
    }


    return kernel(x,w,alpha);

}
