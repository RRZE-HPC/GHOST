#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/densemat.h"
#include "ghost/util.h"
#include "ghost/math.h"
#include "ghost/tsmm.h"
#include "ghost/tsmm_gen.h"
#include "ghost/tsmm_avx_gen.h"

#include <map>

using namespace std;

    
bool operator<(const ghost_tsmm_parameters_t &a, const ghost_tsmm_parameters_t &b) 
{ 
    return ghost_hash(a.dt,a.blocksz1,ghost_hash(a.blocksz2,a.impl,0)) < ghost_hash(b.dt,b.blocksz1,ghost_hash(b.blocksz2,b.impl,0)); 
}


static map<ghost_tsmm_parameters_t, ghost_tsmm_kernel_t> ghost_tsmm_kernels;

ghost_error_t ghost_tsmm(ghost_densemat_t *x, ghost_densemat_t *v, ghost_densemat_t *w, void *alpha, void *beta)
{

    if (x->traits.datatype != v->traits.datatype || x->traits.datatype != w->traits.datatype) {
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
    if (v->traits.storage != GHOST_DENSEMAT_ROWMAJOR) {
        ERROR_LOG("v must be stored row-major!");
        return GHOST_ERR_INVALID_ARG;
    }
    if ((x->traits.flags & GHOST_DENSEMAT_SCATTERED) || (v->traits.flags & GHOST_DENSEMAT_SCATTERED) || (w->traits.flags & GHOST_DENSEMAT_SCATTERED)) {
        ERROR_LOG("Scattered views not supported!");
        return GHOST_ERR_INVALID_ARG;
    }
    
    if (ghost_tsmm_kernels.empty()) {
#include "tsmm.def"
#include "tsmm_avx.def"
    }

    ghost_tsmm_parameters_t p;
#ifdef GHOST_HAVE_MIC
    p.impl = GHOST_IMPLEMENTATION_MIC;
#elif defined(GHOST_HAVE_AVX)
    p.impl = GHOST_IMPLEMENTATION_AVX;
#elif defined(GHOST_HAVE_SSE)
    p.impl = GHOST_IMPLEMENTATION_SSE;
#endif

    p.dt = x->traits.datatype;
    p.blocksz1 = x->traits.ncolspadded;
    p.blocksz2 = v->traits.ncolspadded;

    if (w->traits.ncolspadded < 4 && v->traits.ncolspadded < 4) {
        PERFWARNING_LOG("Try SSE for small densemats");
        p.impl = GHOST_IMPLEMENTATION_SSE;
    }

    ghost_tsmm_kernel_t kernel = ghost_tsmm_kernels[p];
    if (!kernel) {
        PERFWARNING_LOG("Try kernel with arbitrary block sizes");
        p.blocksz1 = -1;
        p.blocksz2 = -1;
    }
    kernel = ghost_tsmm_kernels[p];

    if (!kernel) {
        PERFWARNING_LOG("Try plain implementation");
        p.impl = GHOST_IMPLEMENTATION_PLAIN;
    }
    kernel = ghost_tsmm_kernels[p];

    if (!kernel) {
        INFO_LOG("Could not find TSMM kernel with %d %d %d %d. Fallback to GEMM.",p.impl,p.dt,p.blocksz1,p.blocksz2);
        return ghost_gemm(x,v,"N",w,"N",alpha,beta,GHOST_GEMM_NO_REDUCE);
    }


    return kernel(x,v,w,alpha,beta);
}
