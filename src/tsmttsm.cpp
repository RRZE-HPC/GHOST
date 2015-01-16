#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/densemat.h"
#include "ghost/util.h"
#include "ghost/math.h"
#include "ghost/tsmttsm.h"
#include "ghost/tsmttsm_gen.h"
#include "ghost/tsmttsm_avx_gen.h"

#include <map>

using namespace std;

bool operator<(const ghost_tsmttsm_parameters_t &a, const ghost_tsmttsm_parameters_t &b) 
{ 
    return ghost_hash(a.dt,a.blocksz1,ghost_hash(a.blocksz2,a.impl,0)) < ghost_hash(b.dt,b.blocksz1,ghost_hash(b.blocksz2,b.impl,0)); 
}

static map<ghost_tsmttsm_parameters_t, ghost_tsmttsm_kernel_t> ghost_tsmttsm_kernels;

void ghost_tsmttsm_kernelmap_generate() {
#include "tsmttsm.def"
#include "tsmttsm_avx.def"
}
ghost_error_t ghost_tsmttsm(ghost_densemat_t *x, ghost_densemat_t *v, ghost_densemat_t *w, void *alpha, void *beta)
{
    if (w->traits.storage != GHOST_DENSEMAT_ROWMAJOR) {
        ERROR_LOG("w must be stored row-major!");
        return GHOST_ERR_INVALID_ARG;
    }
    if (v->traits.storage != GHOST_DENSEMAT_ROWMAJOR) {
        ERROR_LOG("v must be stored row-major!");
        return GHOST_ERR_INVALID_ARG;
    }
    if (v->traits.datatype != w->traits.datatype || v->traits.datatype != x->traits.datatype) {
        ERROR_LOG("Different data types!");
        return GHOST_ERR_INVALID_ARG;
    }
    if (v->traits.flags & GHOST_DENSEMAT_SCATTERED || w->traits.flags & GHOST_DENSEMAT_SCATTERED || x->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        PERFWARNING_LOG("Scattered densemats not supported!");
        return GHOST_ERR_INVALID_ARG;
    }
    
    if (ghost_tsmttsm_kernels.empty()) {
#include "tsmttsm.def"
#include "tsmttsm_avx.def"
    }
    
    ghost_tsmttsm_parameters_t p;
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
    
    ghost_tsmttsm_kernel_t kernel = ghost_tsmttsm_kernels[p];
    if (!kernel) {
        PERFWARNING_LOG("Try kernel with arbitrary block sizes");
        p.blocksz1 = -1;
        p.blocksz2 = -1;
    }
    kernel = ghost_tsmttsm_kernels[p];
    
    if (!kernel) {
        PERFWARNING_LOG("Try plain implementation");
        p.impl = GHOST_IMPLEMENTATION_PLAIN;
    }
    kernel = ghost_tsmttsm_kernels[p];
    
    if (!kernel) {
        INFO_LOG("Could not find TSMTTSM kernel with %d %d %d. Fallback to GEMM",p.dt,p.blocksz1,p.blocksz2);
        
            return ghost_gemm(x,v,"T",w,"N",alpha,beta,GHOST_GEMM_ALL_REDUCE);
    }


    return kernel(x,v,w,alpha,beta);
}

