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
    return ghost_hash(a.dt,a.xcols,a.impl) < ghost_hash(b.dt,b.xcols,b.impl); 
}


static map<ghost_tsmm_inplace_parameters_t, ghost_tsmm_inplace_kernel_t> ghost_tsmm_inplace_kernels;

ghost_error_t ghost_tsmm_inplace_valid(ghost_densemat_t *x, ghost_densemat_t *v,  char * transv, 
ghost_densemat_t *w, char *transw, void *alpha, void *beta, int reduce, int printerror)
{
    if (x->traits.datatype != w->traits.datatype) {
        if (printerror) {
            ERROR_LOG("Different data types!");
        }
        return GHOST_ERR_INVALID_ARG;
    }
    if (w->traits.storage != GHOST_DENSEMAT_COLMAJOR) {
        if (printerror) {
            ERROR_LOG("w must be stored col-major!");
        }
        return GHOST_ERR_INVALID_ARG;
    }
    if (x->traits.storage != GHOST_DENSEMAT_ROWMAJOR) {
        if (printerror) {
            ERROR_LOG("x must be stored row-major!");
        }
        return GHOST_ERR_INVALID_ARG;
    }
    if ((x->traits.flags & GHOST_DENSEMAT_SCATTERED) || (w->traits.flags & GHOST_DENSEMAT_SCATTERED)) {
        if (printerror) {
            ERROR_LOG("Scattered views not supported!");
        }
        return GHOST_ERR_INVALID_ARG;
    }
    if (x != v) {
        if (printerror) {
            ERROR_LOG("Densemats must be equal!");
        }
        return GHOST_ERR_INVALID_ARG;
    }
    if (strncasecmp(transv,"N",1)) {
        if (printerror) {
            ERROR_LOG("v must not be transposed!");
        }
        return GHOST_ERR_INVALID_ARG;
    }
    if (strncasecmp(transw,"N",1)) {
        if (printerror) {
            ERROR_LOG("w must not be transposed!");
        }
        return GHOST_ERR_INVALID_ARG;
    }

    return GHOST_SUCCESS;

}

ghost_error_t ghost_tsmm_inplace(ghost_densemat_t *x, ghost_densemat_t *w, void *alpha)
{
    ghost_error_t ret;

    if ((ret = ghost_tsmm_inplace_valid(x,x,"N",w,"N",alpha,NULL,GHOST_GEMM_NO_REDUCE,1)) != GHOST_SUCCESS) {
        return ret;
    }
    
    if (ghost_tsmm_inplace_kernels.empty()) {
#include "tsmm_inplace.def"
    }
    
    ghost_tsmm_inplace_parameters_t p;
    ghost_tsmm_inplace_kernel_t kernel = NULL;

#ifdef GHOST_HAVE_MIC
    p.impl = GHOST_IMPLEMENTATION_MIC;
#elif defined(GHOST_HAVE_AVX)
    p.impl = GHOST_IMPLEMENTATION_AVX;
#elif defined(GHOST_HAVE_SSE)
    p.impl = GHOST_IMPLEMENTATION_SSE;
#else
    p.impl = GHOST_IMPLEMENTATION_PLAIN;
#endif

    p.dt = x->traits.datatype;

    if (!(x->traits.flags & GHOST_DENSEMAT_VIEW)) {
        p.xcols = x->traits.ncolspadded;
        if (p.xcols == 2) {
            p.impl = GHOST_IMPLEMENTATION_SSE;
        }
        if (p.xcols == 1) {
            p.impl = GHOST_IMPLEMENTATION_PLAIN;
        }
        kernel = ghost_tsmm_inplace_kernels[p];
    }
    
    if (!kernel) {
        PERFWARNING_LOG("Try kernel with non-padded blocks");
        p.xcols = x->traits.ncols;
        if (p.xcols == 2) {
            p.impl = GHOST_IMPLEMENTATION_SSE;
        }
        if (p.xcols == 1) {
            p.impl = GHOST_IMPLEMENTATION_PLAIN;
        }
        kernel = ghost_tsmm_inplace_kernels[p];
    }
    if (!kernel) {
        PERFWARNING_LOG("Try kernel with arbitrary block size");
        p.xcols = -1;
        kernel = ghost_tsmm_inplace_kernels[p];
    }
    if (!kernel) {
        PERFWARNING_LOG("Try plain implementation");
        p.impl = GHOST_IMPLEMENTATION_PLAIN;
        kernel = ghost_tsmm_inplace_kernels[p];
    }

    if (!kernel) {
        INFO_LOG("Could not find in-place TSMM kernel with %d %d %d!",p.impl,p.dt,p.xcols);
        return GHOST_ERR_NOT_IMPLEMENTED;
    }


    return kernel(x,w,alpha);

}
