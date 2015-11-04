#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/densemat.h"
#include "ghost/util.h"
#include "ghost/tsmm_inplace.h"
#include "ghost/tsmm_inplace_gen.h"
#include "ghost/tsmm_inplace.h"
#include "ghost/math.h"
#include "ghost/timing.h"
#include <map>

using namespace std;

static bool operator<(const ghost_tsmm_inplace_parameters_t &a, const ghost_tsmm_inplace_parameters_t &b) 
{ 
    return ghost_hash(a.dt,a.ncolsin,ghost_hash(a.ncolsout,a.impl,0)) < ghost_hash(b.dt,b.ncolsin,ghost_hash(b.ncolsout,b.impl,0)); 
}

static map<ghost_tsmm_inplace_parameters_t, ghost_tsmm_inplace_kernel_t> ghost_tsmm_inplace_kernels;

ghost_error_t ghost_tsmm_inplace_valid(ghost_densemat_t *x, ghost_densemat_t *v, const char * transv, 
ghost_densemat_t *w, const char *transw, void *alpha, void *beta, int reduce, int printerror)
{
    if (x->traits.location != GHOST_LOCATION_HOST || v->traits.location != GHOST_LOCATION_HOST) {
        if (printerror) {
            ERROR_LOG("TSMM-inplace only implemented for host densemats!");
        }
        return GHOST_ERR_INVALID_ARG;
    }

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
    if (reduce != GHOST_GEMM_NO_REDUCE) { 
        if (printerror) {
            ERROR_LOG("Only NO_REDUCE valid!");
        }
        return GHOST_ERR_INVALID_ARG;
    }

    UNUSED(alpha);
    UNUSED(beta);

    return GHOST_SUCCESS;

}

ghost_error_t ghost_tsmm_inplace(ghost_densemat_t *x, ghost_densemat_t *w, void *alpha, void *beta)
{
    ghost_error_t ret;
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);

    if ((ret = ghost_tsmm_inplace_valid(x,x,"N",w,"N",alpha,beta,GHOST_GEMM_NO_REDUCE,1)) != GHOST_SUCCESS) {
        INFO_LOG("TSMM-inplace cannot be applied. Checking whether GEMM is fine!");
        if ((ret = ghost_gemm_valid(x,x,"N",w,"N",alpha,beta,GHOST_GEMM_NO_REDUCE,GHOST_GEMM_DEFAULT,1)) != GHOST_SUCCESS) {
            ERROR_LOG("GEMM cannot be applied!");
            return ret;
        } else {
            return ghost_gemm(x,x,"N",w,"N",alpha,beta,GHOST_GEMM_NO_REDUCE,GHOST_GEMM_NOT_SPECIAL);
        }
    }
    
    if (ghost_tsmm_inplace_kernels.empty()) {
#include "tsmm_inplace.def"
    }
    
    ghost_tsmm_inplace_parameters_t p;
    ghost_tsmm_inplace_kernel_t kernel = NULL;

    p.impl = GHOST_IMPLEMENTATION_PLAIN;

    p.dt = x->traits.datatype;
    p.ncolsin = w->traits.nrows;
    p.ncolsout = w->traits.ncols;
   
    INFO_LOG("in %d out %d",p.ncolsin,p.ncolsout); 
    kernel = ghost_tsmm_inplace_kernels[p];
    
    if (!kernel) {
        PERFWARNING_LOG("Try kernel with arbitrary block size ncolsin");
        p.ncolsin = -1;
        p.ncolsout = w->traits.ncols;
        kernel = ghost_tsmm_inplace_kernels[p];
    }
    
    if (!kernel) {
        PERFWARNING_LOG("Try kernel with arbitrary block size ncolsout");
        p.ncolsin = w->traits.nrows;
        p.ncolsout = -1;
        kernel = ghost_tsmm_inplace_kernels[p];
    }
    
    if (!kernel) {
        PERFWARNING_LOG("Try kernel with arbitrary block sizes");
        p.ncolsin = -1;
        p.ncolsout = -1;
        kernel = ghost_tsmm_inplace_kernels[p];
    }

    if (!kernel) {
        INFO_LOG("Could not find in-place TSMM kernel with %d %d %d %d!",p.impl,p.dt,p.ncolsin,p.ncolsout);
        GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);
        return GHOST_ERR_NOT_IMPLEMENTED;
    }

    ret = kernel(x,w,alpha,beta);

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);


    return ret;

}
