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

static bool operator<(const ghost_tsmttsm_parameters_t &a, const ghost_tsmttsm_parameters_t &b) 
{ 
    return ghost_hash(a.dt,a.wcols,ghost_hash(a.vcols,a.impl,0)) < ghost_hash(b.dt,b.wcols,ghost_hash(b.vcols,b.impl,0)); 
}

static map<ghost_tsmttsm_parameters_t, ghost_tsmttsm_kernel_t> ghost_tsmttsm_kernels;


ghost_error_t ghost_tsmttsm_valid(ghost_densemat_t *x, ghost_densemat_t *v, const char * transv, 
ghost_densemat_t *w, const char *transw, void *alpha, void *beta, int reduce, int printerror) 
{
    if (w->traits.storage != GHOST_DENSEMAT_ROWMAJOR) {
        if (printerror) {
            ERROR_LOG("w must be stored row-major!");
        }
        return GHOST_ERR_INVALID_ARG;
    }
    if (v->traits.storage != GHOST_DENSEMAT_ROWMAJOR) {
        if (printerror) {
            ERROR_LOG("v must be stored row-major!");
        }
        return GHOST_ERR_INVALID_ARG;
    }
    if (x->traits.storage != GHOST_DENSEMAT_COLMAJOR) {
        if (printerror) {
            ERROR_LOG("x must be stored row-major!");
        }
        return GHOST_ERR_INVALID_ARG;
    }
    if (v->traits.datatype != w->traits.datatype || v->traits.datatype != x->traits.datatype) {
        if (printerror) {
            ERROR_LOG("Different data types!");
        }
        return GHOST_ERR_INVALID_ARG;
    }
    if (v->traits.flags & GHOST_DENSEMAT_SCATTERED || w->traits.flags & GHOST_DENSEMAT_SCATTERED || x->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        if (printerror) {
            ERROR_LOG("Scattered densemats not supported!");
        }
        return GHOST_ERR_INVALID_ARG;
    }
    if (reduce != GHOST_GEMM_ALL_REDUCE) {
        if (printerror) {
            ERROR_LOG("Only Allreduce supported currently!");
        }
        return GHOST_ERR_INVALID_ARG;
    }
    if (!strncasecmp(transv,"N",1)) {
        if (printerror) {
            ERROR_LOG("v must be transposed!");
        }
        return GHOST_ERR_INVALID_ARG;
    }
    if (strncasecmp(transw,"N",1)) {
        if (printerror) {
            ERROR_LOG("w must not be transposed!");
        }
        return GHOST_ERR_INVALID_ARG;
    }

    UNUSED(alpha);
    UNUSED(beta);

    return GHOST_SUCCESS;
}


ghost_error_t ghost_tsmttsm(ghost_densemat_t *x, ghost_densemat_t *v, ghost_densemat_t *w, void *alpha, void *beta,int reduce,int conjv)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret;

    if ((ret = ghost_tsmttsm_valid(x,v,"T",w,"N",alpha,beta,reduce,1)) != GHOST_SUCCESS) {
        return ret;
    }
    
    if (ghost_tsmttsm_kernels.empty()) {
#include "tsmttsm.def"
#include "tsmttsm_avx.def"
    }
    
    ghost_tsmttsm_parameters_t p;
    ghost_tsmttsm_kernel_t kernel = NULL;

#ifdef GHOST_HAVE_MIC
    p.impl = GHOST_IMPLEMENTATION_MIC;
#elif defined(GHOST_HAVE_AVX)
    p.impl = GHOST_IMPLEMENTATION_AVX;
#elif defined(GHOST_HAVE_SSE)
    p.impl = GHOST_IMPLEMENTATION_SSE;
#endif
    if (x->traits.ncolspadded < 4 || x->traits.flags & GHOST_DENSEMAT_VIEW) {
        p.impl = GHOST_IMPLEMENTATION_PLAIN;
    }

    p.dt = x->traits.datatype;
    
    p.vcols = v->traits.ncols;
    p.wcols = w->traits.ncols;
    kernel = ghost_tsmttsm_kernels[p];
    
    if (!kernel) {
        PERFWARNING_LOG("Try kernel with arbitrary block sizes");
        p.wcols = -1;
        p.vcols = -1;
        kernel = ghost_tsmttsm_kernels[p];
    }
    
    if (!kernel) {
        PERFWARNING_LOG("Try plain implementation");
        p.impl = GHOST_IMPLEMENTATION_PLAIN;
    }
    kernel = ghost_tsmttsm_kernels[p];
    
    if (!kernel) {
        INFO_LOG("Could not find TSMTTSM kernel with %d %d %d. Fallback to GEMM",p.dt,p.wcols,p.vcols);
        return GHOST_ERR_INVALID_ARG;
        
            //return ghost_gemm(x,v,"T",w,"N",alpha,beta,GHOST_GEMM_ALL_REDUCE,GHOST_GEMM_NOT_SPECIAL);
    }
    
    ret = kernel(x,v,w,alpha,beta,conjv);

#ifdef GHOST_HAVE_INSTR_TIMING
    ghost_gemm_perf_args_t tsmttsm_perfargs;
    tsmttsm_perfargs.xcols = p.wcols;
    tsmttsm_perfargs.vcols = p.vcols;
    tsmttsm_perfargs.vrows = v->context->gnrows;
    tsmttsm_perfargs.dt = x->traits.datatype;
    ghost_timing_set_perfFunc(__ghost_functag,ghost_gemm_perf_GFs,(void *)&tsmttsm_perfargs,sizeof(tsmttsm_perfargs),"GF/s");
#endif

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
}

