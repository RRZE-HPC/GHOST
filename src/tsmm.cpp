#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/densemat.h"
#include "ghost/util.h"
#include "ghost/math.h"
#include "ghost/tsmm.h"
#include "ghost/tsmm_gen.h"
#include "ghost/tsmm_avx_gen.h"
#include "ghost/tsmm_sse_gen.h"
#include "ghost/tsmm_cu_gen.h"

#include <map>

using namespace std;

static bool operator<(const ghost_tsmm_parameters_t &a, const ghost_tsmm_parameters_t &b) 
{ 
    return ghost_hash(a.dt,a.xcols,ghost_hash(a.vcols,a.impl,ghost_hash(a.xstor,a.wstor,a.alignment))) < ghost_hash(b.dt,b.xcols,ghost_hash(b.vcols,b.impl,ghost_hash(b.xstor,b.wstor,b.alignment))); 
}

static map<ghost_tsmm_parameters_t, ghost_tsmm_kernel_t> ghost_tsmm_kernels;

ghost_error_t ghost_tsmm_valid(ghost_densemat_t *x, ghost_densemat_t *v,  const char * transv, 
ghost_densemat_t *w, const char *transw, void *alpha, void *beta, int reduce, int printerror)
{
    if (x->traits.datatype != v->traits.datatype || x->traits.datatype != w->traits.datatype) {
        if (printerror) {
            ERROR_LOG("Different data types!");
        }
        return GHOST_ERR_INVALID_ARG;
    }
    if (x == v) {
        if (printerror) {
           ERROR_LOG("x must not be equal to v!");
        }
        return GHOST_ERR_INVALID_ARG;
    }
    if ((x->traits.flags & GHOST_DENSEMAT_SCATTERED) || (v->traits.flags & GHOST_DENSEMAT_SCATTERED) || (w->traits.flags & GHOST_DENSEMAT_SCATTERED)) {
        if (printerror) {
            ERROR_LOG("Scattered views not supported!");
        }
        return GHOST_ERR_INVALID_ARG;
    }
    if (reduce != GHOST_GEMM_NO_REDUCE) { 
        if (printerror) {
            ERROR_LOG("Only NO_REDUCE valid!");
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

    UNUSED(alpha);
    UNUSED(beta);

    return GHOST_SUCCESS;
} 


ghost_error_t ghost_tsmm(ghost_densemat_t *x, ghost_densemat_t *v, ghost_densemat_t *w, void *alpha, void *beta)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret;

    if ((ret = ghost_tsmm_valid(x,v,"N",w,"N",alpha,beta,GHOST_GEMM_NO_REDUCE,1)) != GHOST_SUCCESS) {
        INFO_LOG("TSMM cannot be applied. Checking whether GEMM is fine!");
        if ((ret = ghost_gemm_valid(x,v,"N",w,"N",alpha,beta,GHOST_GEMM_NO_REDUCE,GHOST_GEMM_DEFAULT,1)) != GHOST_SUCCESS) {
            ERROR_LOG("GEMM cannot be applied!");
            return ret;
        } else {
            return ghost_gemm(x,v,"N",w,"N",alpha,beta,GHOST_GEMM_NO_REDUCE,GHOST_GEMM_NOT_SPECIAL);
        }
    }
    
    if (ghost_tsmm_kernels.empty()) {
#include "tsmm.def"
#include "tsmm_avx.def"
#include "tsmm_sse.def"
#ifdef GHOST_HAVE_CUDA
#include "tsmm_cu.def"
#endif
    }

    ghost_tsmm_parameters_t p;
    p.dt = x->traits.datatype;
    p.alignment = GHOST_ALIGNED;
    
    ghost_tsmm_kernel_t kernel = NULL;
#ifdef GHOST_HAVE_MIC
    p.impl = GHOST_IMPLEMENTATION_MIC;
#elif defined(GHOST_HAVE_AVX)
    p.impl = GHOST_IMPLEMENTATION_AVX;
#elif defined(GHOST_HAVE_SSE)
    p.impl = GHOST_IMPLEMENTATION_SSE;
#else
    p.impl = GHOST_IMPLEMENTATION_PLAIN;
#endif
#ifdef GHOST_HAVE_CUDA
    if (x->traits.location & GHOST_LOCATION_DEVICE) {
        p.impl = GHOST_IMPLEMENTATION_CUDA;
        p.dt = GHOST_DT_ANY;
        p.alignment = GHOST_UNALIGNED;
    }
#endif

    p.xstor = x->traits.storage;
    p.wstor = w->traits.storage;

    p.xcols = x->traits.ncols;
    p.vcols = v->traits.ncols;
    
    if (p.vcols < 4 || p.xcols < 4) {
        p.impl = GHOST_IMPLEMENTATION_PLAIN;
    } else if (p.vcols % 4 || p.xcols % 4) {
        if (!(x->traits.flags & GHOST_DENSEMAT_VIEW) && (!(v->traits.ncolspadded % 4) && !(x->traits.ncolspadded % 4))) {
            p.vcols = v->traits.ncolspadded;
            p.xcols = x->traits.ncolspadded;
        } else {
            if (p.xcols == 2) {
#ifdef GHOST_HAVE_SSE
                PERFWARNING_LOG("Use SSE for ncols==2");
                p.impl = GHOST_IMPLEMENTATION_SSE;
#endif
            }
            if (p.xcols == 1) {
                PERFWARNING_LOG("Use plain for ncols==1");
                p.impl = GHOST_IMPLEMENTATION_PLAIN;
            }
            if ((p.xcols % 4 || p.vcols % 4) && (p.impl != GHOST_IMPLEMENTATION_CUDA)) {
                PERFWARNING_LOG("Use SSE for non-multiple of four");
                p.impl = GHOST_IMPLEMENTATION_SSE;
            }
            if ((p.xcols % 2 || p.vcols % 2) && (p.impl != GHOST_IMPLEMENTATION_CUDA)) {
                PERFWARNING_LOG("Use plain for non-even column count");
                p.impl = GHOST_IMPLEMENTATION_PLAIN;
            }
        }
    }

    void *xptr, *vptr, *wptr;
    ghost_densemat_valptr(x,&xptr);
    ghost_densemat_valptr(v,&vptr);
    ghost_densemat_valptr(w,&wptr);

    if (p.impl == GHOST_IMPLEMENTATION_SSE) {
        if (!IS_ALIGNED(xptr,16) || !IS_ALIGNED(vptr,16) || !IS_ALIGNED(wptr,16)) {
            p.alignment = GHOST_UNALIGNED;
            PERFWARNING_LOG("Switching to the unaligned kernel!");
        }
    }
    if (p.impl == GHOST_IMPLEMENTATION_AVX) {
        if (!IS_ALIGNED(xptr,32) || !IS_ALIGNED(vptr,32) || !IS_ALIGNED(wptr,32)) {
            p.alignment = GHOST_UNALIGNED;
            PERFWARNING_LOG("Switching to the unaligned kernel!");
        }
    }
    if (p.impl == GHOST_IMPLEMENTATION_PLAIN) {
        if (!IS_ALIGNED(xptr,64) || !IS_ALIGNED(vptr,64) || !IS_ALIGNED(wptr,64)) {
            p.alignment = GHOST_UNALIGNED;
            PERFWARNING_LOG("Switching to the unaligned kernel!");
        }
    }
    kernel = ghost_tsmm_kernels[p];
    
    if (!kernel) {
        PERFWARNING_LOG("Try kernel with fixed xcols and arbitrary vcols");
        p.xcols = x->traits.ncols;
        p.vcols = -1;
        kernel = ghost_tsmm_kernels[p];
    }

    if (!kernel) {
        PERFWARNING_LOG("Try kernel with fixed vcols and arbitrary xcols");
        p.xcols = -1;
        p.vcols = v->traits.ncols;
        kernel = ghost_tsmm_kernels[p];
    }

    if (!kernel) {
        PERFWARNING_LOG("Try kernel with arbitrary block sizes");
        p.xcols = -1;
        p.vcols = -1;
        kernel = ghost_tsmm_kernels[p];
    }

    if (!kernel) {
        PERFWARNING_LOG("Try plain implementation");
        p.impl = GHOST_IMPLEMENTATION_PLAIN;
        if (!IS_ALIGNED(xptr,64) || !IS_ALIGNED(vptr,64) || !IS_ALIGNED(wptr,64)) {
            p.alignment = GHOST_UNALIGNED;
            PERFWARNING_LOG("Switching to the unaligned kernel!");
        } else {
            p.alignment = GHOST_ALIGNED;
        }
        p.xcols = x->traits.ncols;
        p.vcols = v->traits.ncols;
        kernel = ghost_tsmm_kernels[p];
    }
    if (!kernel) {
        PERFWARNING_LOG("Try kernel with fixed xcols and arbitrary vcols");
        p.xcols = x->traits.ncols;
        p.vcols = -1;
        kernel = ghost_tsmm_kernels[p];
    }

    if (!kernel) {
        PERFWARNING_LOG("Try kernel with fixed vcols and arbitrary xcols");
        p.xcols = -1;
        p.vcols = v->traits.ncols;
        kernel = ghost_tsmm_kernels[p];
    }

    if (!kernel) {
        PERFWARNING_LOG("Try kernel with arbitrary block sizes");
        p.xcols = -1;
        p.vcols = -1;
        kernel = ghost_tsmm_kernels[p];
    }


    if (!kernel) {
        INFO_LOG("Could not find TSMM kernel with %d %d %d %d %d %d %d!",p.alignment,p.impl,p.dt,p.xcols,p.vcols,p.xstor,p.wstor);
        return GHOST_ERR_NOT_IMPLEMENTED;
    }

    ret = kernel(x,v,w,alpha,beta);
    
#ifdef GHOST_HAVE_INSTR_TIMING
    ghost_gemm_perf_args_t tsmm_perfargs;
    tsmm_perfargs.xcols = p.xcols;
    tsmm_perfargs.vcols = p.vcols;
    if (v->context) {
        tsmm_perfargs.vrows = v->context->gnrows;
    } else {
        tsmm_perfargs.vrows = v->traits.nrows;
    }
    tsmm_perfargs.dt = x->traits.datatype;
    ghost_timing_set_perfFunc(__ghost_functag,ghost_tsmm_perf_GBs,(void *)&tsmm_perfargs,sizeof(tsmm_perfargs),"GB/s");
    ghost_timing_set_perfFunc(__ghost_functag,ghost_gemm_perf_GFs,(void *)&tsmm_perfargs,sizeof(tsmm_perfargs),"GF/s");
#endif

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
}

int ghost_tsmm_perf_GBs(double *perf, double time, void *varg)
{
    size_t size;
    ghost_gemm_perf_args_t arg = *(ghost_gemm_perf_args_t *)varg;
    
    ghost_datatype_size(&size,arg.dt);

    *perf = size*(arg.vrows*arg.vcols+arg.vrows*arg.xcols+arg.vcols*arg.xcols)/1.e9/time;

    return 0;
}

