#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/densemat.h"
#include "ghost/util.h"
#include "ghost/math.h"
#include "ghost/tsmm.h"
#include "ghost/tsmm_var2_plain_gen.h"
#include "ghost/tsmm_var2_sse_gen.h"
#include "ghost/tsmm_var2_cu_gen.h"
#include "ghost/tsmm_plain_gen.h"
#include "ghost/tsmm_var1_plain_gen.h"
#include "ghost/tsmm_var2_avx_gen.h"
#include "ghost/tsmm_avx_gen.h"
#include "ghost/tsmm_sse_gen.h"
#include "ghost/tsmm_cu_gen.h"
#include "ghost/timing.h"
#include "ghost/machine.h"
#include "ghost/constants.h"

#include <unordered_map>

using namespace std;

// Hash function for unordered_map
namespace std
{
    template<> struct hash<ghost_tsmm_parameters>
    {
        typedef ghost_tsmm_parameters argument_type;
        typedef std::size_t result_type;
        result_type operator()(argument_type const& a) const
        {
            return ghost_hash(a.dt,a.xcols,ghost_hash(a.vcols,a.impl,ghost_hash(a.xstor,a.wstor,ghost_hash(a.alignment,a.unroll,0))));
        }
    };
}

bool operator==(const ghost_tsmm_parameters& a, const ghost_tsmm_parameters& b)
{
    return a.dt == b.dt && a.xcols == b.xcols && a.vcols == b.vcols && a.impl == b.impl && a.xstor == b.xstor && a.wstor == b.wstor && a.alignment == b.alignment && a.unroll == b.unroll;
}

static unordered_map<ghost_tsmm_parameters, ghost_tsmm_kernel> ghost_tsmm_kernels;

ghost_error ghost_tsmm_valid(ghost_densemat *x, ghost_densemat *v,  const char * transv, 
ghost_densemat *w, const char *transw, void *alpha, void *beta, int reduce, int printerror)
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


ghost_error ghost_tsmm(ghost_densemat *x, ghost_densemat *v, ghost_densemat *w, void *alpha, void *beta)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error ret;

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
#include "tsmm_var2_plain.def"
#include "tsmm_avx.def"
#include "tsmm_var2_avx.def"
#include "tsmm_var1_plain.def"
#include "tsmm_sse.def"
#include "tsmm_var2_sse.def"
#ifdef GHOST_HAVE_CUDA
#include "tsmm_cu.def"
#include "tsmm_var2_cu.def"
#endif
    }

    ghost_tsmm_parameters p;
    p.dt = x->traits.datatype;
    p.alignment = GHOST_ALIGNED;
    
    ghost_tsmm_kernel kernel = NULL;
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

    int simd = ghost_machine_simd_width();
    
    // alignment of large input data
    // the alignment of the w matrix does not matter because we can easily re-allocate it accordingly
    int al = ghost_machine_alignment();
    if (IS_ALIGNED(x->val,al) && IS_ALIGNED(v->val,al) && !((x->stride*x->elSize) % al) && !((v->stride*v->elSize) % al)) {
        p.alignment = GHOST_ALIGNED;
    } else {
        p.alignment = GHOST_UNALIGNED;
    }

    if (x->traits.flags & GHOST_DENSEMAT_VIEW || v->traits.flags & GHOST_DENSEMAT_VIEW) {
        p.unroll = 1;
        if (((p.xcols*v->elSize) % simd) || ((p.vcols*v->elSize) % simd)) {
            p.impl = GHOST_IMPLEMENTATION_PLAIN;
        }
    } else {
        p.unroll = GHOST_MAX_ROWS_UNROLL;
    }
    
    INFO_LOG("Inital search for kernel dt=%d xcols=%d vcols=%d xstor=%d wstor=%d align=%d unroll=%d!",p.dt,p.xcols,p.vcols,p.xstor,p.wstor,p.alignment,p.unroll);
    
    kernel = ghost_tsmm_kernels[p];
    
    if (!kernel) {
        PERFWARNING_LOG("Try plain implementation");
        p.impl = GHOST_IMPLEMENTATION_PLAIN;
        kernel = ghost_tsmm_kernels[p];
    }
    
    if (!kernel) {
        PERFWARNING_LOG("Decrease unroll size");
        while (p.unroll > 1 && !kernel) {
            p.unroll /= 2;
            kernel = ghost_tsmm_kernels[p];
        }
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
        PERFWARNING_LOG("Try unaligned kernel");
        p.alignment = GHOST_UNALIGNED;
        kernel = ghost_tsmm_kernels[p];
    }



    if (!kernel) {
        INFO_LOG("Could not find TSMM kernel with %d %d %d %d %d %d %d!",p.alignment,p.impl,p.dt,p.xcols,p.vcols,p.xstor,p.wstor);
        GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);
        return GHOST_ERR_INVALID_ARG;
    }

    ret = kernel(x,v,w,alpha,beta);
    
#ifdef GHOST_HAVE_INSTR_TIMING
    ghost_gemm_perf_args_t tsmm_perfargs;
    tsmm_perfargs.n = p.xcols;
    tsmm_perfargs.k = p.vcols;
    if (v->context) {
        tsmm_perfargs.m = v->context->gnrows;
    } else {
        tsmm_perfargs.m = v->traits.nrows;
    }
    tsmm_perfargs.dt = x->traits.datatype;
    tsmm_perfargs.betaiszero = ghost_iszero(beta,p.dt);
    tsmm_perfargs.alphaisone = ghost_isone(alpha,p.dt);
    ghost_timing_set_perfFunc(NULL,__ghost_functag,ghost_gemm_perf_GBs,(void *)&tsmm_perfargs,sizeof(tsmm_perfargs),"GB/s");
    ghost_timing_set_perfFunc(NULL,__ghost_functag,ghost_gemm_perf_GFs,(void *)&tsmm_perfargs,sizeof(tsmm_perfargs),"GF/s");
#endif

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
}


