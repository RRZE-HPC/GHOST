#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/densemat.h"
#include "ghost/util.h"
#include "ghost/math.h"
#include "ghost/tsmm.h"
#include "ghost/tsmm_var2_plain_gen.h"
#include "ghost/tsmm_var2_sse_gen.h"
#include "ghost/tsmm_plain_gen.h"
#include "ghost/tsmm_varincols_plain_gen.h"
#include "ghost/tsmm_varoutcols_plain_gen.h"
#include "ghost/tsmm_var2_avx_gen.h"
#include "ghost/tsmm_avx_gen.h"
#include "ghost/tsmm_sse_gen.h"
#ifdef GHOST_HAVE_CUDA
#include "ghost/tsmm_var2_cu_gen.h"
#include "ghost/tsmm_cu_gen.h"
#endif
#include "ghost/timing.h"
#include "ghost/machine.h"
#include "ghost/constants.h"

#include <unordered_map>
#include <vector>

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
            return ghost_hash(a.dt,a.xcols,ghost_hash(a.vcols,a.impl,ghost_hash(a.xstor,a.alignment,ghost_hash(a.unroll,a.multipleof,999))));
        }
    };
}

static bool operator==(const ghost_tsmm_parameters& a, const ghost_tsmm_parameters& b)
{
    return a.dt == b.dt && a.xcols == b.xcols && a.vcols == b.vcols && a.impl == b.impl && a.xstor == b.xstor && a.alignment == b.alignment && a.unroll == b.unroll && a.multipleof == b.multipleof;
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


ghost_error ghost_tsmm(ghost_densemat *x, ghost_densemat *v, ghost_densemat *w_in, void *alpha, void *beta)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error ret;

    if ((ret = ghost_tsmm_valid(x,v,"N",w_in,"N",alpha,beta,GHOST_GEMM_NO_REDUCE,1)) != GHOST_SUCCESS) {
        INFO_LOG("TSMM cannot be applied. Checking whether GEMM is fine!");
        if ((ret = ghost_gemm_valid(x,v,"N",w_in,"N",alpha,beta,GHOST_GEMM_NO_REDUCE,GHOST_GEMM_DEFAULT,1)) != GHOST_SUCCESS) {
            ERROR_LOG("GEMM cannot be applied!");
            return ret;
        } else {
            return ghost_gemm(x,v,"N",w_in,"N",alpha,beta,GHOST_GEMM_NO_REDUCE,GHOST_GEMM_NOT_SPECIAL);
        }
    }
    
    if (ghost_tsmm_kernels.empty()) {
#include "tsmm_var2_plain.def"
#include "tsmm_avx.def"
#include "tsmm_var2_avx.def"
#include "tsmm_varincols_plain.def"
#include "tsmm_varoutcols_plain.def"
#include "tsmm_sse.def"
#include "tsmm_var2_sse.def"
#include "tsmm_plain.def"
#ifdef GHOST_HAVE_CUDA
#include "tsmm_cu.def"
#include "tsmm_var2_cu.def"
#endif
    }

    ghost_densemat *w;
    ghost_tsmm_parameters p;
    ghost_alignment opt_align;
    int opt_unroll;
    ghost_tsmm_kernel kernel = NULL;

    if (w_in->traits.storage == GHOST_DENSEMAT_COLMAJOR) {
        w = w_in;
    } else {
        PERFWARNING_LOG("Need to transpose input densemat w!");
        ghost_densemat_traits wtraits = w_in->traits;
        wtraits.flags &= (ghost_densemat_flags)~GHOST_DENSEMAT_VIEW;
        wtraits.storage = GHOST_DENSEMAT_COLMAJOR;
        ghost_densemat_create(&w,ghost_map_create_light(w_in->map->dim,w_in->map->mpicomm),wtraits);
        ghost_densemat_init_densemat(w,w_in,0,0);
    }


    std::vector<ghost_densemat_storage> try_xstor;
    
    // fix properties
    if (x->traits.ncols == 1 && x->stride == 1 && v->traits.ncols == 1 && v->stride == 1) {    
        try_xstor.push_back(GHOST_DENSEMAT_COLMAJOR);
    }
    try_xstor.push_back(x->traits.storage);

    // possible implementations
    std::vector<ghost_implementation> try_impl;
#ifdef GHOST_HAVE_CUDA
    if (x->traits.location & GHOST_LOCATION_DEVICE && x->traits.compute_at != GHOST_LOCATION_HOST) {
        try_impl.push_back(GHOST_IMPLEMENTATION_CUDA);
    } else {
#endif
#ifdef GHOST_BUILD_MIC
        try_impl.push_back(GHOST_IMPLEMENTATION_MIC);
        try_impl.push_back(GHOST_IMPLEMENTATION_PLAIN);
#elif defined(GHOST_BUILD_AVX2)
        try_impl.push_back(GHOST_IMPLEMENTATION_AVX2);
        try_impl.push_back(GHOST_IMPLEMENTATION_AVX);
        try_impl.push_back(GHOST_IMPLEMENTATION_SSE);
        try_impl.push_back(GHOST_IMPLEMENTATION_PLAIN);
#elif defined(GHOST_BUILD_AVX)
        try_impl.push_back(GHOST_IMPLEMENTATION_AVX);
        try_impl.push_back(GHOST_IMPLEMENTATION_SSE);
        try_impl.push_back(GHOST_IMPLEMENTATION_PLAIN);
#elif defined(GHOST_BUILD_SSE)
        try_impl.push_back(GHOST_IMPLEMENTATION_SSE);
        try_impl.push_back(GHOST_IMPLEMENTATION_PLAIN);
#else
        try_impl.push_back(GHOST_IMPLEMENTATION_PLAIN);
#endif
#ifdef GHOST_HAVE_CUDA
    }
#endif
    
    
    // alignment of large input data
    // the alignment of the result array does not matter because we can easily re-allocate it accordingly
    int al = ghost_machine_alignment();
    if (IS_ALIGNED(x->val,al) && IS_ALIGNED(v->val,al) && ((x->traits.storage == GHOST_DENSEMAT_COLMAJOR || (x->traits.ncols == 1 && x->stride == 1)) || (!((x->stride*x->elSize) % al) && !((v->stride*v->elSize) % al)))) {
        opt_align = GHOST_ALIGNED;
    } else {
        opt_align = GHOST_UNALIGNED;
    }
    
    ghost_lidx try_xcols[2] = {x->traits.ncols,-1};
    ghost_lidx try_vcols[2] = {v->traits.ncols,-1};
    ghost_datatype try_dt[2] = {v->traits.datatype,GHOST_DT_ANY};

    if (w->traits.flags & GHOST_DENSEMAT_VIEW || v->traits.flags & GHOST_DENSEMAT_VIEW) {
        opt_unroll = 1;
    } else {
        opt_unroll = 2;
    }

#ifdef GHOST_HAVE_CUDA
    if (x->traits.location & GHOST_LOCATION_DEVICE && x->traits.compute_at != GHOST_LOCATION_HOST) {
        try_dt[0] = GHOST_DT_ANY;
        opt_align = GHOST_UNALIGNED;
    }
#endif

    std::vector<ghost_lidx> try_multipleof;
    if (ISPOWEROFTWO(x->traits.ncols) && ISPOWEROFTWO(v->traits.ncols)) {
        ghost_lidx smallerdim = MIN(x->traits.ncols,v->traits.ncols);
        while (smallerdim > 0) {
            try_multipleof.push_back(smallerdim);
            smallerdim /= 2;
        }
    } else {
        try_multipleof.push_back(1);
    }

    
    int n_xcols = sizeof(try_xcols)/sizeof(ghost_lidx); 
    int n_vcols = sizeof(try_vcols)/sizeof(ghost_lidx); 
    int n_dt = sizeof(try_dt)/sizeof(ghost_datatype); 
    int pos_xcols, pos_vcols, pos_dt;
    bool optimal = true; // if we find a kernel with highest specialization grade (regardless unrolling), this remains true and no performance warning gets printed

    for (pos_xcols = 0; pos_xcols < n_xcols; pos_xcols++) {  
        for (pos_vcols = 0; pos_vcols < n_vcols; pos_vcols++) {  
            for (std::vector<ghost_densemat_storage>::iterator xstor = try_xstor.begin(); xstor != try_xstor.end(); xstor++) {
                for (std::vector<ghost_implementation>::iterator impl = try_impl.begin(); impl != try_impl.end(); impl++) {
                    for (p.alignment = opt_align; (int)p.alignment >= GHOST_UNALIGNED; p.alignment = (ghost_alignment)((int)p.alignment-1)) {
                        for (std::vector<ghost_lidx>::iterator mult = try_multipleof.begin(); mult != try_multipleof.end(); mult++) {
                            for (p.unroll = opt_unroll; p.unroll > 0; p.unroll /= 2) {
                                for (pos_dt = 0; pos_dt < n_dt; pos_dt++) {
                                    p.xstor = *xstor;
                                    p.xcols = try_xcols[pos_xcols];
                                    p.vcols = try_vcols[pos_vcols];
                                    p.dt = try_dt[pos_dt];
                                    p.impl = *impl;
                                    p.multipleof = *mult;
                                    INFO_LOG("Try xcols=%s, vcols=%s, impl=%s, %s, unroll=%d, dt=%s, multipleof=%d, storage=%s",
                                            p.xcols==-1?"arbitrary":to_string((long long)p.xcols).c_str(),p.vcols==-1?"arbitrary":to_string((long long)p.vcols).c_str(),
                                            ghost_implementation_string(p.impl),p.alignment==GHOST_UNALIGNED?"unaligned":"aligned",p.unroll,ghost_datatype_string(p.dt),p.multipleof,ghost_densemat_storage_string(*xstor));
                                    kernel = ghost_tsmm_kernels[p];
                                    if (kernel) {
                                        goto end_of_loop;
                                    }
                                }
                            }
                        }
                    }
                    optimal = false;
                }
            }
        }
    }

end_of_loop:

    if (kernel) {
        if (optimal) {
            INFO_LOG("Found kernel with highest specialization grade: dt=%d xcols=%d vcols=%d xstor=%s align=%d unroll=%d impl=%s multipleof=%d",p.dt,p.xcols,p.vcols,ghost_densemat_storage_string(p.xstor),p.alignment,p.unroll,ghost_implementation_string(p.impl),p.multipleof);
        } else {
            PERFWARNING_LOG("Using potentially non-optimal kernel: dt=%d xcols=%d vcols=%d xstor=%s align=%d unroll=%d impl=%s multipleof=%d",p.dt,p.xcols,p.vcols,ghost_densemat_storage_string(p.xstor),p.alignment,p.unroll,ghost_implementation_string(p.impl),p.multipleof);
        }

        ret = kernel(x,v,w,alpha,beta);
    } else {
        PERFWARNING_LOG("Could not find TSMM kernel. Fallback to GEMM");
        ret = ghost_gemm(x,v,"N",w,"N",alpha,beta,GHOST_GEMM_NO_REDUCE,GHOST_GEMM_NOT_SPECIAL);
    }

#ifdef GHOST_INSTR_TIMING
    ghost_gemm_perf_args tsmm_perfargs;
    tsmm_perfargs.n = x->traits.ncols;
    tsmm_perfargs.k = v->traits.ncols;
    tsmm_perfargs.m = v->map->gdim;
    tsmm_perfargs.dt = x->traits.datatype;
    tsmm_perfargs.betaiszero = ghost_iszero(beta,x->traits.datatype);
    tsmm_perfargs.alphaisone = ghost_isone(alpha,x->traits.datatype);
    tsmm_perfargs.aisc = false;
    ghost_timing_set_perfFunc(NULL,__ghost_functag,ghost_gemm_perf_GBs,(void *)&tsmm_perfargs,sizeof(tsmm_perfargs),"GB/s");
    ghost_timing_set_perfFunc(NULL,__ghost_functag,ghost_gemm_perf_GFs,(void *)&tsmm_perfargs,sizeof(tsmm_perfargs),"GF/s");
#endif

    if (w != w_in) {
        ghost_densemat_init_densemat(w_in,w,0,0);
        ghost_densemat_destroy(w);
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);
        
    return ret;
}


