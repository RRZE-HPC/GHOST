#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/densemat.h"
#include "ghost/util.h"
#include "ghost/tsmm_inplace.h"
#include "ghost/tsmm_inplace_cu.h"
#include "ghost/tsmm_inplace_plain_gen.h"
#include "ghost/tsmm_inplace_varincols_plain_gen.h"
#include "ghost/tsmm_inplace_varoutcols_plain_gen.h"
#include "ghost/tsmm_inplace_var2_plain_gen.h"
#include "ghost/tsmm_inplace_var2_cu_gen.h"
#include "ghost/tsmm_inplace_cu_gen.h"
#include "ghost/tsmm_inplace.h"
#include "ghost/math.h"
#include "ghost/timing.h"
#include "ghost/machine.h"

#include <unordered_map>
#include <vector>

using namespace std;

// Hash function for unordered_map
namespace std
{
    template<> struct hash<ghost_tsmm_inplace_parameters>
    {
        typedef ghost_tsmm_inplace_parameters argument_type;
        typedef std::size_t result_type;
        result_type operator()(argument_type const& a) const
        {
            return ghost_hash(a.dt,a.ncolsin,ghost_hash(a.ncolsout,a.impl,0));
        }
    };
}

static bool operator==(const ghost_tsmm_inplace_parameters& a, const ghost_tsmm_inplace_parameters& b)
{
    return a.dt == b.dt && a.ncolsin == b.ncolsin && a.ncolsout == b.ncolsout && a.impl == b.impl;
}

static unordered_map<ghost_tsmm_inplace_parameters, ghost_tsmm_inplace_kernel> ghost_tsmm_inplace_kernels;

ghost_error ghost_tsmm_inplace_valid(ghost_densemat *x, ghost_densemat *v, const char * transv, 
ghost_densemat *w, const char *transw, void *alpha, void *beta, int reduce, int printerror)
{
/*    if (x->traits.location != GHOST_LOCATION_HOST || v->traits.location != GHOST_LOCATION_HOST) {
        if (printerror) {
            ERROR_LOG("TSMM-inplace only implemented for host densemats!");
        }
        return GHOST_ERR_INVALID_ARG;
    }
*/
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

ghost_error ghost_tsmm_inplace(ghost_densemat *x, ghost_densemat *w, void *alpha, void *beta)
{
    ghost_error ret;
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);

    if ((ret = ghost_tsmm_inplace_valid(x,x,"N",w,"N",alpha,beta,GHOST_GEMM_NO_REDUCE,1)) != GHOST_SUCCESS) {
        INFO_LOG("TSMM-inplace cannot be applied. Checking whether GEMM is fine!");
        if ((ret = ghost_gemm_valid(x,x,"N",w,"N",alpha,beta,GHOST_GEMM_NO_REDUCE,NULL,GHOST_GEMM_DEFAULT,1)) != GHOST_SUCCESS) {
            ERROR_LOG("GEMM cannot be applied!");
            return ret;
        } else {
            return ghost_gemm(x,x,"N",w,"N",alpha,beta,GHOST_GEMM_NO_REDUCE,NULL,GHOST_GEMM_NOT_SPECIAL);
        }
    }
    
    if (ghost_tsmm_inplace_kernels.empty()) {
#include "tsmm_inplace_plain.def"
#include "tsmm_inplace_varincols_plain.def"
#include "tsmm_inplace_varoutcols_plain.def"
#include "tsmm_inplace_var2_plain.def"
#ifdef GHOST_HAVE_CUDA
#include "tsmm_inplace_cu.def"
#include "tsmm_inplace_var2_cu.def"
#endif
    }
    
    ghost_tsmm_inplace_parameters p;
    ghost_alignment opt_align;
    ghost_tsmm_inplace_kernel kernel = NULL;
    
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
    if (IS_ALIGNED(x->val,al) && !((x->stride*x->elSize) % al)) {
        opt_align = GHOST_ALIGNED;
    } else {
        opt_align = GHOST_UNALIGNED;
    }
    
    ghost_lidx try_ncolsout[2] = {w->traits.ncols,-1};
    ghost_lidx try_ncolsin[2] = {w->traits.nrows,-1};
    ghost_datatype try_dt[2] = {x->traits.datatype,GHOST_DT_ANY};

#ifdef GHOST_HAVE_CUDA
    if (x->traits.location & GHOST_LOCATION_DEVICE && x->traits.compute_at != GHOST_LOCATION_HOST) {
        try_dt[0] = GHOST_DT_ANY;
        opt_align = GHOST_UNALIGNED;
    }
#endif

    
    int n_ncolsout = sizeof(try_ncolsout)/sizeof(ghost_lidx); 
    int n_ncolsin = sizeof(try_ncolsin)/sizeof(ghost_lidx); 
    int n_dt = sizeof(try_dt)/sizeof(ghost_datatype); 
    int pos_ncolsout, pos_ncolsin, pos_dt;
    bool optimal = true; // if we find a kernel with highest specialization grade, this remains true and no performance warning gets printed

    for (pos_ncolsout = 0; pos_ncolsout < n_ncolsout; pos_ncolsout++) {  
        for (pos_ncolsin = 0; pos_ncolsin < n_ncolsin; pos_ncolsin++) {  
            for (std::vector<ghost_implementation>::iterator impl = try_impl.begin(); impl != try_impl.end(); impl++) {
                for (p.alignment = opt_align; (int)p.alignment >= GHOST_UNALIGNED; p.alignment = (ghost_alignment)((int)p.alignment-1)) {
                    for (pos_dt = 0; pos_dt < n_dt; pos_dt++) {
                        p.ncolsout = try_ncolsout[pos_ncolsout];
                        p.ncolsin = try_ncolsin[pos_ncolsin];
                        p.dt = try_dt[pos_dt];
                        p.impl = *impl;
                        INFO_LOG("Try ncolsout=%s, ncolsin=%s, impl=%s, %s, dt=%s",
                                p.ncolsout==-1?"arbitrary":to_string((long long)p.ncolsout).c_str(),p.ncolsin==-1?"arbitrary":to_string((long long)p.ncolsin).c_str(),
                                ghost_implementation_string(p.impl),p.alignment==GHOST_UNALIGNED?"unaligned":"aligned",ghost_datatype_string(p.dt));
                        kernel = ghost_tsmm_inplace_kernels[p];
                        if (kernel) {
                            goto end_of_loop;
                        }
                    }
                }
                optimal = false;
            }
        }
    }

end_of_loop:

    if (kernel) {
        if (optimal) {
            INFO_LOG("Found kernel with highest specialization grade: dt=%d ncolsout=%d ncolsin=%d align=%d impl=%s",p.dt,p.ncolsout,p.ncolsin,p.alignment,ghost_implementation_string(p.impl));
        } else {
            PERFWARNING_LOG("Using potentially non-optimal kernel: dt=%d ncolsout=%d ncolsin=%d align=%d impl=%s",p.dt,p.ncolsout,p.ncolsin,p.alignment,ghost_implementation_string(p.impl));
        }

        ret = kernel(x,w,alpha,beta);
    } else {
        PERFWARNING_LOG("Could not find TSMM-inplace kernel. Fallback to GEMM");
        ret = ghost_gemm(x,x,"N",w,"N",alpha,beta,GHOST_GEMM_NO_REDUCE,NULL,GHOST_GEMM_NOT_SPECIAL);
    }

#ifdef GHOST_INSTR_TIMING
    ghost_gemm_perf_args tsmm_inplace_perfargs;
    tsmm_inplace_perfargs.n = w->traits.ncols;
    tsmm_inplace_perfargs.k = w->traits.nrows;
    tsmm_inplace_perfargs.m = x->traits.gnrows;
    tsmm_inplace_perfargs.dt = x->traits.datatype;
    tsmm_inplace_perfargs.betaiszero = ghost_iszero(beta,x->traits.datatype);
    tsmm_inplace_perfargs.alphaisone = ghost_isone(alpha,x->traits.datatype);
    tsmm_inplace_perfargs.aisc = true;
    ghost_timing_set_perfFunc(NULL,__ghost_functag,ghost_gemm_perf_GBs,(void *)&tsmm_inplace_perfargs,sizeof(tsmm_inplace_perfargs),"GB/s");
    ghost_timing_set_perfFunc(NULL,__ghost_functag,ghost_gemm_perf_GFs,(void *)&tsmm_inplace_perfargs,sizeof(tsmm_inplace_perfargs),"GF/s");
#endif

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);


    return ret;

}
