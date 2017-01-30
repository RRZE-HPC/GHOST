#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/densemat.h"
#include "ghost/util.h"
#include "ghost/math.h"
#include "ghost/tsmttsm.h"
#include "ghost/tsmttsm_var2_plain_gen.h"
#include "ghost/tsmttsm_var2_avx_gen.h"
#include "ghost/tsmttsm_plain_gen.h"
#include "ghost/tsmttsm_varcols1_plain_gen.h"
#include "ghost/tsmttsm_varcols2_plain_gen.h"
#include "ghost/tsmttsm_avx2_gen.h"
#include "ghost/tsmttsm_avx_gen.h"
#include "ghost/tsmttsm_sse_gen.h"
#ifdef GHOST_HAVE_CUDA
#include "ghost/tsmttsm_var2_cu_gen.h"
#include "ghost/tsmttsm_cu_gen.h"
#endif
#include "ghost/tsmttsm_kahan_var2_plain_gen.h"
#include "ghost/tsmttsm_kahan_plain_gen.h"
#include "ghost/timing.h"
#include "ghost/machine.h"
#include "ghost/constants.h"
#include "ghost/locality.h"
#include "ghost/cpp11_fixes.h"

#include <unordered_map>
#include <vector>

typedef ghost_tsmttsm_parameters ghost_tsmttsm_kahan_parameters;
typedef ghost_tsmttsm_parameters ghost_tsmttsm_kahan_parameters;

// Hash function for unordered_map
namespace std
{
    template<> struct hash<ghost_tsmttsm_parameters>
    {
        typedef ghost_tsmttsm_parameters argument_type;
        typedef std::size_t result_type;
        result_type operator()(argument_type const& a) const
        {
            return ghost_hash(a.dt,a.wcols,ghost_hash(a.vcols,a.impl,ghost_hash(a.wstor,a.unroll,a.alignment)));
        }
    };
}

static bool operator==(const ghost_tsmttsm_parameters& a, const ghost_tsmttsm_parameters& b)
{
    return a.dt == b.dt && a.wcols == b.wcols && a.vcols == b.vcols && a.impl == b.impl && a.wstor == b.wstor && a.alignment == b.alignment && a.unroll == b.unroll;
}

static std::unordered_map<ghost_tsmttsm_parameters, ghost_tsmttsm_kernel> ghost_tsmttsm_kernels;
static std::unordered_map<ghost_tsmttsm_parameters, ghost_tsmttsm_kernel> ghost_tsmttsm_kahan_kernels;


ghost_error ghost_tsmttsm_valid(ghost_densemat *x, ghost_densemat *v, const char * transv, 
ghost_densemat *w, const char *transw, void *alpha, void *beta, int reduce, ghost_gemm_flags flags, int printerror) 
{
    /*if (w->traits.storage != GHOST_DENSEMAT_ROWMAJOR) {
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
            ERROR_LOG("x must be stored col-major!");
        }
        return GHOST_ERR_INVALID_ARG;
    }
    if (x->traits.location != GHOST_LOCATION_HOST || v->traits.location != GHOST_LOCATION_HOST || w->traits.location != GHOST_LOCATION_HOST) {
        if (printerror) {
            ERROR_LOG("TSMTTSM only implemented for host densemats!");
        }
        return GHOST_ERR_INVALID_ARG;
    }*/

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
    UNUSED(reduce);
    UNUSED(flags);

    return GHOST_SUCCESS;
}


ghost_error ghost_tsmttsm(ghost_densemat *x_in, ghost_densemat *v, ghost_densemat *w, void *alpha, void *beta,int reduce,int conjv,ghost_gemm_flags flags)
{
    ghost_error ret;

    const char *vtrans;
    if (conjv && v->traits.datatype & GHOST_DT_COMPLEX) {
        vtrans = "C";
    } else {
        vtrans = "T";
    }

    if ((ret = ghost_tsmttsm_valid(x_in,v,vtrans,w,"N",alpha,beta,reduce,flags,1)) != GHOST_SUCCESS) {
        INFO_LOG("TSMTTSM cannot be applied. Checking whether GEMM is fine!");
        if ((ret = ghost_gemm_valid(x_in,v,vtrans,w,"N",alpha,beta,reduce,GHOST_GEMM_DEFAULT,1)) != GHOST_SUCCESS) {
            ERROR_LOG("GEMM cannot be applied!");
            return ret;
        } else {
            return ghost_gemm(x_in,v,vtrans,w,"N",alpha,beta,reduce,GHOST_GEMM_NOT_SPECIAL);
        }
    }
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
   
    std::unordered_map<ghost_tsmttsm_parameters, ghost_tsmttsm_kernel> kernels;
    if (flags & GHOST_GEMM_KAHAN) { 
        if (ghost_tsmttsm_kahan_kernels.empty()) {
#include "tsmttsm_kahan_plain.def"
#include "tsmttsm_kahan_var2_plain.def"
        }
        kernels = ghost_tsmttsm_kahan_kernels;
    } else {
        if (ghost_tsmttsm_kernels.empty()) {
#include "tsmttsm_plain.def"
#include "tsmttsm_var2_plain.def"
#include "tsmttsm_varcols1_plain.def"
#include "tsmttsm_varcols2_plain.def"
#include "tsmttsm_var2_avx.def"
#include "tsmttsm_avx2.def"
#include "tsmttsm_avx.def"
#include "tsmttsm_var2_avx.def"
#include "tsmttsm_sse.def"
#ifdef GHOST_HAVE_CUDA
#include "tsmttsm_cu.def"
#include "tsmttsm_var2_cu.def"
#endif
        }
        kernels = ghost_tsmttsm_kernels;
    }

    int me=0;
    ghost_rank(&me,v->map->mpicomm);
    // make sure that the initial x only gets added up once
    if (me && (reduce != GHOST_GEMM_NO_REDUCE)) {
        memset(beta,0,x_in->elSize);
    }


    ghost_densemat *x = NULL;
    ghost_tsmttsm_parameters p;
    ghost_alignment opt_align;
    int opt_unroll;
    ghost_tsmttsm_kernel kernel = NULL;

    if (x_in->traits.storage == GHOST_DENSEMAT_COLMAJOR) {
        x = x_in;
    } else {
        PERFWARNING_LOG("Need to transpose output densemat x!");
        ghost_densemat_traits xtraits = x_in->traits;
        xtraits.flags &= (ghost_densemat_flags)~GHOST_DENSEMAT_VIEW;
        xtraits.storage = GHOST_DENSEMAT_COLMAJOR;
        ghost_densemat_create(&x,ghost_map_create_light(x_in->map->dim,x_in->map->mpicomm),xtraits);
        ghost_densemat_init_densemat(x,x_in,0,0);
    }
    
    // fix properties    
    p.wstor = w->traits.storage;

    // possible implementations
    std::vector<ghost_implementation> try_impl;
#ifdef GHOST_HAVE_CUDA
    if (x->traits.location & GHOST_LOCATION_DEVICE && x->traits.compute_at != GHOST_LOCATION_HOST) {
        try_impl.push_back(GHOST_IMPLEMENTATION_CUDA);
    } else {
#endif
        if (x->traits.compute_with == GHOST_IMPLEMENTATION_DEFAULT) {
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
        } else {
            try_impl.push_back(x->traits.compute_with);
        }
#ifdef GHOST_HAVE_CUDA
    } 
#endif
    
    // alignment of large input data
    // the alignment of the result array does not matter because we can easily re-allocate it accordingly
    int al = ghost_machine_alignment();
    if (IS_ALIGNED(w->val,al) && IS_ALIGNED(v->val,al) && !((w->stride*w->elSize) % al) && !((v->stride*v->elSize) % al)) {
        opt_align = GHOST_ALIGNED;
    } else {
        opt_align = GHOST_UNALIGNED;
    }

    ghost_lidx try_wcols[2] = {w->traits.ncols,-1};
    ghost_lidx try_vcols[2] = {v->traits.ncols,-1};
    ghost_datatype try_dt[2] = {v->traits.datatype,GHOST_DT_ANY};

    if (x->traits.flags & GHOST_DENSEMAT_VIEW || v->traits.flags & GHOST_DENSEMAT_VIEW || x->map->dimpad % 2 || v->map->dimpad % 2) {
        opt_unroll = 1;
    } else {
        opt_unroll = GHOST_MAX_ROWS_UNROLL;
    }
    
#ifdef GHOST_HAVE_CUDA
    if (x->traits.location & GHOST_LOCATION_DEVICE && x->traits.compute_at != GHOST_LOCATION_HOST) {
        try_dt[0] = GHOST_DT_ANY;
        opt_align = GHOST_UNALIGNED;
    }
#endif
    
    int n_wcols = sizeof(try_wcols)/sizeof(ghost_lidx); 
    int n_vcols = sizeof(try_vcols)/sizeof(ghost_lidx); 
    int n_dt = sizeof(try_dt)/sizeof(ghost_datatype); 
    int pos_wcols, pos_vcols, pos_dt;
    bool optimal = true; // if we find a kernel with highest specialization grade (regardless unrolling), this remains true and no performance warning gets printed

    for (pos_wcols = 0; pos_wcols < n_wcols; pos_wcols++) {  
        for (pos_vcols = 0; pos_vcols < n_vcols; pos_vcols++) {  
            for (std::vector<ghost_implementation>::iterator impl = try_impl.begin(); impl != try_impl.end(); impl++) {
                for (p.alignment = opt_align; (int)p.alignment >= GHOST_UNALIGNED; p.alignment = (ghost_alignment)((int)p.alignment-1)) {
                    for (p.unroll = opt_unroll; p.unroll > 0; p.unroll /= 2) {
                        for (pos_dt = 0; pos_dt < n_dt; pos_dt++) {
                            p.wcols = try_wcols[pos_wcols];
                            p.vcols = try_vcols[pos_vcols];
                            p.dt = try_dt[pos_dt];
                            p.impl = *impl;
                            INFO_LOG("Try wstor=%s, wcols=%s, vcols=%s, impl=%s, %s, unroll=%d, dt=%s",
                                    ghost_densemat_storage_string(w->traits.storage), p.wcols==-1?"arbitrary":ghost::to_string((long long)p.wcols).c_str(),p.vcols==-1?"arbitrary":ghost::to_string((long long)p.vcols).c_str(),
                                    ghost_implementation_string(p.impl),p.alignment==GHOST_UNALIGNED?"unaligned":"aligned",p.unroll,ghost_datatype_string(p.dt));
                            kernel = kernels[p];
                            if (kernel) {
                                goto end_of_loop;
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
            INFO_LOG("Found kernel with highest specialization grade: dt=%d wcols=%d vcols=%d wstor=%d align=%d unroll=%d impl=%s",p.dt,p.wcols,p.vcols,p.wstor,p.alignment,p.unroll,ghost_implementation_string(p.impl));
        } else {
            PERFWARNING_LOG("Using potentially non-optimal kernel: dt=%d wcols=%d vcols=%d wstor=%d align=%d unroll=%d impl=%s",p.dt,p.wcols,p.vcols,p.wstor,p.alignment,p.unroll,ghost_implementation_string(p.impl));
        }

        ret = kernel(x,v,w,alpha,beta,conjv);
        if (reduce != GHOST_GEMM_NO_REDUCE) {
            ghost_densemat_reduce(x,reduce);
        }
    } else if (flags & GHOST_GEMM_KAHAN) { 
        WARNING_LOG("Could not find TSMTTSM-Kahan kernel. Trying non-Kahan version!");
        flags &= ~GHOST_GEMM_KAHAN;
        if (x != x_in) {
            ghost_densemat_destroy(x);
        }
        x = x_in;
        ret = ghost_gemm(x_in,v,conjv?"C":"T",w,"N",alpha,beta,reduce,flags);
    } else {
        PERFWARNING_LOG("Could not find TSMTTSM kernel. Fallback to GEMM");
        if (x != x_in) {
            ghost_densemat_destroy(x);
        }
        x = x_in;
        ret = ghost_gemm(x_in,v,conjv?"C":"T",w,"N",alpha,beta,reduce,GHOST_GEMM_NOT_SPECIAL);
    }


#ifdef GHOST_INSTR_TIMING
    ghost_gemm_perf_args tsmttsm_perfargs;
    tsmttsm_perfargs.n = w->traits.ncols;
    tsmttsm_perfargs.m = v->traits.ncols;
    tsmttsm_perfargs.k = v->map->gdim;
    tsmttsm_perfargs.dt = x->traits.datatype;
    tsmttsm_perfargs.betaiszero = ghost_iszero(beta,x->traits.datatype);
    tsmttsm_perfargs.alphaisone = ghost_isone(alpha,x->traits.datatype);
    tsmttsm_perfargs.aisc = false;
    ghost_timing_set_perfFunc(NULL,__ghost_functag,ghost_gemm_perf_GBs,(void *)&tsmttsm_perfargs,sizeof(tsmttsm_perfargs),"GB/s");
    ghost_timing_set_perfFunc(NULL,__ghost_functag,ghost_gemm_perf_GFs,(void *)&tsmttsm_perfargs,sizeof(tsmttsm_perfargs),"GF/s");
#endif
    
    if (x != x_in) {
        ghost_densemat_init_densemat(x_in,x,0,0);
        ghost_densemat_destroy(x);
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
}


