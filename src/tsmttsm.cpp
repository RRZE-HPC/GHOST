#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/densemat.h"
#include "ghost/util.h"
#include "ghost/math.h"
#include "ghost/tsmttsm.h"
#include "ghost/tsmttsm_var2_plain_gen.h"
#include "ghost/tsmttsm_var2_avx_gen.h"
#include "ghost/tsmttsm_var2_cu_gen.h"
#include "ghost/tsmttsm_plain_gen.h"
#include "ghost/tsmttsm_var1_plain_gen.h"
#include "ghost/tsmttsm_avx2_gen.h"
#include "ghost/tsmttsm_avx_gen.h"
#include "ghost/tsmttsm_sse_gen.h"
#include "ghost/tsmttsm_cu_gen.h"
#include "ghost/tsmttsm_kahan_var2_plain_gen.h"
#include "ghost/tsmttsm_kahan_plain_gen.h"
#include "ghost/timing.h"
#include "ghost/machine.h"
#include "ghost/constants.h"

#include <map>

typedef ghost_tsmttsm_parameters_t ghost_tsmttsm_kahan_parameters_t;
typedef ghost_tsmttsm_parameters_t ghost_tsmttsm_kahan_parameters_t;

using namespace std;

static bool operator<(const ghost_tsmttsm_parameters_t &a, const ghost_tsmttsm_parameters_t &b) 
{ 
    return ghost_hash(a.dt,a.wcols,ghost_hash(a.vcols,a.impl,ghost_hash(a.xstor,a.wstor,ghost_hash(a.alignment,a.unroll,0)))) < ghost_hash(b.dt,b.wcols,ghost_hash(b.vcols,b.impl,ghost_hash(b.xstor,b.wstor,ghost_hash(b.alignment,b.unroll,0)))); 
}

static map<ghost_tsmttsm_parameters_t, ghost_tsmttsm_kernel_t> ghost_tsmttsm_kernels;
static map<ghost_tsmttsm_parameters_t, ghost_tsmttsm_kernel_t> ghost_tsmttsm_kahan_kernels;


ghost_error_t ghost_tsmttsm_valid(ghost_densemat_t *x, ghost_densemat_t *v, const char * transv, 
ghost_densemat_t *w, const char *transw, void *alpha, void *beta, int reduce, ghost_gemm_flags_t flags, int printerror) 
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


ghost_error_t ghost_tsmttsm(ghost_densemat_t *x, ghost_densemat_t *v, ghost_densemat_t *w, void *alpha, void *beta,int reduce,int conjv,ghost_gemm_flags_t flags)
{
    ghost_error_t ret;

    const char *vtrans;
    if (conjv && v->traits.datatype & GHOST_DT_COMPLEX) {
        vtrans = "C";
    } else {
        vtrans = "T";
    }

    if ((ret = ghost_tsmttsm_valid(x,v,vtrans,w,"N",alpha,beta,reduce,flags,1)) != GHOST_SUCCESS) {
        INFO_LOG("TSMTTSM cannot be applied. Checking whether GEMM is fine!");
        if ((ret = ghost_gemm_valid(x,v,vtrans,w,"N",alpha,beta,reduce,GHOST_GEMM_DEFAULT,1)) != GHOST_SUCCESS) {
            ERROR_LOG("GEMM cannot be applied!");
            return ret;
        } else {
            return ghost_gemm(x,v,vtrans,w,"N",alpha,beta,reduce,GHOST_GEMM_NOT_SPECIAL);
        }
    }
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
   
    map<ghost_tsmttsm_parameters_t, ghost_tsmttsm_kernel_t> kernels;
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
#include "tsmttsm_var1_plain.def"
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


    
    ghost_tsmttsm_parameters_t p;
    ghost_implementation_t opt_impl;
    ghost_alignment_t opt_align;
    int opt_unroll;
    ghost_tsmttsm_kernel_t kernel = NULL;
    
    // fix properties    
    p.xstor = x->traits.storage;
    p.wstor = w->traits.storage;

    // initial implementation
#ifdef GHOST_HAVE_MIC
    opt_impl = GHOST_IMPLEMENTATION_MIC;
#elif defined(GHOST_HAVE_AVX2)
    opt_impl = GHOST_IMPLEMENTATION_AVX2;
#elif defined(GHOST_HAVE_AVX)
    opt_impl = GHOST_IMPLEMENTATION_AVX;
#elif defined(GHOST_HAVE_SSE)
    opt_impl = GHOST_IMPLEMENTATION_SSE;
#else
    opt_impl = GHOST_IMPLEMENTATION_PLAIN;
#endif
    
    
    // alignment of large input data
    // the alignment of the result array does not matter because we can easily re-allocate it accordingly
    int al = ghost_machine_alignment();
    if (IS_ALIGNED(w->val,al) && IS_ALIGNED(v->val,al) && !((w->stride*w->elSize) % al) && !((v->stride*v->elSize) % al)) {
        opt_align = GHOST_ALIGNED;
    } else {
        opt_align = GHOST_UNALIGNED;
    }

#ifdef GHOST_HAVE_CUDA
    if (x->traits.location & GHOST_LOCATION_DEVICE) {
        opt_impl = GHOST_IMPLEMENTATION_CUDA;
        opt_align = GHOST_UNALIGNED;
    }
#endif
    
    ghost_lidx_t try_wcols[2] = {w->traits.ncols,-1};
    ghost_lidx_t try_vcols[2] = {v->traits.ncols,-1};
    ghost_datatype_t try_dt[2] = {v->traits.datatype,GHOST_DT_ANY};

    if (x->traits.flags & GHOST_DENSEMAT_VIEW || v->traits.flags & GHOST_DENSEMAT_VIEW) {
        opt_unroll = 1;
    } else {
        opt_unroll = GHOST_MAX_ROWS_UNROLL;
    }
    
    int n_wcols = sizeof(try_wcols)/sizeof(ghost_lidx_t); 
    int n_vcols = sizeof(try_vcols)/sizeof(ghost_lidx_t); 
    int n_dt = sizeof(try_dt)/sizeof(ghost_datatype_t); 
    int pos_wcols, pos_vcols, pos_dt;
    bool optimal = true; // if we find a kernel with highest specialization grade (regardless unrolling), this remains true and no performance warning gets printed

    for (pos_wcols = 0; pos_wcols < n_wcols; pos_wcols++) {  
        for (pos_vcols = 0; pos_vcols < n_vcols; pos_vcols++) {  
            for (p.impl = opt_impl; (int)p.impl == GHOST_IMPLEMENTATION_CUDA || (int)p.impl >= GHOST_IMPLEMENTATION_PLAIN; p.impl  = (ghost_implementation_t)((int)p.impl-1)) {
                for (p.alignment = opt_align; (int)p.alignment >= GHOST_UNALIGNED; p.alignment = (ghost_alignment_t)((int)p.alignment-1)) {
                    for (p.unroll = opt_unroll; p.unroll > 0; p.unroll /= 2) {
                        for (pos_dt = 0; pos_dt < n_dt; pos_dt++) {
                            p.wcols = try_wcols[pos_wcols];
                            p.vcols = try_vcols[pos_vcols];
                            p.dt = try_dt[pos_dt];
                            INFO_LOG("Try xstor=%s, wstor=%s, wcols=%s, vcols=%s, impl=%s, %s, unroll=%d, dt=%s",
                                    ghost_densemat_storage_string(x),ghost_densemat_storage_string(w),
                                    p.wcols==-1?"arbitrary":to_string(p.wcols).c_str(),p.vcols==-1?"arbitrary":to_string(p.vcols).c_str(),
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
            INFO_LOG("Found kernel with highest specialization grade: dt=%d wcols=%d vcols=%d xstor=%d wstor=%d align=%d unroll=%d impl=%s",p.dt,p.wcols,p.vcols,p.xstor,p.wstor,p.alignment,p.unroll,ghost_implementation_string(p.impl));
        } else {
            PERFWARNING_LOG("Using potentially non-optimal kernel: dt=%d wcols=%d vcols=%d xstor=%d wstor=%d align=%d unroll=%d impl=%s",p.dt,p.wcols,p.vcols,p.xstor,p.wstor,p.alignment,p.unroll,ghost_implementation_string(p.impl));
        }

        ret = kernel(x,v,w,alpha,beta,conjv);
        if (reduce != GHOST_GEMM_NO_REDUCE && v->context) {
            x->reduce(x,v->context->mpicomm,reduce);
        }
    } else {
        PERFWARNING_LOG("Could not find TSMTTSM kernel. Fallback to GEMM");
        ret = ghost_gemm(x,v,conjv?"C":"T",w,"N",alpha,beta,reduce,GHOST_GEMM_NOT_SPECIAL);
    }


#ifdef GHOST_HAVE_INSTR_TIMING
    ghost_gemm_perf_args_t tsmttsm_perfargs;
    tsmttsm_perfargs.n = w->traits.ncols;
    tsmttsm_perfargs.m = v->traits.ncols;
    if (v->context) {
        tsmttsm_perfargs.k = v->context->gnrows;
    } else {
        tsmttsm_perfargs.k = v->traits.nrows;
    }
    tsmttsm_perfargs.dt = x->traits.datatype;
    tsmttsm_perfargs.betaiszero = ghost_iszero(beta,p.dt);
    tsmttsm_perfargs.alphaisone = ghost_isone(alpha,p.dt);
    ghost_timing_set_perfFunc(NULL,__ghost_functag,ghost_gemm_perf_GBs,(void *)&tsmttsm_perfargs,sizeof(tsmttsm_perfargs),"GB/s");
    ghost_timing_set_perfFunc(NULL,__ghost_functag,ghost_gemm_perf_GFs,(void *)&tsmttsm_perfargs,sizeof(tsmttsm_perfargs),"GF/s");
#endif

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
}


