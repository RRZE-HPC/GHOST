#include "ghost/types.h"
#include "ghost/complex.h"
#include "ghost/locality.h"
#include "ghost/util.h"
#include "ghost/timing.h"
#include "ghost/machine.h"
#include "ghost/sparsemat.h"
#include "ghost/math.h"
#include "ghost/sell_kacz_mc_gen.h"
//#include "ghost/sell_kacz_avx_gen.h"
#include "ghost/sell_kacz_bmc_cm_gen.h"
#include "ghost/sell_kacz_bmc_rm_gen.h"
#include "ghost/sell_kacz_bmc_normal_gen.h"
#include "ghost/sell_kacz_bmc_shift_gen.h"
#include <complex>
#include <unordered_map>

using namespace std;

const ghost_kacz_opts GHOST_KACZ_OPTS_INITIALIZER = {
    .omega = NULL,
    .shift = NULL,
    .num_shifts = 0,
    .direction = GHOST_KACZ_DIRECTION_UNDEFINED,
    .normalize = no
};

const ghost_carp_opts GHOST_CARP_OPTS_INITIALIZER = {
    .omega = NULL,
    .shift = NULL,
    .num_shifts = 0,
    .normalize = no
};


// Hash function for unordered_map
namespace std
{
    template<> struct hash<ghost_kacz_parameters>
    {
        typedef ghost_kacz_parameters argument_type;
        typedef std::size_t result_type;
        result_type operator()(argument_type const& a) const
        {
            return ghost_hash(ghost_hash(a.mdt,a.blocksz,a.storage),
                    ghost_hash(a.vdt,a.impl,a.chunkheight),ghost_hash(a.alignment,a.method,0));
        }
    };
}

static bool operator==(const ghost_kacz_parameters& a, const ghost_kacz_parameters& b)
{
    return a.mdt == b.mdt && a.blocksz == b.blocksz && a.storage == b.storage && 
           a.vdt == b.vdt && a.impl == b.impl && a.chunkheight == b.chunkheight &&
           a.alignment == b.alignment && a.method == b.method;
}

static unordered_map<ghost_kacz_parameters, ghost_kacz_kernel> 
ghost_kacz_kernels = unordered_map<ghost_kacz_parameters,ghost_kacz_kernel>();


template<typename m_t, typename v_t, bool forward>
static ghost_error kacz_fallback(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    
    if (!mat->color_ptr || mat->ncolors == 0) {
        WARNING_LOG("Matrix has not been colored!");
    }
    if (x->traits.ncols > 1) {
        ERROR_LOG("Multi-vec not implemented!");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
   
    ghost_lidx c;
    ghost_lidx row;
    ghost_lidx rowinchunk;
    ghost_lidx j;
    ghost_lidx color;
    ghost_sell *sellmat = SELL(mat);
    ghost_lidx fchunk, lchunk;
    v_t *bval = (v_t *)(b->val);
    v_t *xval = (v_t *)(x->val);
    m_t *mval = (m_t *)sellmat->val;
    v_t omega = *(v_t *)opts.omega;


    int rank;
    ghost_rank(&rank,mat->context->mpicomm);

    ghost_lidx firstcolor, lastcolor, stride;
    
    if (forward) {
        firstcolor = 0;
        lastcolor = mat->ncolors;
        stride = 1;
    } else {
        firstcolor = mat->ncolors-1;
        lastcolor = -1;
        stride = -1;
    }

    
    for (color=firstcolor; color!=lastcolor; color+=stride) {
        fchunk = mat->color_ptr[color]/mat->traits.C;
        lchunk = mat->color_ptr[color+1]/mat->traits.C;
#pragma omp parallel
        { 
            m_t *rownorm;
            ghost_malloc((void **)&rownorm,mat->traits.C*sizeof(m_t));
#pragma omp for private(j,row,rowinchunk)
            for (c=fchunk; c<lchunk; c++) {
                for (rowinchunk = 0; rowinchunk < mat->traits.C; rowinchunk++) {
                    row = rowinchunk + c*mat->traits.C;
                    rownorm[rowinchunk] = 0.;

                    ghost_lidx idx = sellmat->chunkStart[c]+rowinchunk;
                    v_t scal = -bval[row];

                    for (j=0; j<sellmat->rowLen[row]; j++) {
                        scal += (v_t)mval[idx] * xval[sellmat->col[idx]];
                        rownorm[rowinchunk] += mval[idx]*mval[idx];
                        idx += mat->traits.C;
                    }

                    idx -= mat->traits.C*sellmat->rowLen[row];
                    scal /= (v_t)rownorm[rowinchunk];

                    for (j=0; j<sellmat->rowLen[row]; j++) {
                        xval[sellmat->col[idx]] = xval[sellmat->col[idx]] - omega * scal * (v_t)mval[idx];
                        idx += mat->traits.C;
                    }
                }
            }
            free(rownorm);
            rownorm = NULL;
        }
    }
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    return GHOST_SUCCESS;
}

ghost_error ghost_kacz(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error ret = GHOST_SUCCESS;
    ghost_kacz_parameters p;

    if(opts.shift) {     
    	if(!(mat->traits.flags & GHOST_SPARSEMAT_COLOR)) {
    		if(!(mat->traits.flags & GHOST_SPARSEMAT_BLOCKCOLOR) && (mat->kaczRatio >= 2*mat->kacz_setting.active_threads)) {
			INFO_LOG("BMC KACZ_shift without transition called");
			p.method = BMCshift;//Now BMC_RB can run with BMC 
    		}
    		else {
			INFO_LOG("BMC KACZ_shift with transition called");
			p.method = BMCshift;
    		}
    	} else {
        	ERROR_LOG("Shift ignored MC with shift not implemented");
		p.method = MC;
    	}
    } else if(opts.normalize==yes) {
    	if(!(mat->traits.flags & GHOST_SPARSEMAT_COLOR)) {
    		if(!(mat->traits.flags & GHOST_SPARSEMAT_BLOCKCOLOR) && (mat->kaczRatio >= 2*mat->kacz_setting.active_threads)) {
			INFO_LOG("BMC KACZ without transition, for Normalized system called");
			p.method = BMCNORMAL;//Now BMC_RB can run with BMC 
    		}
    		else {
			INFO_LOG("BMC KACZ with transition, for Normalized system called");
			p.method = BMCNORMAL;
    		}
    	} else {
        	INFO_LOG("Using unoptimal kernel KACZ with MC");
		p.method = MC;
    	}
    } else {
    	if(!(mat->traits.flags & GHOST_SPARSEMAT_COLOR)) {
    		if(!(mat->traits.flags & GHOST_SPARSEMAT_BLOCKCOLOR) && (mat->kaczRatio >= 2*mat->kacz_setting.active_threads)) {
			INFO_LOG("BMC KACZ without transition called");
			p.method = BMC;//Now BMC_RB can run with BMC 
    		}
    		else {
			INFO_LOG("BMC KACZ with transition called");
			p.method = BMC;
    		}
    	} else {
        	INFO_LOG("Using unoptimal kernel KACZ with MC");
		p.method = MC;
    	}
    }
	 
 
    // if map is empty include generated code for map construction
    if (ghost_kacz_kernels.empty()) {
#include "sell_kacz_bmc_cm.def"
#include "sell_kacz_bmc_rm.def"
#include "sell_kacz_bmc_normal.def"
#include "sell_kacz_bmc_shift.def"
#include "sell_kacz_mc.def"
//#include "sell_kacz_avx.def"
      
    }
    
    ghost_kacz_kernel kernel = NULL;
    ghost_implementation opt_impl;
    ghost_alignment opt_align;
    
    ghost_densemat_storage try_storage[2] = {GHOST_DENSEMAT_COLMAJOR,GHOST_DENSEMAT_ROWMAJOR};
    int n_storage, pos_storage, first_storage, n_mdt, n_vdt;
    
    ghost_datatype try_vdt[2] = {x->traits.datatype,GHOST_DT_ANY};
    ghost_datatype try_mdt[2] = {mat->traits.datatype,GHOST_DT_ANY};
    n_vdt = sizeof(try_vdt)/sizeof(ghost_datatype); 
    n_mdt = sizeof(try_mdt)/sizeof(ghost_datatype); 
   
    if (x->traits.ncols == 1 && b->traits.ncols == 1 && 
            (x->traits.storage == GHOST_DENSEMAT_COLMAJOR || x->stride == 1) && 
            (b->traits.storage == GHOST_DENSEMAT_COLMAJOR || b->stride == 1)) {
        INFO_LOG("Try both col- and row-major for 1-column densemat with stride 1");
        n_storage = 2;
        first_storage = 0;
    } else {
        n_storage = 1;
        if (x->traits.storage == GHOST_DENSEMAT_ROWMAJOR && b->traits.storage == x->traits.storage) {
            first_storage = 1;
        } else {
            first_storage = 0;
        }
    }
    if ((b->traits.flags & GHOST_DENSEMAT_SCATTERED) || 
            (x->traits.flags & GHOST_DENSEMAT_SCATTERED)) {
        PERFWARNING_LOG("Use plain implementation for scattered views");
        opt_impl = GHOST_IMPLEMENTATION_PLAIN;
    } else {
        if (x->stride > 1 && x->traits.storage == GHOST_DENSEMAT_ROWMAJOR) {
            opt_impl = ghost_get_best_implementation_for_bytesize(x->traits.ncols*x->elSize);
            if (opt_impl == GHOST_IMPLEMENTATION_PLAIN) {
                // this branch is taken for odd numbers
                // choose a version with remainder loops in this case!
                opt_impl = ghost_get_best_implementation_for_bytesize(PAD(x->traits.ncols*x->elSize,ghost_machine_simd_width()));
            }
        } else {
            opt_impl = ghost_get_best_implementation_for_bytesize(mat->traits.C*mat->elSize);
        }
    }
    
    int try_chunkheight[2] = {mat->traits.C,-1}; 
    int try_blocksz[2] = {x->traits.ncols,-1}; 

    int n_chunkheight = sizeof(try_chunkheight)/sizeof(int);
    int n_blocksz = sizeof(try_blocksz)/sizeof(int);
    int pos_chunkheight, pos_blocksz, pos_mdt, pos_vdt;

    bool optimal = true;
    
   for (pos_chunkheight = 0; pos_chunkheight < n_chunkheight; pos_chunkheight++) {  
        for (pos_blocksz = 0; pos_blocksz < n_blocksz; pos_blocksz++) {  
            for (p.impl = opt_impl; (int)p.impl >= GHOST_IMPLEMENTATION_PLAIN; p.impl  = (ghost_implementation)((int)p.impl-1)) {

                int al = ghost_implementation_alignment(p.impl);
                if (IS_ALIGNED(b->val,al) && IS_ALIGNED(x->val,al) && ((b->traits.ncols == 1 && b->stride == 1) || (!((b->stride*b->elSize) % al) && !((x->stride*x->elSize) % al)))) {
                    opt_align = GHOST_ALIGNED;
                } else {
                    if (!IS_ALIGNED(b->val,al)) {
                        PERFWARNING_LOG("Using unaligned kernel because base address of result vector is not aligned");
                    }
                    if (!IS_ALIGNED(x->val,al)) {
                        PERFWARNING_LOG("Using unaligned kernel because base address of input vector is not aligned");
                    }
                    if (b->stride*b->elSize % al) {
                        PERFWARNING_LOG("Using unaligned kernel because stride of result vector does not yield aligned addresses");
                    }
                    if (x->stride*b->elSize % al) {
                        PERFWARNING_LOG("Using unaligned kernel because stride of input vector does not yield aligned addresses");
                    }
                    opt_align = GHOST_UNALIGNED;
                }

                for (p.alignment = opt_align; (int)p.alignment >= GHOST_UNALIGNED; p.alignment = (ghost_alignment)((int)p.alignment-1)) {
                    for (pos_mdt = 0; pos_mdt < n_mdt; pos_mdt++) {
                        for (pos_vdt = 0; pos_vdt < n_vdt; pos_vdt++) {
                            for (pos_storage = first_storage; pos_storage < n_storage+first_storage; pos_storage++) {  
                                p.chunkheight = try_chunkheight[pos_chunkheight];
                                p.blocksz = try_blocksz[pos_blocksz];
                                p.storage = try_storage[pos_storage];
                                p.mdt = try_mdt[pos_mdt];
                                p.vdt = try_vdt[pos_vdt];


                            	INFO_LOG("Try chunkheight=%s, blocksz=%s, impl=%s, %s, method %s",
                                    p.chunkheight==-1?"arbitrary":std::to_string((long long)p.chunkheight).c_str(),
                                    p.blocksz==-1?"arbitrary":std::to_string((long long)p.blocksz).c_str(),
                                    ghost_implementation_string(p.impl),p.alignment==GHOST_UNALIGNED?"unaligned":"aligned",p.method==BMC?"BMC":p.method==MC?"MC":p.method==BMCNORMAL?"BMC_NORMAL":"BMC_shift");
                            	kernel = ghost_kacz_kernels[p];
                            	if (kernel) {
                            		goto end_of_loop;
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
            INFO_LOG("Found kernel with highest specialization grade: C=%d blocksz=%d align=%d impl=%s",p.chunkheight,p.blocksz,p.alignment,ghost_implementation_string(p.impl));
        } else {
            PERFWARNING_LOG("Using potentially non-optimal kernel: C=%d blocksz=%d align=%d impl=%s",p.chunkheight,p.blocksz,p.alignment,ghost_implementation_string(p.impl));
        }
        ret = kernel(x,mat,b,opts);
    } else { // execute plain kernel as fallback
        PERFWARNING_LOG("Execute fallback Kaczmarz kernel which is potentially slow!");
    
        if (b->traits.datatype & GHOST_DT_COMPLEX) {
            if (b->traits.datatype & GHOST_DT_DOUBLE) {
                if (opts.direction == GHOST_KACZ_DIRECTION_FORWARD) {
                    ret = kacz_fallback<std::complex<double>, std::complex<double>, true>(x,mat,b,opts);
                } else {
                    ret = kacz_fallback<std::complex<double>, std::complex<double>, false>(x,mat,b,opts);
                }
            } else {
                if (opts.direction == GHOST_KACZ_DIRECTION_FORWARD) {
                    ret = kacz_fallback<std::complex<float>, std::complex<float>, true>(x,mat,b,opts);
                } else {
                    ret = kacz_fallback<std::complex<float>, std::complex<float>, false>(x,mat,b,opts);
                }
            }
        } else {
            if (b->traits.datatype & GHOST_DT_DOUBLE) {
                if (opts.direction == GHOST_KACZ_DIRECTION_FORWARD) {
                    ret = kacz_fallback<double, double, true>(x,mat,b,opts);
                } else {
                    ret = kacz_fallback<double, double, false>(x,mat,b,opts);
                }
            } else {
                if (opts.direction == GHOST_KACZ_DIRECTION_FORWARD) {
                    ret = kacz_fallback<float, float, true>(x,mat,b,opts);
                } else {
                    ret = kacz_fallback<float, float, false>(x,mat,b,opts);
                }
            }
        }
    }
   
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);
    return ret;
}
