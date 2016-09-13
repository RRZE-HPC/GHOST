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
    .mode = GHOST_KACZ_MODE_NORMAL,
    .best_block_size = 1,
    .normalize = GHOST_KACZ_NORMALIZE_NO,
    .scale = NULL,
    .initialized = false
};

const ghost_carp_opts GHOST_CARP_OPTS_INITIALIZER = {
    .omega = NULL,
    .shift = NULL,
    .num_shifts = 0,
    .mode = GHOST_KACZ_MODE_NORMAL,
    .best_block_size = 1,
    .normalize = GHOST_KACZ_NORMALIZE_NO,
    .scale = NULL,
    .initialized = false
};

template<typename m_t>
static ghost_error ghost_carp_init_tmpl(ghost_sparsemat *mat, ghost_densemat *rhs, ghost_carp_opts *opts)
{
    
    typedef m_t v_t;
    //normalize the system if normalize is yes
    if(opts->normalize == GHOST_KACZ_NORMALIZE_YES) {
        opts->initialized = true;
        ghost_lidx chunkHeight = mat->traits.C;
        ghost_lidx nchunks = mat->nrows/chunkHeight;
        ghost_lidx remchunk = mat->nrows%chunkHeight;
        m_t *scal;
        ghost_malloc((void **)&scal,mat->nrows*sizeof(m_t));
        ghost_lidx row,idx;
        m_t* mval = ((m_t*)mat->sell->val);
        v_t* bval = ((v_t*)rhs->val);
        
        #ifdef GHOST_HAVE_OPENMP 
        #pragma omp parallel for schedule(runtime) private(row,idx) 
        #endif
        for(ghost_lidx chunk=0; chunk < nchunks; ++chunk) {
            #pragma simd
            for(ghost_lidx chunkinrow=0; chunkinrow<mat->traits.C; ++chunkinrow) {
                //TODO convert this into function for outer loop vectorisation
                row = chunk*mat->traits.C + chunkinrow;
                idx = mat->sell->chunkStart[chunk] + chunkinrow;              
                scal[row] = 0;
                for(ghost_lidx j=0; j<mat->sell->chunkLen[chunk]; ++j) {
                    scal[row] += mval[idx]*mval[idx];
                    idx+=chunkHeight;
                }    
                scal[row] = sqrt(scal[row]);
                idx -= mat->sell->chunkLen[chunk]*chunkHeight;               
                for(ghost_lidx j=0; j<mat->sell->chunkLen[chunk]; ++j) {
                    mval[idx] = ((m_t)mval[idx])/scal[row];
                    idx+=chunkHeight;
                }     
                if(bval != NULL)
                    bval[row] = ((v_t)bval[row])/(m_t)scal[row];
            } 
        }
        
        ghost_lidx chunk = nchunks; 
        for(ghost_lidx chunkinrow=0; chunkinrow<remchunk; ++chunkinrow) {
            row = chunk*mat->traits.C + chunkinrow;
            idx = mat->sell->chunkStart[chunk] + chunkinrow;              
            scal[row] = 0;
            for(ghost_lidx j=0; j<mat->sell->chunkLen[chunk]; ++j) {
                scal[row] += mval[idx]*mval[idx];
                idx+=chunkHeight;
            }    
            scal[row] = sqrt(scal[row]);
            idx -= mat->sell->chunkLen[chunk]*chunkHeight;               
            for(ghost_lidx j=0; j<mat->sell->chunkLen[chunk]; ++j) {
                mval[idx] = ((m_t)mval[idx])/scal[row];
                idx+=chunkHeight;
            }
            if(bval != NULL)
                bval[row] = ((v_t)bval[row])/(m_t)scal[row];
        } 
        
        opts->scale = scal;     
    } 
    
    return GHOST_SUCCESS;
}

ghost_error ghost_carp_init(ghost_sparsemat *mat, ghost_densemat *rhs, ghost_carp_opts *opts)
{
    ghost_error ret = GHOST_SUCCESS;
    SELECT_TMPL_1DATATYPE(mat->traits.datatype,std::complex,ret,ghost_carp_init_tmpl,mat,rhs,opts);
    return ret;
}

//This finds optimum parameters for the CARP, depending on the matrix
//options like shift should be set before entering this
template<typename m_t>
ghost_error ghost_carp_perf_init_tmpl(ghost_sparsemat *mat, ghost_carp_opts *opts) {
    ghost_error ret = GHOST_SUCCESS; 
    if(opts->mode == GHOST_KACZ_MODE_PERFORMANCE) {
        //check for optimal block value
        ghost_densemat *test_rhs, *test_x;
        ghost_densemat_traits vtraits_col = GHOST_DENSEMAT_TRAITS_INITIALIZER;
        ghost_densemat_traits vtraits_row = GHOST_DENSEMAT_TRAITS_INITIALIZER;
        vtraits_col.storage = GHOST_DENSEMAT_ROWMAJOR;
        vtraits_col.permutemethod = COLUMN;
        vtraits_row.storage = GHOST_DENSEMAT_ROWMAJOR;
        vtraits_row.permutemethod = ROW;
        vtraits_row.datatype = (ghost_datatype)mat->traits.datatype;
        double start,end;
        double max_flop = 0;
        //as a safety factor complex number is taken as x
        if(opts->shift != NULL) {
            for(int i=0; i<opts->num_shifts; ++i) {
                //no mixing of double and float, also in the kernels, so this is valid
                if(mat->traits.datatype & GHOST_DT_DOUBLE) {
                    vtraits_col.datatype = (ghost_datatype)(GHOST_DT_COMPLEX|GHOST_DT_DOUBLE);
                } else {
                    vtraits_col.datatype = (ghost_datatype)(GHOST_DT_COMPLEX|GHOST_DT_FLOAT);
                }
            }
        }  
        
        std::complex<double> zero=0;
        std::complex<double> one=1;
        
        int total_sizes = 6;
        int *block_size;
        ghost_malloc((void **)&block_size, total_sizes*sizeof(int));
        //TODO block values should be precompiled, now 1,4,8,16,32,64,128,256
        if(total_sizes > 1) {
            block_size[0] = 1;
        }
        if(total_sizes > 2) {
            block_size[1] = 4;
        }
        if(total_sizes>3) {
            for(int i=2; i<total_sizes; ++i) {
                block_size[i] = block_size[i-1]*2; 
            }
        }
        int nIter = 1;//things like alpha=0 wouldn't be considered, but the LC 
        for (int i=0; i<total_sizes; ++i) { 
            vtraits_col.ncols = block_size[i];
            vtraits_row.ncols = block_size[i];
            ghost_densemat_create(&test_x, mat->context, vtraits_col);
            ghost_densemat_create(&test_rhs, mat->context, vtraits_row);
            test_x->fromScalar(test_x,&zero);
            test_rhs->fromScalar(test_rhs,&one);
            ghost_barrier();
            ghost_timing_wcmilli(&start);
            
            for(int iter=0; iter<nIter; ++iter) {
                ghost_carp(mat, test_x, test_rhs, *opts);
            }
            
            ghost_barrier();
            ghost_timing_wcmilli(&end);
            double flop = ((nIter*mat->context->gnnz)*vtraits_col.ncols*1*8*1e-6)/(end-start);
            if(flop > max_flop) {
                max_flop = std::max(max_flop,flop);
            } else {
                //Some LC broken
                opts->best_block_size = block_size[i-1];
                free(block_size);
                return ret;
            }
        }
        opts->best_block_size = block_size[total_sizes-1];
        free(block_size);
    }
    return ret;
}

ghost_error ghost_carp_perf_init(ghost_sparsemat *mat, ghost_carp_opts *opts) {
    ghost_error ret = GHOST_SUCCESS;
    SELECT_TMPL_1DATATYPE(mat->traits.datatype,std::complex,ret,ghost_carp_perf_init_tmpl,mat,opts);
    return ret;
}


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
    
    ghost_lidx rank;
    ghost_rank(&rank,mat->context->mpicomm);
    
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    
    if (!mat->color_ptr || mat->ncolors == 0) {
        WARNING_LOG("Matrix has not been colored!");
    }
    if (x->traits.ncols > 1) {
        ERROR_LOG("Multi-vec not implemented!");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
    
    if(opts.normalize == GHOST_KACZ_NORMALIZE_YES && opts.initialized == false) {
        WARNING_LOG("Kacz on normalized system is called, and the system has not been normalized at the start"); 
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

ghost_error ghost_kacz(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *rhs, ghost_kacz_opts opts)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error ret = GHOST_SUCCESS;
    ghost_kacz_parameters p;
    
    //if rectangular matrix
    if(x->traits.permutemethod != COLUMN && mat->nrows != mat->context->nrowspadded) {
        ERROR_LOG("Output vector is not COLUMN(Right sided) vector, please set permutemethod to column in traits field")
    }
    
    ghost_densemat *b;
    //deal with NULL pointer of b
    if(rhs==NULL) {
        if(opts.num_shifts != 0)       
            ghost_densemat_create_and_view_densemat(&b, x, x->traits.nrows, 0, x->traits.ncols/opts.num_shifts, 0);
        else       
            ghost_densemat_create_and_view_densemat(&b, x, x->traits.nrows, 0, x->traits.ncols, 0);
        
        b->val = NULL; 
    } else {
        b = rhs;
    }
    
    if(opts.shift) {     
        if(!(mat->traits.flags & GHOST_SPARSEMAT_COLOR)) {
            if(!(mat->traits.flags & GHOST_SPARSEMAT_BLOCKCOLOR) && (mat->kaczRatio >= 2*mat->kacz_setting.active_threads)) {
                INFO_LOG("BMC KACZ_shift without transition called");
                p.method = GHOST_KACZ_METHOD_BMCshift;//Now BMC_RB can run with BMC 
            }
            else {
                INFO_LOG("BMC KACZ_shift with transition called");
                p.method = GHOST_KACZ_METHOD_BMCshift;
            }
        } else {
            ERROR_LOG("Shift ignored MC with shift not implemented");
            p.method = GHOST_KACZ_METHOD_MC;
        }
    } else if(opts.normalize==GHOST_KACZ_NORMALIZE_YES) {
        if(!(mat->traits.flags & GHOST_SPARSEMAT_COLOR)) {
            if(!(mat->traits.flags & GHOST_SPARSEMAT_BLOCKCOLOR) && (mat->kaczRatio >= 2*mat->kacz_setting.active_threads)) {
                INFO_LOG("BMC KACZ without transition, for Normalized system called");
                p.method = GHOST_KACZ_METHOD_BMCNORMAL;//Now BMC_RB can run with BMC 
            }
            else {
                INFO_LOG("BMC KACZ with transition, for Normalized system called");
                p.method = GHOST_KACZ_METHOD_BMCNORMAL;
            }
        } else {
            INFO_LOG("Using unoptimal kernel KACZ with MC");
            p.method = GHOST_KACZ_METHOD_MC;
        }
    } else {
        if(!(mat->traits.flags & GHOST_SPARSEMAT_COLOR)) {
            if(!(mat->traits.flags & GHOST_SPARSEMAT_BLOCKCOLOR) && (mat->kaczRatio >= 2*mat->kacz_setting.active_threads)) {
                INFO_LOG("BMC KACZ without transition called");
                p.method = GHOST_KACZ_METHOD_BMC;//Now BMC_RB can run with BMC 
            }
            else {
                INFO_LOG("BMC KACZ with transition called");
                p.method = GHOST_KACZ_METHOD_BMC;
            }
        } else {
            INFO_LOG("Using unoptimal kernel KACZ with MC");
            p.method = GHOST_KACZ_METHOD_MC;
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
            if (x->traits.storage == GHOST_DENSEMAT_COLMAJOR && b->traits.storage == x->traits.storage) {
                first_storage = 0;
            } else {
                first_storage = 1;
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
            int try_blocksz[2] = {b->traits.ncols,-1}; 
            
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
                                        
                                        
                                        INFO_LOG("Try chunkheight=%s, blocksz=%s, impl=%s, %s, method %s, storage %s, vec DT %s",
                                                 p.chunkheight==-1?"arbitrary":std::to_string((long long)p.chunkheight).c_str(),
                                                 p.blocksz==-1?"arbitrary":std::to_string((long long)p.blocksz).c_str(),
                                                 ghost_implementation_string(p.impl),p.alignment==GHOST_UNALIGNED?"unaligned":"aligned",p.method==GHOST_KACZ_METHOD_BMC?"BMC":p.method==GHOST_KACZ_METHOD_MC?"MC":p.method==GHOST_KACZ_METHOD_BMCNORMAL?"BMC_NORMAL":"BMC_shift",ghost_densemat_storage_string(p.storage),ghost_datatype_string(p.vdt));
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
