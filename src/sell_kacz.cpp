#include "ghost/types.h"
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
#include "ghost/sell_kacz_fallback.h"
#include "ghost/compatibility_check.h"
#include "ghost/autogen.h"
#include "ghost/cpp11_fixes.h"
#include <complex>
#include <unordered_map>
#include <vector>

    template<typename m_t>
static ghost_error ghost_carp_init_tmpl(ghost_sparsemat *mat, ghost_densemat *rhs, ghost_carp_opts *opts)
{
    opts->initialized = true;
 
    typedef m_t v_t;
    //normalize the system if normalize is yes
    if(opts->normalize & (GHOST_KACZ_NORMALIZE_YES | GHOST_KACZ_NORMALIZE_SAVE)) {
        ghost_lidx chunkHeight = mat->traits.C;
        ghost_lidx nchunks = SPM_NROWS(mat)/chunkHeight;
        ghost_lidx remchunk = SPM_NROWS(mat)%chunkHeight;
        m_t scal_var = 0;
        m_t *scal = NULL;
        if(opts->normalize & GHOST_KACZ_NORMALIZE_SAVE) {
            ghost_malloc((void **)&scal,SPM_NROWS(mat)*sizeof(m_t));
        }
        ghost_lidx row,idx;
        m_t* mval = ((m_t*)mat->val);
        v_t* bval = ((v_t*)rhs->val);

#ifdef GHOST_HAVE_OPENMP 
#pragma omp parallel for schedule(runtime) private(row,idx) 
#endif
        for(ghost_lidx chunk=0; chunk < nchunks; ++chunk) {
#pragma simd
            for(ghost_lidx chunkinrow=0; chunkinrow<mat->traits.C; ++chunkinrow) {
                //TODO convert this into function for outer loop vectorisation
                row = chunk*mat->traits.C + chunkinrow;
                idx = mat->chunkStart[chunk] + chunkinrow;              
                scal_var = 0;
                for(ghost_lidx j=0; j<mat->chunkLen[chunk]; ++j) {
                    scal_var += ghost::norm( mval[idx]);
                    idx+=chunkHeight;
                }    
                scal_var = std::sqrt(scal_var);

                if(opts->normalize & GHOST_KACZ_NORMALIZE_YES) {
                    idx -= mat->chunkLen[chunk]*chunkHeight;               
                    for(ghost_lidx j=0; j<mat->chunkLen[chunk]; ++j) {
                        mval[idx] = ((m_t)mval[idx])/scal_var;
                        idx+=chunkHeight;
                    }     
                    if(bval != NULL) {
                        bval[row] = ((v_t)bval[row])/(m_t)scal_var;
                    }
                }
                if(opts->normalize & GHOST_KACZ_NORMALIZE_SAVE) { 
                    scal[row] = scal_var;
                } 
            } 
        }

        ghost_lidx chunk = nchunks; 
        for(ghost_lidx chunkinrow=0; chunkinrow<remchunk; ++chunkinrow) {
            row = chunk*mat->traits.C + chunkinrow;
            idx = mat->chunkStart[chunk] + chunkinrow;              
            scal_var = 0;
            for(ghost_lidx j=0; j<mat->chunkLen[chunk]; ++j) {
                scal_var += ghost::norm( mval[idx]);
                idx+=chunkHeight;
            }    
            scal_var = std::sqrt(scal_var);

            if(opts->normalize & GHOST_KACZ_NORMALIZE_YES) {
                idx -= mat->chunkLen[chunk]*chunkHeight;               
                for(ghost_lidx j=0; j<mat->chunkLen[chunk]; ++j) {
                    mval[idx] = ((m_t)mval[idx])/scal_var;
                    idx+=chunkHeight;
                }     
                if(bval != NULL) {
                    bval[row] = ((v_t)bval[row])/(m_t)scal_var;
                }
            }
            if(opts->normalize & GHOST_KACZ_NORMALIZE_SAVE) { 
                scal[row] = scal_var;
            } 
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

void ghost_carp_destroy(ghost_carp_opts *opts)
{
    if(opts) {
        if(opts->scale){
            free(opts->scale);
            opts->scale = NULL;
        }
    }
}

//This finds optimum parameters for the CARP, depending on the matrix
//options like shift should be set before entering this
template<typename m_t>
static ghost_error ghost_carp_perf_init_tmpl(ghost_sparsemat *mat, ghost_carp_opts *opts) {
    ghost_error ret = GHOST_SUCCESS; 
    if(opts->mode == GHOST_KACZ_MODE_PERFORMANCE) {
        //check for optimal block value
        ghost_densemat *test_rhs, *test_x;
        ghost_densemat_traits vtraits_col = GHOST_DENSEMAT_TRAITS_INITIALIZER;
        ghost_densemat_traits vtraits_row = GHOST_DENSEMAT_TRAITS_INITIALIZER;
        vtraits_col.storage = GHOST_DENSEMAT_ROWMAJOR;
        vtraits_row.storage = GHOST_DENSEMAT_ROWMAJOR;
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
        std::vector<int> block_size;
        int total_sizes = 0;
        //TODO block values should be precompiled, now 1,4,8,16,32,64,128,256
        int curr_block_size = 64; // since most of the matrix violate LC after 64

        int *block_sizes;
        int nblock_sizes;
        ghost_autogen_kacz_nvecs(&block_sizes,&nblock_sizes,mat->traits.C,opts->num_shifts);

        int i;

        while(curr_block_size > 0)
        {
            for (i=nblock_sizes-1; i>=0; i--) {
                if (block_sizes[i] <= curr_block_size) {
                    curr_block_size = block_sizes[i];
                    break;
                }
            }

            if(i == -1) {
                GHOST_ERROR_LOG("Please compile block size 1")
                    break;
            }
            block_size.push_back(curr_block_size);
            curr_block_size -= 1;
            total_sizes += 1;
        }
        free(block_sizes);

        if(block_size.back() !=1) {
            GHOST_WARNING_LOG("Please compile block size = 1")
        }

        int nIter = 1;//things like alpha=0 wouldn't be considered, but the LC 
        for (i=0; i<total_sizes; ++i) { 
            vtraits_col.ncols = block_size[i];
            vtraits_row.ncols = block_size[i];
            ghost_densemat_create(&test_x, mat->context->col_map, vtraits_col);
            ghost_densemat_create(&test_rhs, mat->context->row_map, vtraits_row);
            ghost_densemat_init_val(test_x,&zero);
            ghost_densemat_init_val(test_rhs,&one);
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
                return ret;
            }
        }
        opts->best_block_size = block_size[total_sizes-1];
        //free(block_size);
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
                    ghost_hash(a.vdt,a.impl,a.chunkheight),ghost_hash(a.alignment,a.method,a.nshifts));
        }
    };
}

static bool operator==(const ghost_kacz_parameters& a, const ghost_kacz_parameters& b)
{
    return a.mdt == b.mdt && a.blocksz == b.blocksz && a.storage == b.storage && 
        a.vdt == b.vdt && a.impl == b.impl && a.chunkheight == b.chunkheight &&
        a.alignment == b.alignment && a.method == b.method && a.nshifts == b.nshifts;
}

static std::unordered_map<ghost_kacz_parameters, ghost_kacz_kernel> 
ghost_kacz_kernels = std::unordered_map<ghost_kacz_parameters,ghost_kacz_kernel>();


ghost_error ghost_kacz(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *rhs, ghost_kacz_opts opts)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);

    if(!(opts.initialized)) {
      GHOST_WARNING_LOG("You have not initialized CARP");
    }

    ghost_error ret = GHOST_SUCCESS;

    //////////////// check compatibility /////////////
    ghost_compatible_mat_vec check = GHOST_COMPATIBLE_MAT_VEC_INITIALIZER;
    check.mat = mat;
    check.right1 = x;
    check.left1 = rhs;

    ret = ghost_check_mat_vec_compatibility(&check,mat->context);
    ///////////////////////////////////////////////////

    ghost_kacz_parameters p;

    //if rectangular matrix
    if(x->map->type != GHOST_MAP_COL && SPM_NROWS(mat) != mat->context->col_map->dim) {
        GHOST_ERROR_LOG("Output vector is not COLUMN(Right sided) vector, please set the map to the column map")
    }

    ghost_densemat *b;
    //deal with NULL pointer of b
    if(rhs==NULL) {
        if(opts.num_shifts != 0)       
            ghost_densemat_create_and_view_densemat(&b, x, DM_NROWS(x), 0, x->traits.ncols/opts.num_shifts, 0);
        else       
            ghost_densemat_create_and_view_densemat(&b, x, DM_NROWS(x), 0, x->traits.ncols, 0);

        b->val = NULL; 
    } else {
        b = rhs;
    }

    if(opts.shift) {     
        if(!(mat->traits.flags & GHOST_SPARSEMAT_COLOR)) {
            p.nshifts = opts.num_shifts;
            if(!(mat->traits.flags & GHOST_SPARSEMAT_BLOCKCOLOR) && (mat->context->kaczRatio >= 2*mat->context->kacz_setting.active_threads)) {
                GHOST_INFO_LOG("BMC KACZ_shift without transition called");
                p.method = GHOST_KACZ_METHOD_BMCshift;//Now BMC_RB can run with BMC 
            }
            else {
                GHOST_INFO_LOG("BMC KACZ_shift with transition called");
                p.method = GHOST_KACZ_METHOD_BMCshift;
            }
        } else {
            GHOST_ERROR_LOG("Shift ignored MC with shift not implemented");
            p.method = GHOST_KACZ_METHOD_MC;
        }
    } else if(opts.normalize==GHOST_KACZ_NORMALIZE_YES) {
        if(!(mat->traits.flags & GHOST_SPARSEMAT_COLOR)) {
            if(!(mat->traits.flags & GHOST_SPARSEMAT_BLOCKCOLOR) && (mat->context->kaczRatio >= 2*mat->context->kacz_setting.active_threads)) {
                GHOST_INFO_LOG("BMC KACZ without transition, for Normalized system called");
                p.method = GHOST_KACZ_METHOD_BMCNORMAL;//Now BMC_RB can run with BMC 
            }
            else {
                GHOST_INFO_LOG("BMC KACZ with transition, for Normalized system called");
                p.method = GHOST_KACZ_METHOD_BMCNORMAL;
            }
        } else {
            GHOST_INFO_LOG("Using unoptimal kernel KACZ with MC");
            p.method = GHOST_KACZ_METHOD_MC;
        }
    } else {
        if(!(mat->traits.flags & GHOST_SPARSEMAT_COLOR)) {
            if(!(mat->traits.flags & GHOST_SPARSEMAT_BLOCKCOLOR) && (mat->context->kaczRatio >= 2*mat->context->kacz_setting.active_threads)) {
                GHOST_INFO_LOG("BMC KACZ without transition called");
                p.method = GHOST_KACZ_METHOD_BMC;//Now BMC_RB can run with BMC 
            }
            else {
                GHOST_INFO_LOG("BMC KACZ with transition called");
                p.method = GHOST_KACZ_METHOD_BMC;
            }
        } else {
            GHOST_INFO_LOG("Using unoptimal kernel KACZ with MC");
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
        GHOST_INFO_LOG("Try both col- and row-major for 1-column densemat with stride 1");
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
        GHOST_PERFWARNING_LOG("Use plain implementation for scattered views");
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
                        GHOST_PERFWARNING_LOG("Using unaligned kernel because base address of result vector is not aligned");
                    }
                    if (!IS_ALIGNED(x->val,al)) {
                        GHOST_PERFWARNING_LOG("Using unaligned kernel because base address of input vector is not aligned");
                    }
                    if (b->stride*b->elSize % al) {
                        GHOST_PERFWARNING_LOG("Using unaligned kernel because stride of result vector does not yield aligned addresses");
                    }
                    if (x->stride*b->elSize % al) {
                        GHOST_PERFWARNING_LOG("Using unaligned kernel because stride of input vector does not yield aligned addresses");
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


                                GHOST_INFO_LOG("Try chunkheight=%s, blocksz=%s, shifts=%s, impl=%s, %s, method %s, storage %s, vec DT %s",
                                        p.chunkheight==-1?"arbitrary":ghost::to_string((long long)p.chunkheight).c_str(),
                                        p.blocksz==-1?"arbitrary":ghost::to_string((long long)p.blocksz).c_str(),
                                        ghost::to_string((long long)p.nshifts).c_str(),
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
            GHOST_INFO_LOG("Found kernel with highest specialization grade: C=%d blocksz=%d align=%d impl=%s",p.chunkheight,p.blocksz,p.alignment,ghost_implementation_string(p.impl));
        } else {
            GHOST_PERFWARNING_LOG("Using potentially non-optimal kernel: C=%d blocksz=%d align=%d impl=%s",p.chunkheight,p.blocksz,p.alignment,ghost_implementation_string(p.impl));
        }
        ret = kernel(x,mat,b,opts);
    } else { // execute plain kernel as fallback
        GHOST_PERFWARNING_LOG("Execute fallback Kaczmarz kernel which is potentially slow!");
        ghost_kacz_fallback(x, mat, b, opts);

    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);
    return ret;
}
