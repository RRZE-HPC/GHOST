#include "ghost/types.h"
#include "ghost/omp.h"

#include "ghost/util.h"
#include "ghost/densemat.h"
#include "ghost/math.h"
#include "ghost/log.h"
#include "ghost/machine.h"
#include "ghost/timing.h"
#include "ghost/sell_spmv_cu_fallback.h"
#include "ghost/cpp11_fixes.h"
#include "ghost/autogen.h"

#include "ghost/sell_spmtv_RACE_gen.h"
#include "ghost/sell_spmtv_RACE_sse_gen.h"
#include "ghost/sell_spmtv_RACE_avx_gen.h"
#include "ghost/sell_spmtv_RACE_avx2_gen.h"
#include "ghost/sell_spmtv_RACE_mic_gen.h"



#include <complex>
#include <unordered_map>
#include <vector>

// Hash function for unordered_map
namespace std
{
    template<> struct hash<ghost_spmtv_RACE_parameters>
    {
        typedef ghost_spmtv_RACE_parameters argument_type;
        typedef std::size_t result_type;
        result_type operator()(argument_type const& a) const
        {
            //printf("hash = %d\n",ghost_hash(ghost_hash(a.mdt,a.blocksz,a.storage), ghost_hash(a.vdt,a.impl,a.chunkheight),a.alignment));
            return ghost_hash(ghost_hash(a.mdt,a.blocksz,a.storage),
                    ghost_hash(a.vdt,a.impl,a.chunkheight),a.alignment);
        }
    };
}

static bool operator==(const ghost_spmtv_RACE_parameters& a, const ghost_spmtv_RACE_parameters& b)
{
    return a.mdt == b.mdt && a.blocksz == b.blocksz && a.storage == b.storage && 
        a.vdt == b.vdt && a.impl == b.impl && a.chunkheight == b.chunkheight &&
        a.alignment == b.alignment;
}


static std::unordered_map<ghost_spmtv_RACE_parameters, ghost_spmtv_RACE_kernel>
ghost_spmtv_RACE_kernels = std::unordered_map<ghost_spmtv_RACE_parameters,ghost_spmtv_RACE_kernel>();


extern "C" ghost_error ghost_spmtv_RACE(ghost_densemat *lhs, ghost_sparsemat *mat, ghost_densemat *rhs, int iterations)
{
    ghost_error ret = GHOST_SUCCESS;


    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    if (rhs->traits.storage != lhs->traits.storage) {
        ERROR_LOG("Different storage layout for in- and output densemats!");
        return GHOST_ERR_INVALID_ARG;
    }
    if (rhs->traits.ncols != lhs->traits.ncols) {
        ERROR_LOG("The number of columns for the densemats does not match!");
        return GHOST_ERR_INVALID_ARG;
    }
    if (!(mat->context->flags & GHOST_PERM_NO_DISTINCTION) && DM_NROWS(lhs) != SPM_NROWS(mat)) {
        ERROR_LOG("Different number of rows for the densemats and matrix!");
        return GHOST_ERR_INVALID_ARG;
    }
    if (((rhs->traits.storage == GHOST_DENSEMAT_COLMAJOR) && 
                (DM_NROWS(rhs->src) != DM_NROWS(rhs))) || 
            ((lhs->traits.storage == GHOST_DENSEMAT_COLMAJOR) && 
            (DM_NROWS(lhs->src) != DM_NROWS(lhs)))) {
        ERROR_LOG("Col-major densemats with masked out rows currently not "
                "supported!");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }

//    ghost_spmtv_RACE_parameters checkPars;

    // if map is empty include generated code for map construction
    if (ghost_spmtv_RACE_kernels.empty()) {
#include "sell_spmtv_RACE.def"
#include "sell_spmtv_RACE_sse.def"
#include "sell_spmtv_RACE_avx.def"
#include "sell_spmtv_RACE_avx2.def"
#include "sell_spmtv_RACE_mic.def"
    }

    ghost_spmtv_RACE_kernel kernel = NULL;
    ghost_spmtv_RACE_parameters p;
    ghost_alignment opt_align;
    ghost_implementation opt_impl;

    int try_storage[2] = {rhs->traits.storage, -1};
    int n_storage = 1;
    if ( (rhs->traits.storage == GHOST_DENSEMAT_COLMAJOR && rhs->nblock == 1 && lhs->nblock == 1) || (rhs->traits.storage == GHOST_DENSEMAT_ROWMAJOR && rhs->stride == 1 && lhs->stride==1) )
    {
        //Try both
        try_storage[0] = GHOST_DENSEMAT_ROWMAJOR;
        try_storage[1] = GHOST_DENSEMAT_COLMAJOR;
        n_storage = 2;
    }

    std::vector<ghost_implementation> try_impl;

    if ((lhs->traits.flags & GHOST_DENSEMAT_SCATTERED) ||
            (rhs->traits.flags & GHOST_DENSEMAT_SCATTERED)) {
        ERROR_LOG("SCATTERED VIEWS NOT IMPLEMENTED");
        opt_impl = GHOST_IMPLEMENTATION_PLAIN;
    } else {
        if (rhs->stride > 1 && rhs->traits.storage == GHOST_DENSEMAT_ROWMAJOR) {
            opt_impl = ghost_get_best_implementation_for_bytesize(rhs->traits.ncols*rhs->elSize);
            if (opt_impl == GHOST_IMPLEMENTATION_PLAIN) {
                // this branch is taken for odd numbers
                // choose a version with remainder loops in this case!
                opt_impl = ghost_get_best_implementation_for_bytesize(PAD(rhs->traits.ncols*rhs->elSize,ghost_machine_simd_width()));
            }
        } else {
            opt_impl = ghost_get_best_implementation_for_bytesize(mat->traits.C*mat->elSize);
        }
    }

    try_impl.push_back(opt_impl);

    //push all lower implementations
    for(int impl = opt_impl; impl>=0; --impl)
    {
        try_impl.push_back((ghost_implementation)impl);
    }

    p.vdt = rhs->traits.datatype;
    p.mdt = mat->traits.datatype;

    int try_chunkheight[2] = {mat->traits.C,-1};
    int try_blocksz[2] = {rhs->traits.ncols,-1};

    int n_chunkheight = sizeof(try_chunkheight)/sizeof(int);
    int n_blocksz = sizeof(try_blocksz)/sizeof(int);
    int pos_chunkheight, pos_blocksz, pos_storage;

    void *lval, *rval;
    lval = lhs->val;
    rval = rhs->val;

    bool optimal = true;

    for (pos_chunkheight = 0; pos_chunkheight < n_chunkheight; pos_chunkheight++) {
        for (pos_blocksz = 0; pos_blocksz < n_blocksz; pos_blocksz++) {
            for (pos_storage = 0; pos_storage < n_storage; pos_storage++) {
                p.storage = (ghost_densemat_storage)try_storage[pos_storage];
                for (std::vector<ghost_implementation>::iterator impl = try_impl.begin(); impl != try_impl.end(); impl++) {
                    p.impl = *impl;
                    int al = ghost_implementation_alignment(p.impl);
                    if (IS_ALIGNED(lval,al) && IS_ALIGNED(rval,al) && ((lhs->traits.ncols == 1 && lhs->stride == 1) || (!((lhs->stride*lhs->elSize) % al) && !((rhs->stride*rhs->elSize) % al)))) {
                        opt_align = GHOST_ALIGNED;
                    } else {
                        if (!IS_ALIGNED(lval,al)) {
                            PERFWARNING_LOG("Using unaligned kernel because base address of result vector is not aligned");
                        }
                        if (!IS_ALIGNED(rval,al)) {
                            PERFWARNING_LOG("Using unaligned kernel because base address of input vector is not aligned");
                        }
                        if (lhs->stride*lhs->elSize % al) {
                            PERFWARNING_LOG("Using unaligned kernel because stride of result vector does not yield aligned addresses");
                        }
                        if (rhs->stride*lhs->elSize % al) {
                            PERFWARNING_LOG("Using unaligned kernel because stride of input vector does not yield aligned addresses");
                        }
                        opt_align = GHOST_UNALIGNED;
                    }

                    for (p.alignment = opt_align; (int)p.alignment >= GHOST_UNALIGNED; p.alignment = (ghost_alignment)((int)p.alignment-1)) {
                        p.chunkheight = try_chunkheight[pos_chunkheight];
                        p.blocksz = try_blocksz[pos_blocksz];

                        INFO_LOG("Try chunkheight=%s, blocksz=%s, impl=%s, storage=%s, %s",
                                p.chunkheight==-1?"arbitrary":ghost::to_string((long long)p.chunkheight).c_str(),
                                p.blocksz==-1?"arbitrary":ghost::to_string((long long)p.blocksz).c_str(),
                                ghost_implementation_string(p.impl),(p.storage==GHOST_DENSEMAT_ROWMAJOR)?"rowmajor":"colmajor",p.alignment==GHOST_UNALIGNED?"unaligned":(p.alignment==GHOST_ALIGNED?"aligned":"alignment_any"));

                        kernel = ghost_spmtv_RACE_kernels[p];

                        if(kernel!=NULL) {
                           goto end_of_loop;
                        }

                        if(p.alignment == GHOST_UNALIGNED) {
                            optimal = false;
                        }
                    }
                }
            }
        }
    }

end_of_loop:

    if (pos_blocksz || pos_chunkheight) {
        ghost_autogen_set_missing();
    }
    std::ostringstream oss;
    oss << try_chunkheight[0] << "," << try_blocksz[0];
    ghost_autogen_string_add("SPMTV",oss.str().c_str());


    if (kernel) {
        if (optimal) {
            INFO_LOG("Found kernel with highest specialization grade: C=%d blocksz=%d align=%d impl=%s",p.chunkheight,p.blocksz,p.alignment,ghost_implementation_string(p.impl));
        } else {
            PERFWARNING_LOG("Using potentially non-optimal kernel: C=%d blocksz=%d align=%d impl=%s",p.chunkheight,p.blocksz,p.alignment,ghost_implementation_string(p.impl));
        }
        ret = kernel(lhs,mat,rhs,iterations);
    } else { // execute plain kernel as fallback
           PERFWARNING_LOG("Using fallback kernel");
           ghost_spmtv_RACE_fallback(lhs,mat,rhs,iterations);
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);
    return ret;
}


