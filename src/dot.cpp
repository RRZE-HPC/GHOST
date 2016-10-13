#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/densemat.h"
#include "ghost/densemat_rm.h"
#include "ghost/densemat_cm.h"
#include "ghost/util.h"
#include "ghost/dot_avx_gen.h"
#include "ghost/dot_plain_gen.h"
#include "ghost/machine.h"
#include "ghost/dot.h"
#include "ghost/timing.h"
#include "ghost/compatibility_check.h"

#include <unordered_map>

using namespace std;

// Hash function for unordered_map
namespace std
{
    template<> struct hash<ghost_dot_parameters>
    {
        typedef ghost_dot_parameters argument_type;
        typedef std::size_t result_type;
        result_type operator()(argument_type const& a) const
        {
            return ghost_hash(a.dt,a.blocksz,ghost_hash(a.impl,a.storage,a.alignment));
        }
    };
}


static bool operator==(const ghost_dot_parameters& a, const ghost_dot_parameters& b)
{
    return a.dt == b.dt && a.blocksz == b.blocksz && a.impl == b.impl && a.storage == b.storage && a.alignment == b.alignment;
}

static unordered_map<ghost_dot_parameters, ghost_dot_kernel> ghost_dot_kernels;

ghost_error ghost_localdot(void *res, ghost_densemat *vec1, ghost_densemat *vec2)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);

    ghost_error ret = GHOST_SUCCESS;

    //////////////// check compatibility /////////////
    ghost_compatible_vec_vec check = GHOST_COMPATIBLE_VEC_VEC_INITIALIZER;
    check.A = vec1;
    check.transA = 'T';
    check.B = vec2;   

    ret = ghost_check_vec_vec_compatibility(&check,vec1->context);
    ///////////////////////////////////////////////////
 
    if (ghost_dot_kernels.empty()) {
#include "dot_avx.def"
#include "dot_plain.def"
    }

    memset(res,0,vec1->traits.ncols*vec2->elSize);

    int al = ghost_machine_alignment();
    ghost_dot_parameters p;
    p.dt = vec1->traits.datatype;
    p.alignment = GHOST_ALIGNED;
    
    ghost_dot_kernel kernel = NULL;
#ifdef GHOST_BUILD_MIC
    p.impl = GHOST_IMPLEMENTATION_MIC;
#elif defined(GHOST_BUILD_AVX)
    p.impl = GHOST_IMPLEMENTATION_AVX;
#elif defined(GHOST_BUILD_SSE)
    p.impl = GHOST_IMPLEMENTATION_SSE;
#else
    p.impl = GHOST_IMPLEMENTATION_PLAIN;
#endif
#ifdef GHOST_HAVE_CUDA
    ghost_type type;
    ghost_type_get(&type);
    if (type == GHOST_TYPE_CUDA) {
        p.impl = GHOST_IMPLEMENTATION_CUDA;
    }
#endif

    if (vec1->traits.flags & GHOST_DENSEMAT_SCATTERED || vec2->traits.flags & GHOST_DENSEMAT_SCATTERED ||
            p.impl == GHOST_IMPLEMENTATION_CUDA) {
        goto out;
    }

    p.storage = vec1->traits.storage;
    p.blocksz = vec1->traits.ncols;

    if (IS_ALIGNED(vec1->val,al) && IS_ALIGNED(vec2->val,al) && !((vec1->stride*vec1->elSize) % al) && !((vec2->stride*vec2->elSize) % al)) {
        p.alignment = GHOST_ALIGNED;
    } else {
        p.alignment = GHOST_UNALIGNED;
    }

    INFO_LOG("Initial try: storage=%s, blocksz=%d, alignment=%d, impl=%s",ghost_densemat_storage_string(vec1->traits.storage),p.blocksz,p.alignment,ghost_implementation_string(p.impl));
    kernel = ghost_dot_kernels[p];
    if (!kernel && p.alignment == GHOST_ALIGNED) {
        PERFWARNING_LOG("Try unaligned version");
        p.alignment = GHOST_UNALIGNED;
        kernel = ghost_dot_kernels[p];
    }
    if (!kernel) {
        PERFWARNING_LOG("Try plain version");
        p.impl = GHOST_IMPLEMENTATION_PLAIN;
        kernel = ghost_dot_kernels[p];
    }

    if (kernel) {
        ret = kernel(res,vec1,vec2);
    }

out:
    if (!kernel) {
        ghost_dot_perf_args dot_perfargs;
        dot_perfargs.ncols = vec1->traits.ncols;
        dot_perfargs.globnrows = DM_GNROWS(vec1);
        dot_perfargs.dt = vec1->traits.datatype;
        if (vec1 == vec2) {
            dot_perfargs.samevec = true;
        } else {
            dot_perfargs.samevec = false;
        }
        ghost_timing_set_perfFunc(NULL,__ghost_functag,ghost_dot_perf,(void *)&dot_perfargs,sizeof(dot_perfargs),"GB/s");
        PERFWARNING_LOG("Fallback to vanilla dot implementation");
        
        ghost_location commonlocation = vec1->traits.location & vec2->traits.location;
        
        typedef ghost_error (*ghost_dot_kernel)(ghost_densemat*, void*, ghost_densemat*);
        ghost_dot_kernel kernels[2][2] = {{NULL,NULL},{NULL,NULL}};
        kernels[GHOST_HOST_IDX][GHOST_RM_IDX] = &ghost_densemat_rm_dotprod_selector;
        kernels[GHOST_HOST_IDX][GHOST_CM_IDX] = &ghost_densemat_cm_dotprod_selector;
#ifdef GHOST_HAVE_CUDA
        kernels[GHOST_DEVICE_IDX][GHOST_RM_IDX] = &ghost_densemat_cu_rm_dotprod;
        kernels[GHOST_DEVICE_IDX][GHOST_CM_IDX] = &ghost_densemat_cu_cm_dotprod;
#endif

        SELECT_BLAS1_KERNEL(kernels,commonlocation,vec1->traits.compute_at,vec1->traits.storage,ret,vec1,res,vec2);
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);
    return ret;
}

ghost_error ghost_dot(void *res, ghost_densemat *vec1, ghost_densemat *vec2)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);

    ghost_error ret = GHOST_SUCCESS;
    GHOST_CALL_RETURN(ghost_localdot(res,vec1,vec2));

#ifdef GHOST_HAVE_MPI
    ghost_mpi_comm mpicomm = MPI_COMM_WORLD;
    // TODO use MPI comm of densemat
    if (mpicomm != MPI_COMM_NULL) {
        GHOST_INSTR_START("reduce")
        ghost_mpi_op sumOp;
        ghost_mpi_datatype mpiDt;
        ghost_mpi_op_sum(&sumOp,vec1->traits.datatype);
        ghost_mpi_datatype_get(&mpiDt,vec1->traits.datatype);
        int v;
        for (v=0; v<MIN(vec1->traits.ncols,vec2->traits.ncols); v++) {
            MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE, (char *)res+vec1->elSize*v, 1, mpiDt, sumOp, mpicomm));
        }
        GHOST_INSTR_STOP("reduce")
    }
#endif


    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);
    return ret;


}
