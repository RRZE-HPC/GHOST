#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/densemat.h"
#include "ghost/util.h"
#include "ghost/dot_avx_gen.h"
#include "ghost/dot_plain_gen.h"
#include "ghost/machine.h"
#include "ghost/dot.h"
#include "ghost/timing.h"

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
            return ghost_hash(a.dt,a.blocksz,ghost_hash(a.impl,a.storage,0));
        }
    };
}


static bool operator==(const ghost_dot_parameters& a, const ghost_dot_parameters& b)
{
    return a.dt == b.dt && a.blocksz == b.blocksz && a.impl == b.impl && a.storage == b.storage;
}

static unordered_map<ghost_dot_parameters, ghost_dot_kernel> ghost_dot_kernels;

ghost_error ghost_dot(void *res, ghost_densemat *vec1, ghost_densemat *vec2)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_COMMUNICATION);
    GHOST_CALL_RETURN(ghost_localdot(res,vec1,vec2));
#ifdef GHOST_HAVE_MPI
    if (vec1->context) {
        GHOST_INSTR_START("reduce")
        ghost_mpi_op sumOp;
        ghost_mpi_datatype mpiDt;
        ghost_mpi_op_sum(&sumOp,vec1->traits.datatype);
        ghost_mpi_datatype_get(&mpiDt,vec1->traits.datatype);
        int v;
        if (vec1->context) {
            for (v=0; v<MIN(vec1->traits.ncols,vec2->traits.ncols); v++) {
                MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE, (char *)res+vec1->elSize*v, 1, mpiDt, sumOp, vec1->context->mpicomm));
            }
        }
        GHOST_INSTR_STOP("reduce")
    }
#endif

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_COMMUNICATION);
    return GHOST_SUCCESS;
}

ghost_error ghost_localdot(void *res, ghost_densemat *vec1, ghost_densemat *vec2)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);

    ghost_error ret = GHOST_SUCCESS;

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

    kernel = ghost_dot_kernels[p];
    if (!kernel) {
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
        if (vec1->context) {
            dot_perfargs.globnrows = vec1->context->gnrows;
        } else {
            dot_perfargs.globnrows = vec1->traits.nrows;
        }
        dot_perfargs.dt = vec1->traits.datatype;
        if (vec1 == vec2) {
            dot_perfargs.samevec = true;
        } else {
            dot_perfargs.samevec = false;
        }
        ghost_timing_set_perfFunc(NULL,__ghost_functag,ghost_dot_perf,(void *)&dot_perfargs,sizeof(dot_perfargs),"GB/s");
        PERFWARNING_LOG("Fallback to vanilla dot implementation");
        ret = vec1->localdot_vanilla(vec1,res,vec2);
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);
    return ret;


}
