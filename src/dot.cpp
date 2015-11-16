#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/densemat.h"
#include "ghost/util.h"
#include "ghost/dot_avx_gen.h"
#include "ghost/dot_plain_gen.h"
#include "ghost/machine.h"
#include "ghost/dot.h"
#include "ghost/timing.h"

#include <map>

using namespace std;

static bool operator<(const ghost_dot_parameters_t &a, const ghost_dot_parameters_t &b) 
{ 
    return (ghost_hash(a.dt,a.blocksz,ghost_hash(a.impl,a.storage,0)) < ghost_hash(b.dt,b.blocksz,ghost_hash(b.impl,b.storage,0)));
}

static map<ghost_dot_parameters_t, ghost_dot_kernel_t> ghost_dot_kernels;

ghost_error_t ghost_dot(void *res, ghost_densemat_t *vec1, ghost_densemat_t *vec2)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_COMMUNICATION);
    GHOST_CALL_RETURN(ghost_localdot(res,vec1,vec2));
#ifdef GHOST_HAVE_MPI
    if (vec1->context) {
        GHOST_INSTR_START("reduce")
        ghost_mpi_op_t sumOp;
        ghost_mpi_datatype_t mpiDt;
        ghost_mpi_op_sum(&sumOp,vec1->traits.datatype);
        ghost_mpi_datatype(&mpiDt,vec1->traits.datatype);
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

ghost_error_t ghost_localdot(void *res, ghost_densemat_t *vec1, ghost_densemat_t *vec2)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);

    ghost_error_t ret = GHOST_SUCCESS;

    if (ghost_dot_kernels.empty()) {
#include "dot_avx.def"
#include "dot_plain.def"
    }

    memset(res,0,vec1->traits.ncols*vec2->elSize);

    ghost_dot_parameters_t p;
    p.dt = vec1->traits.datatype;
    p.alignment = GHOST_ALIGNED;
    
    ghost_dot_kernel_t kernel = NULL;
#ifdef GHOST_HAVE_MIC
    p.impl = GHOST_IMPLEMENTATION_MIC;
#elif defined(GHOST_HAVE_AVX)
    p.impl = GHOST_IMPLEMENTATION_AVX;
#elif defined(GHOST_HAVE_SSE)
    p.impl = GHOST_IMPLEMENTATION_SSE;
#elif defined(GHOST_HAVE_CUDA)
    ghost_type type;
    ghost_type_get(&type);
    if (type == GHOST_TYPE_CUDA) {
        p.impl = GHOST_IMPLEMENTATION_CUDA;
    }
#else
    p.impl = GHOST_IMPLEMENTATION_PLAIN;
#endif

    if (vec1->traits.flags & GHOST_DENSEMAT_SCATTERED || vec2->traits.flags & GHOST_DENSEMAT_SCATTERED ||
            p.impl == GHOST_IMPLEMENTATION_CUDA) {
        PERFWARNING_LOG("Fallback to vanilla dot implementation");
        GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);
        return vec1->localdot_vanilla(vec1,res,vec2);
    }

    p.storage = vec1->traits.storage;
    p.blocksz = vec1->traits.ncols;

    int al = ghost_machine_alignment();
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
    } else {
        PERFWARNING_LOG("Calling fallback dot");
        ret = vec1->localdot_vanilla(vec1,res,vec2);
    }


    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);
    return ret;


}
