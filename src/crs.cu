#include <ghost_config.h>
#undef GHOST_HAVE_MPI
#include <ghost_types.h>
#include <ghost_util.h>
#include <ghost_constants.h>
#include <ghost_vec.h>

#include <cuda_runtime.h>
#include <cusparse_v2.h>

void ghost_cu_crsspmv(ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
    WARNING_LOG("Not implemented!");

}
