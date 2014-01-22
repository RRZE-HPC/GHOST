#include "ghost/config.h"
#undef GHOST_HAVE_MPI
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/constants.h"
#include "ghost/vec.h"

#include <cuda_runtime.h>
#include <cusparse_v2.h>

void ghost_cu_crsspmv(ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
    WARNING_LOG("Not implemented!");

}
