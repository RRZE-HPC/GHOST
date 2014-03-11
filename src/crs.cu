#include "ghost/config.h"
#undef GHOST_HAVE_MPI
#include "ghost/types.h"
#include "ghost/crs.h"
#include "ghost/log.h"

#include <cuda_runtime.h>
#include <cusparse_v2.h>

ghost_error_t ghost_cu_crsspmv(ghost_sparsemat_t *mat, ghost_densemat_t * lhs, ghost_densemat_t * rhs, int options)
{
    ERROR_LOG("CUDA CRS spMV not implemented");
    return GHOST_ERR_NOT_IMPLEMENTED;

}
