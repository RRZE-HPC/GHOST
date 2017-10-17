#include "ghost/cu_sell_kernel.h"
#include "ghost/cu_util.h"
#include "ghost/error.h"
#include "ghost/types.h"
#include "ghost/util.h"

#ifdef GHOST_HAVE_CUDA
#include <cuda.h>
#endif

ghost_error ghost_cu_reduce(void *out, void *data, ghost_datatype dt, ghost_lidx n)
{
#ifdef GHOST_HAVE_CUDA

    int blockSize;
    int blockCount;

    ghost_cu_deviceprop devProp;
    GHOST_CALL_RETURN(ghost_cu_deviceprop_get(&devProp));
    if (devProp.major < 6) {
        blockSize = 1024;
        blockCount = 1;
    } else {
        blockSize = 256;
        blockCount = CEILDIV(n, blockSize);
    }

    if (dt & GHOST_DT_COMPLEX) {
        if (dt & GHOST_DT_DOUBLE) {
            ghost_deviceReduceSum<cuDoubleComplex><<<blockCount, blockSize>>>((cuDoubleComplex *)data, (cuDoubleComplex *)out, n);
        } else {
            ghost_deviceReduceSum<cuFloatComplex><<<blockCount, blockSize>>>((cuFloatComplex *)data, (cuFloatComplex *)out, n);
        }
    } else {
        if (dt & GHOST_DT_DOUBLE) {
            ghost_deviceReduceSum<double><<<blockCount, blockSize>>>((double *)data, (double *)out, n);
        } else {
            ghost_deviceReduceSum<float><<<blockCount, blockSize>>>((float *)data, (float *)out, n);
        }
    }

#else
    UNUSED(out);
    UNUSED(data);
    UNUSED(dt);
    UNUSED(n);
#endif
    return GHOST_SUCCESS;
}

ghost_error ghost_cu_reduce_multiple(void *out, void *data, ghost_datatype dt, ghost_lidx n, ghost_lidx ncols)
{
#ifdef GHOST_HAVE_CUDA

    int blockSize;
    int blockCount;

    ghost_cu_deviceprop devProp;
    GHOST_CALL_RETURN(ghost_cu_deviceprop_get(&devProp));

    blockSize = 1024;
    blockCount = ncols;

    if (dt & GHOST_DT_COMPLEX) {
        if (dt & GHOST_DT_DOUBLE) {
            ghost_deviceReduceSumMultiple<cuDoubleComplex><<<blockCount, blockSize>>>((cuDoubleComplex *)data, (cuDoubleComplex *)out, n, ncols);
        } else {
            ghost_deviceReduceSumMultiple<cuFloatComplex><<<blockCount, blockSize>>>((cuFloatComplex *)data, (cuFloatComplex *)out, n, ncols);
        }
    } else {
        if (dt & GHOST_DT_DOUBLE) {
            ghost_deviceReduceSumMultiple<double><<<blockCount, blockSize>>>((double *)data, (double *)out, n, ncols);
        } else {
            ghost_deviceReduceSumMultiple<float><<<blockCount, blockSize>>>((float *)data, (float *)out, n, ncols);
        }
    }

#else
    UNUSED(out);
    UNUSED(data);
    UNUSED(dt);
    UNUSED(n);
#endif
    return GHOST_SUCCESS;
}
