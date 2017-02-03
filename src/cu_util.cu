#include "ghost/types.h"
#include "ghost/error.h"
#include "ghost/util.h"
#include "ghost/cu_util.h"
#include "ghost/cu_sell_kernel.h"
#ifdef GHOST_HAVE_CUDA
#include <cuda.h>
#include <cuda.h>
#endif

ghost_error ghost_cu_reduce(void *out, void *data, ghost_datatype dt, ghost_lidx n)
{
#ifdef GHOST_HAVE_CUDA
    struct cudaDeviceProp devProp;
    int cu_device;
    GHOST_CALL_RETURN(ghost_cu_device(&cu_device));
    CUDA_CALL_RETURN(cudaGetDeviceProperties(&devProp,cu_device));
    if (devProp.major < 6) {
        // call version which requires only a single block
        if (dt & GHOST_DT_COMPLEX) {
            if (dt & GHOST_DT_DOUBLE) {
                ghost_deviceReduceSum<cuDoubleComplex><<<1,1024>>>((cuDoubleComplex *)data,(cuDoubleComplex *)out,n);
            } else {
                ghost_deviceReduceSum<cuFloatComplex><<<1,1024>>>((cuFloatComplex *)data,(cuFloatComplex *)out,n);
            }
        } else {
            if (dt & GHOST_DT_DOUBLE) {
                ghost_deviceReduceSum<double><<<1,1024>>>((double *)data,(double *)out,n);
            } else {
                ghost_deviceReduceSum<float><<<1,1024>>>((float *)data,(float *)out,n);
            }
        }

    } else {
        // call version with atomic adds
        const int block = 256;
        const int grid = CEILDIV(n,block);
        if (dt & GHOST_DT_COMPLEX) {
            if (dt & GHOST_DT_DOUBLE) {
                ghost_deviceReduceSum<cuDoubleComplex><<<grid,block>>>((cuDoubleComplex *)data,(cuDoubleComplex *)out,n);
            } else {
                ghost_deviceReduceSum<cuFloatComplex><<<grid,block>>>((cuFloatComplex *)data,(cuFloatComplex *)out,n);
            }
        } else {
            if (dt & GHOST_DT_DOUBLE) {
                ghost_deviceReduceSum<double><<<grid,block>>>((double *)data,(double *)out,n);
            } else {
                ghost_deviceReduceSum<float><<<grid,block>>>((float *)data,(float *)out,n);
            }
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
