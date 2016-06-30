/**
 * @file tsmm_inplace_cu_kernel.h
 * @brief TSMM-inplace CUDA kernels.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_TSMM_INPLACE_CU_KERNEL_H
#define GHOST_TSMM_INPLACE_CU_KERNEL_H

#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/cu_complex.h"

template<typename T,int NCOLSOUT, int NCOLSIN> __global__ void ghost_tsmm_inplace_cu_rm_cm(T * x, const T * const __restrict__ w, const T alpha, const T beta, ghost_lidx nrows, ghost_lidx stridex, ghost_lidx stridew)
{
    int row = blockIdx.x*blockDim.y+threadIdx.y;
    int m;
    T tmp[NCOLSOUT];

    for (;row < nrows; row+=gridDim.x*blockDim.y) {
        tmp[threadIdx.x] = scale<T>(x[row*stridex+threadIdx.x],beta);
        for (m=0; m<NCOLSIN; m++) {
            tmp[threadIdx.x] = axpy<T,T>(tmp[threadIdx.x],alpha,scale<T>(x[row*stridex+m],w[threadIdx.x*stridew+m]));
        }

        x[row*stridex+threadIdx.x] = tmp[threadIdx.x];
    }
}

#endif
