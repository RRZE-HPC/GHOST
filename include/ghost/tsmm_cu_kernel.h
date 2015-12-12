/**
 * @file tsmm_cu_kernel.h
 * @brief TSMM CUDA kernels.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_TSMM_CU_KERNEL_H
#define GHOST_TSMM_CU_KERNEL_H

#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/cu_complex.h"

template<typename T,int M, int K> 
static __global__ void ghost_tsmm_cu_rm_cm(T * const __restrict__ x, const T * const __restrict__ v, const T * const __restrict__ w, const T alpha, const T beta, ghost_lidx_t nrows, ghost_lidx_t stridex, ghost_lidx_t stridev, ghost_lidx_t stridew)
{
    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int k;
    int m;
    T tmp;

    for (;row < nrows; row+=gridDim.x*blockDim.x) {
        for (k=0; k<K; k++) {
            tmp = scale<T>(x[row*stridex+k],beta);
            for (m=0; m<M; m++) {
                tmp = axpy<T,T>(tmp,alpha,scale<T>(v[row*stridev+m],w[k*stridew+m]));
            }
            x[row*stridex+k] = tmp;
        }
    }
}

#endif
