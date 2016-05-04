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
#include "ghost/cu_sell_kernel.h"

extern __shared__ char shared[];

template<typename T,int M, int K, bool betaiszero> 
static __global__ void ghost_tsmm_cu_rm_cm(T * const __restrict__ x, const T * const __restrict__ v, const T * const __restrict__ w, const T alpha, const T beta, ghost_lidx nrows, ghost_lidx stridex, ghost_lidx stridev, ghost_lidx stridew)
{
    if (0) {//M >= K && !(M%K)) {
        const int xheight = blockDim.x/K;
        T * shmem = (T *)shared;
        const int firstrowofblock = blockIdx.x*xheight;
        const int rowinblock = threadIdx.x/K;
        const int xrow = firstrowofblock + rowinblock;
        const int xcol = threadIdx.x%K;
        int m;
        T tmp;
        
        int i;

#pragma unroll
        for (i=0; i<M/K; i++) {
            shmem[i*blockDim.x+threadIdx.x] = v[firstrowofblock*stridev+i*blockDim.x+threadIdx.x];
        }

        if (betaiszero) {
            zero<T>(tmp);
        } else {
            tmp = scale<T>(x[xrow*stridex+threadIdx.x],beta);
        }

#pragma unroll
        for (m=0; m<M; m++) {
            tmp = axpy<T,T>(tmp,alpha,scale<T>(shmem[rowinblock*M+m],w[xcol*stridew+m]));
        }

        x[xrow*stridex+xcol] = tmp;

    } else {

        int row = blockIdx.x*blockDim.y+threadIdx.y;
        int m;
        T tmp;

        for (;row < nrows; row+=gridDim.x*blockDim.y) {
            if (betaiszero) {
                zero<T>(tmp);
            } else {
                tmp = scale<T>(x[row*stridex+threadIdx.x],beta);
            }
#pragma unroll
            for (m=0; m<M; m++) {
                tmp = axpy<T,T>(tmp,alpha,scale<T>(v[row*stridev+m],w[threadIdx.x*stridew+m]));
            }
            x[row*stridex+threadIdx.x] = tmp;
        }
    }
    
}

#endif
