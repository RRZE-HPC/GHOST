/**
 * @file tsmm_cu_kernel.h
 * @brief TSMM CUDA kernels.
 * @author Dominik Ernst <dominik.ernst@fau.de>
 */
#ifndef GHOST_TSMM_CU_KERNEL_H
#define GHOST_TSMM_CU_KERNEL_H

#include "ghost/config.h"
#include "ghost/cu_complex.h"
#include "ghost/cu_util.h"
#include "ghost/types.h"
#include <cublas_v2.h>
#include <iostream>
#include <typeinfo>

namespace {

template<typename T>
bool eq(const T lhs, const T rhs)
{
    return lhs == rhs;
};

template<>
bool eq<cuDoubleComplex>(const cuDoubleComplex lhs, const cuDoubleComplex rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y;
}

template<>
bool eq<cuFloatComplex>(const cuFloatComplex lhs, const cuFloatComplex rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y;
}

template<typename T, typename iT, int M, int N, int BLOCKSIZE, bool BETAISZERO>
static __global__ void __launch_bounds__(BLOCKSIZE)
    tsmm_fix_fb_kernel(const T *__restrict__ A, const iT *__restrict__ B, T *out, const int K,
        const int lda, const int ldb, const int ldc, iT alpha, iT beta)
{
    int tidx = blockIdx.x * BLOCKSIZE + threadIdx.x;
    int n = tidx % N;

    const bool fitsShm = (M * N * sizeof(iT) <= (1 << 14)); // bCache fits in 16kB shared memory

    __shared__ iT bCache[fitsShm ? M : 1][fitsShm ? N : 1];

#pragma unroll(1)
    for (int mn = threadIdx.x; mn < M * N; mn += BLOCKSIZE) {
        int tn = mn / M;
        int tm = mn % M;
        bCache[tm][tn] = B[tn * ldb + tm];
    }

    __syncthreads();

    if (tidx / N == gridDim.x * BLOCKSIZE / N && !BETAISZERO) return;

    int row = tidx / N;
    for (; row < K / 2; row += gridDim.x * BLOCKSIZE / N) {
        iT sum1, sum2;
        zero(sum1);
        zero(sum2);

        const int o1 = row * lda;
        const int o2 = (row + K / 2) * lda;

        for (int m = 0; m < M; m++) {
            iT bV = bCache[m][n];
            sum1 = axpy(sum1, (iT)A[o1 + m], bV);
            sum2 = axpy(sum2, (iT)A[o2 + m], bV);
        }
        if (BETAISZERO) {
            out[row * ldc + n] = scale(alpha, sum1);
            out[(row + K / 2) * ldc + n] = scale(alpha, sum2);
        } else {
            out[row * ldc + n] = axpby(sum1, (iT)out[row * ldc + n], alpha, beta);
            out[(row + K / 2) * ldc + n] = axpby(sum2, (iT)out[(row + K / 2) * ldc + n], alpha, beta);
        }
    }

    // remainder loop
    for (row += K / 2; row < K; row += gridDim.x * BLOCKSIZE / N) {
        iT sum;
        zero(sum);

#pragma unroll(M <= 8 ? M : 1)
        for (int m = 0; m < M; m++) { sum = axpy(sum, (iT)A[row * lda + m], bCache[m][n]); }
        if (BETAISZERO) {
            out[row * ldc + n] = scale(alpha, sum);
        } else {
            out[row * ldc + n] = axpby(sum, (iT)out[row * ldc + n], alpha, beta);
        }
    }
}
} // namespace

template<typename T, typename iT, int M, int N>
bool ghost_tsmm_cu_rm_cm(T *C, const T *A, const iT *B, const iT alpha, const iT beta,
    const ghost_lidx K, const ghost_lidx ldc, const ghost_lidx lda, const ghost_lidx ldb)
{

    const bool fitsShm = (M * N * sizeof(iT) <= (1 << 14)); // bCache fits in 16kB shared memory
    if (!fitsShm) return false;

    const int threadsPerBlock = (M * N > 1024) ? (M * N > 55 ? 1024 : 512) : 256;

    int deviceUsed;
    cudaGetDevice(&deviceUsed);
    cudaDeviceProp prop;
    ghost_cu_deviceprop_get(&prop);

    int numBlocks;
    ghost_error ret = GHOST_SUCCESS;

    CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
                  tsmm_fix_fb_kernel<T, iT, M, N, threadsPerBlock, false>, threadsPerBlock, 0),
        ret);
    int blockCount = prop.multiProcessorCount * numBlocks;
    iT Tzero;
    zero(Tzero);
    if (eq(beta, Tzero)) {
        tsmm_fix_fb_kernel<T, iT, M, N, threadsPerBlock, true>
            <<<blockCount, threadsPerBlock>>>(A, B, C, K, lda, ldb, ldc, alpha, beta);
    } else {
        tsmm_fix_fb_kernel<T, iT, M, N, threadsPerBlock, false>
            <<<blockCount, threadsPerBlock>>>(A, B, C, K, lda, ldb, ldc, alpha, beta);
    }
    CUDA_CALL(cudaGetLastError(), ret);
    if (ret != GHOST_SUCCESS) return false;
    return true;
}

#endif
