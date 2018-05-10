/**
 * @file tsmttsm_cu_kernel.h
 * @brief TSMTTSM CUDA kernels.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 * @author Dominik Ernst <dominik.ernst@fau.de>
 */
#ifndef GHOST_TSMTTSM_CU_KERNEL_H
#define GHOST_TSMTTSM_CU_KERNEL_H
#include <cublas_v2.h>
#include <iostream>
#include <typeinfo>
#include "ghost/config.h"
#include "ghost/cu_complex.h"
#include "ghost/cu_util.h"
#include "ghost/types.h"
#include "ghost/cu_temp_buffer_malloc.h"

namespace {
void *d_temp_storage = NULL;
size_t temp_storage_bytes = 0;
template<typename oT, int M, int N>
__global__ void deviceReduce(
    oT *blockResults, oT *result, oT alpha, oT beta, int blockCount, int lda, int ldb, int ldc)
{
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= M * N) return;
    int n = tidx / M;
    int m = tidx % M;

    oT sum;
    zero(sum);
    for (int i = 0; i < blockCount; i++) {
        sum = accu(sum, blockResults[i * N * M + n * M + m]);
    }

    result[n * ldc + m] = accu(scale(result[n * ldc + m], beta), scale2(sum, alpha));
}

template<typename T, bool conjv, bool TRANSPOSE>
__device__ T condConj1(T v)
{
    if (conjv && !TRANSPOSE) v = conj(v);
    return v;
}
template<typename T, bool conjv, bool TRANSPOSE>
__device__ T condConj2(T v)
{
    if (conjv && TRANSPOSE) v = conj(v);
    return v;
}

// round to next smaller power of two
__device__ int roundPoT(int v)
{
    int r = v;
    r |= r >> 1;
    r |= r >> 2;
    r |= r >> 4;
    r |= r >> 8;
    r |= r >> 16;
    r -= (r >> 1);
    return r;
}

template<typename T, typename oT, int conjv, int M, int N, int BLOCKSIZE, bool TRANSPOSE, bool SELF>
__global__ void __launch_bounds__(BLOCKSIZE) genv7_blockProductKernel(
    const T *A, const T *B, oT *out, const int K, const int lda, const int ldb, const int ldc)
{
    const int rowsPerBlock = BLOCKSIZE / M;
    int m = threadIdx.x % M;
    int localRow = threadIdx.x / M;
    int bOffset = localRow * ldb + m;
    int aOffset = localRow * lda + m;
    if (m >= N) bOffset = localRow * ldb + 0;
    if (bOffset >= rowsPerBlock * ldb) bOffset = 0;
    if (aOffset >= rowsPerBlock * lda) aOffset = 0;

    __shared__ oT blockStorage[rowsPerBlock * M * (sizeof(T) > sizeof(oT) ? 2 : 1)];
    T *rowCache = reinterpret_cast<T *>(blockStorage);

    zero(blockStorage[threadIdx.x]);
    __syncthreads();

    oT threadSum[N];
    for (int n = 0; n < N; n++) {
        zero(threadSum[n]);
    }

    // Block synchronous loop
    int idx = blockIdx.x * rowsPerBlock;
    T avNow = __ldg(A + idx * lda + aOffset);
    T bvNow = __ldg(B + idx * ldb + bOffset);
    T avNext;
    T bvNext;
    zero(avNext);
    zero(bvNext);

    for (; idx < K - rowsPerBlock; idx += gridDim.x * rowsPerBlock) {
        int idxNext = min(K - rowsPerBlock, idx + gridDim.x * rowsPerBlock);
        avNext = __ldg(A + idxNext * lda + aOffset);

        if (!SELF) {
            bvNext = __ldg(B + idxNext * ldb + bOffset);
        } else {
            bvNext = avNext;
        }
        __syncthreads();
        rowCache[threadIdx.x] = bvNow;
        __syncthreads();

        int localAddress = threadIdx.x - m;
        for (int n = 0; n < N; n++) {
            threadSum[n] = axpy(threadSum[n], condConj1<oT, conjv, TRANSPOSE>((oT)avNow),
                condConj2<oT, conjv, TRANSPOSE>((oT)rowCache[localAddress + n]));
        }
        avNow = avNext;
        bvNow = bvNext;
    }

    // Remainder loop
    for (idx = idx + localRow; idx < K; idx += gridDim.x * rowsPerBlock) {
        T av = A[idx * lda + m];
        for (int n = 0; n < N; n++) {
            threadSum[n] = axpy(threadSum[n], condConj1<oT, conjv, TRANSPOSE>((oT)av),
                condConj2<oT, conjv, TRANSPOSE>((oT)B[idx * ldb + n]));
        }
    }

    const int redSteps = roundPoT(rowsPerBlock);

    // Calculate block results
    for (int n = 0; n < N; n++) {
        __syncthreads();
        blockStorage[threadIdx.x] = threadSum[n];
        __syncthreads();

        for (unsigned int s = redSteps; s > 0; s /= 2) {
            if (localRow < s && localRow < rowsPerBlock - s) {
                blockStorage[localRow * M + m] =
                    accu(blockStorage[localRow * M + m], blockStorage[(localRow + s) * M + m]);
            }
            __syncthreads();
        }

        if (threadIdx.x < M) {
            if (TRANSPOSE) {
                out[blockIdx.x * M * N + m * N + n] = blockStorage[m];
            } else {
                out[blockIdx.x * N * M + n * M + m] = blockStorage[m];
            }
        }
    }
}
} // namespace

template<typename T, typename oT, int M, int N, int conjv>
static ghost_error ghost_tsmttsm_cu_rm(oT *const __restrict__ C, const T *const __restrict__ A,
    const T *const __restrict__ B, const oT alpha, const oT beta, ghost_lidx K, ghost_lidx ldc,
    ghost_lidx lda, ghost_lidx ldb)
{
    ghost_error ret = GHOST_SUCCESS;

    const int targetBlockSize = 256;
    int deviceUsed;
    cudaGetDevice(&deviceUsed);
    cudaDeviceProp prop;
    ghost_cu_deviceprop_get(&prop);

    int numBlocks;

    if (N > M) {
        int const blockSize = (targetBlockSize / N) * N;
        CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
                      genv7_blockProductKernel<T, oT, conjv, M, N, blockSize, true, false>, blockSize, 0),
            ret);
    } else {
        int const blockSize = (targetBlockSize / M) * M;
        if (M == N && A == B) {
            CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
                          genv7_blockProductKernel<T, oT, conjv, M, N, blockSize, false, true>,
                          blockSize, 0),
                ret);
        } else {
            CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
                          genv7_blockProductKernel<T, oT, conjv, M, N, blockSize, false, false>,
                          blockSize, 0),
                ret);
        }
    }
    int blockCount = min( prop.multiProcessorCount * numBlocks, K*N / 10 / targetBlockSize + 1);


    size_t required_temp_storage_bytes = M * N * blockCount * sizeof(oT);
    ghost_cu_temp_buffer_malloc(&d_temp_storage, required_temp_storage_bytes);


    if (N > M) {
        int const blockSize = (targetBlockSize / N) * N;
        genv7_blockProductKernel<T, oT, conjv, N, M, blockSize, true, false>
            <<<blockCount, blockSize>>>(B, A, (oT *)d_temp_storage, K, ldb, lda, ldc);
    } else {
        int const blockSize = (targetBlockSize / M) * M;
        if (M == N && A == B) {
            genv7_blockProductKernel<T, oT, conjv, M, N, blockSize, false, true>
                <<<blockCount, blockSize>>>(A, B, (oT *)d_temp_storage, K, lda, ldb, ldc);
        } else {
            genv7_blockProductKernel<T, oT, conjv, M, N, blockSize, false, false>
                <<<blockCount, blockSize>>>(A, B, (oT *)d_temp_storage, K, lda, ldb, ldc);
        }
    }

    CUDA_CALL(cudaGetLastError(), ret);
    deviceReduce<oT, M, N>
        <<<(M * N) / 256 + 1, 256>>>((oT *)d_temp_storage, C, alpha, beta, blockCount, lda, ldb, ldc);
    CUDA_CALL(cudaGetLastError(), ret);
    ghost_cu_temp_buffer_free(d_temp_storage);
    return ret;
}
#endif
