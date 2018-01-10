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

namespace {
void *d_temp_storage = NULL;
size_t temp_storage_size = 0;
template<typename T, typename iT, int M, int N>
__global__ void deviceReduce(
    iT *blockResults, T *result, T alpha, T beta, int blockCount, int lda, int ldb, int ldc)
{
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= M * N) return;
    int n = tidx / M;
    int m = tidx % M;

    iT sum;
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

template<typename T, typename iT, int conjv, int M, int N, int BLOCKSIZE, bool TRANSPOSE, bool SELF>
__global__ void __launch_bounds__(BLOCKSIZE) genv7_blockProductKernel(
    const T *A, const T *B, iT *out, const int K, const int lda, const int ldb, const int ldc)
{
    const int rowsPerBlock = BLOCKSIZE / M;
    int m = threadIdx.x % M;
    int localRow = threadIdx.x / M;
    int bOffset = localRow * ldb + m;
    int aOffset = localRow * lda + m;
    if (m >= N) bOffset = localRow * ldb + 0;
    if (bOffset >= rowsPerBlock * ldb) bOffset = 0;
    if (aOffset >= rowsPerBlock * lda) aOffset = 0;

    __shared__ iT blockStorage[rowsPerBlock * M * (sizeof(T) > sizeof(iT) ? 2 : 1)];
    T *rowCache = reinterpret_cast<T *>(blockStorage);

    zero(blockStorage[threadIdx.x]);
    __syncthreads();

    iT threadSum[N];
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

        //    if (!SELF) {
        bvNext = __ldg(B + idxNext * ldb + bOffset);
        //} else {
        // bvNext = avNext;
        //}
        __syncthreads();
        rowCache[threadIdx.x] = bvNow;
        __syncthreads();

        int localAddress = threadIdx.x - m;
        for (int n = 0; n < N; n++) {
            threadSum[n] = axpy(threadSum[n], condConj1<T, conjv, TRANSPOSE>(avNow),
                condConj2<T, conjv, TRANSPOSE>(rowCache[localAddress + n]));
        }
        avNow = avNext;
        bvNow = bvNext;
    }

    // Remainder loop
    for (idx = idx + localRow; idx < K; idx += gridDim.x * rowsPerBlock) {
        T av = A[idx * lda + m];
        for (int n = 0; n < N; n++) {
            threadSum[n] = axpy(threadSum[n], condConj1<T, conjv, TRANSPOSE>(av),
                condConj2<T, conjv, TRANSPOSE>(B[idx * ldb + n]));
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
}


template<typename T, int M, int N, int conjv>
static ghost_error ghost_tsmttsm_cu_rm(T *const __restrict__ C, const T *const __restrict__ A,
    const T *const __restrict__ B, const T alpha, const T beta, ghost_lidx K, ghost_lidx ldc,
    ghost_lidx lda, ghost_lidx ldb)
{
    ghost_error ret = GHOST_SUCCESS;


    if (M > 64 || N > 64) {
        cublasHandle_t handle;
        ghost_cu_cublas_handle(&handle);
        cublasOperation_t op = (conjv == 1) ? CUBLAS_OP_C : CUBLAS_OP_T;
        if (typeid(T) == typeid(double)) {
            CUBLAS_CALL(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K, (double *)&alpha,
                            (double *)A, lda, (double *)B, ldb, (double *)&beta, (double *)C, ldc),
                ret);
        } else if (typeid(T) == typeid(float)) {
            CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K, (float *)&alpha,
                            (float *)A, lda, (float *)B, ldb, (float *)&beta, (float *)C, ldc),
                ret);
        } else if (typeid(T) == typeid(cuDoubleComplex)) {
            CUBLAS_CALL(cublasZgemm(handle, CUBLAS_OP_N, op, M, N, K, (cuDoubleComplex *)&alpha,
                            (cuDoubleComplex *)A, lda, (cuDoubleComplex *)B, ldb,
                            (cuDoubleComplex *)&beta, (cuDoubleComplex *)C, ldc),
                ret);
        } else if (typeid(T) == typeid(cuFloatComplex)) {
            CUBLAS_CALL(cublasCgemm(handle, CUBLAS_OP_N, op, M, N, K, (cuFloatComplex *)&alpha,
                            (cuFloatComplex *)A, lda, (cuFloatComplex *)B, ldb,
                            (cuFloatComplex *)&beta, (cuFloatComplex *)C, ldc),
                ret);
        }
        return ret;
    }


    const int targetBlockSize = 256;
    int deviceUsed;
    cudaGetDevice(&deviceUsed);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceUsed);
    int numBlocks;

    if (N > M) {
        int const blockSize = (targetBlockSize / N) * N;
        CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
                      genv7_blockProductKernel<T, T, conjv, M, N, blockSize, true, false>, blockSize, 0),
            ret);
    } else {
        int const blockSize = (targetBlockSize / M) * M;
        if (M == N && A == B) {
            CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
                          genv7_blockProductKernel<T, T, conjv, M, N, blockSize, false, true>,
                          blockSize, 0),
                ret);
        } else {
            CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
                          genv7_blockProductKernel<T, T, conjv, M, N, blockSize, false, false>,
                          blockSize, 0),
                ret);
        }
    }
    int blockCount = prop.multiProcessorCount * numBlocks;

    // CUDA_CALL(cudaMalloc(&d_temp_storage, 100 * 100 * 1000 * sizeof(T)), ret);

    size_t required_temp_storage_size = M * N * blockCount;
    //    if (temp_storage_size < required_temp_storage_size) {
    // CUDA_CALL(cudaFree(d_temp_storage), ret);
    temp_storage_size = required_temp_storage_size;
    CUDA_CALL(cudaMalloc(&d_temp_storage, sizeof(T) * temp_storage_size), ret);
    //}

    /*    size_t required_temp_storage_bytes = blockCount * sizeof(T) * N * ldc;
    if (temp_storage_bytes < required_temp_storage_bytes || temp_storage == NULL) {
        temp_storage_bytes = required_temp_storage_bytes;
        ghost_cu_malloc(&temp_storage, temp_storage_bytes);
        if (temp_storage == NULL) temp_storage_bytes = 0;
        }*/


    if (N > M) {
        int const blockSize = (targetBlockSize / N) * N;
        genv7_blockProductKernel<T, T, conjv, N, M, blockSize, true, false>
            <<<blockCount, blockSize>>>(B, A, (T *)d_temp_storage, K, ldb, lda, ldc);
    } else {
        int const blockSize = (targetBlockSize / M) * M;
        if (M == N && A == B) {
            genv7_blockProductKernel<T, T, conjv, M, N, blockSize, false, true>
                <<<blockCount, blockSize>>>(A, B, (T *)d_temp_storage, K, lda, ldb, ldc);
        } else {
            genv7_blockProductKernel<T, T, conjv, M, N, blockSize, false, false>
                <<<blockCount, blockSize>>>(A, B, (T *)d_temp_storage, K, lda, ldb, ldc);
        }
    }
    CUDA_CALL(cudaDeviceSynchronize(), ret);


    CUDA_CALL(cudaGetLastError(), ret);
    deviceReduce<T, T, M, N>
        <<<(M * N) / 256 + 1, 256>>>((T *)d_temp_storage, C, alpha, beta, blockCount, lda, ldb, ldc);
    CUDA_CALL(cudaGetLastError(), ret);

    CUDA_CALL(cudaFree(d_temp_storage), ret);
    return ret;
}
#endif
