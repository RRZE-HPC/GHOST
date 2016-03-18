/**
 * @file tsmttsm_cu_kernel.h
 * @brief TSMTTSM CUDA kernels.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 * @author Dominik Ernst <dominik.ernst@fau.de>
 */
#ifndef GHOST_TSMTTSM_CU_KERNEL_H
#define GHOST_TSMTTSM_CU_KERNEL_H

#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/cu_complex.h"

namespace {

void *temp_storage = NULL;
size_t temp_storage_bytes = 0;

namespace GENV3 {

template <typename T, int M, int N>
__global__ void deviceReduce(T *blockResults, T *result, T alpha, T beta,
                             int blockCount, size_t lda, size_t ldb,
                             size_t ldc) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx >= M * N) return;

  int n = tidx / M;
  int m = tidx % M;

  T sum;
  zero(sum);
  for (int i = 0; i < blockCount; i++) {
    sum = accu(sum,blockResults[i * N * ldc + n * ldc + m]);
  }

  result[n * ldc + m] = axpby(sum, result[n * ldc + m], alpha, beta);
}

template <typename T, bool conjv, int M, int N, int BLOCKSIZE, bool TRANSPOSE>
__global__ void blockProductKernel(const T *A, const T *B, T *out, size_t K,
                                   size_t lda, size_t ldb, size_t ldc) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ T blockStorage[BLOCKSIZE];

  zero(blockStorage[threadIdx.x]);

  int m = tidx % M;

  if (blockDim.x * gridDim.x / M == tidx / M) return;

  T threadSum[N];
  for (int n = 0; n < N; n++) {
    zero(threadSum[n]);
  }

  if (conjv) {
      if (TRANSPOSE) {
          for (size_t idx = tidx / M; idx < K; idx += blockDim.x * gridDim.x / M) {
            for (int n = 0; n < N; n++) {
              threadSum[n] = accu(threadSum[n], mulConj(B[idx * ldb + n],A[idx * lda + m]));
            }
          }
      } else {
          for (size_t idx = tidx / M; idx < K; idx += blockDim.x * gridDim.x / M) {
            for (int n = 0; n < N; n++) {
              threadSum[n] = accu(threadSum[n], mulConj(A[idx * ldb + n],B[idx * lda + m]));
            }
          }
      }
  } else {
      for (size_t idx = tidx / M; idx < K; idx += blockDim.x * gridDim.x / M) {
        for (int n = 0; n < N; n++) {
          threadSum[n] = axpy(threadSum[n], A[idx * lda + m], B[idx * ldb + n]);
        }
      }
  }

  for (int n = 0; n < N; n++) {
    __syncthreads();
    blockStorage[threadIdx.x] = threadSum[n];
    __syncthreads();

    if (threadIdx.x < M) {
      T blockSum;
      zero(blockSum);
      for (int i = threadIdx.x; i < BLOCKSIZE; i += M) {
        blockSum = accu(blockSum,blockStorage[i]);
      }
      if (TRANSPOSE) {
        out[blockIdx.x * M * ldc + m * ldc + n] = blockSum;
      } else {
        out[blockIdx.x * N * ldc + n * ldc + m] = blockSum;
      }
    }
  }
}
}
}

template <typename T, int M, int N, int conjv>
static void ghost_tsmttsm_cu_rm(T* const __restrict__ C,
                                   const T* const __restrict__ A,
                                   const T* const __restrict__ B, const T alpha,
                                   const T beta, ghost_lidx K,
                                   ghost_lidx ldc, ghost_lidx lda,
                                   ghost_lidx ldb) {

  const int threadsPerBlock = 256;
  int deviceUsed;
  cudaGetDevice(&deviceUsed);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, deviceUsed);
  int numBlocks;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocks, GENV3::blockProductKernel<T, conjv, N, M, threadsPerBlock, true>,
      threadsPerBlock, 0);
  int blockCount = prop.multiProcessorCount * numBlocks;

  if (temp_storage_bytes == 0 || temp_storage == NULL) {
    temp_storage_bytes = blockCount * sizeof(T) * N * ldc;
    ghost_cu_malloc(&temp_storage, temp_storage_bytes);
    if (temp_storage == NULL) temp_storage_bytes = 0;
  }
  if (N > M) {
    GENV3::blockProductKernel<T, conjv, N, M, threadsPerBlock,
                              true><<<blockCount, threadsPerBlock>>>(
        B, A, (T *)temp_storage, K, ldb, lda, ldc);
  } else {
    GENV3::blockProductKernel<T, conjv, M, N, threadsPerBlock,
                              false><<<blockCount, threadsPerBlock>>>(
        A, B, (T *)temp_storage, K, lda, ldb, ldc);
  }

  GENV3::deviceReduce<T, M, N><<<M * N / 256 + 1, 256>>>(
      (T *)temp_storage, C, alpha, beta, blockCount, lda, ldb, ldc);
  
  ghost_cu_free(temp_storage);
  temp_storage = NULL;
  temp_storage_bytes = 0;
}

#endif
