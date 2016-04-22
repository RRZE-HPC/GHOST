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
#include <cublas_v2.h>
#include <typeinfo>
#include <iostream>

namespace {

void *temp_storage = NULL;
size_t temp_storage_bytes = 0;

namespace SPECSMALL {
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
    sum = accu(sum, blockResults[i * N * ldc + n * ldc + m]);
  }

  result[n * ldc + m] = axpby(sum, result[n * ldc + m], alpha, beta);
}

template <typename T, bool conjv, bool TRANSPOSE>
__device__ T condConj1(T v) {
  if (conjv && !TRANSPOSE) v = conj(v);
  return v;
}
template <typename T, bool conjv, bool TRANSPOSE>
__device__ T condConj2(T v) {
  if (conjv && TRANSPOSE) v = conj(v);
  return v;
}

template <typename T, bool conjv, int M, int N, int BLOCKSIZE, bool TRANSPOSE,
          bool SELF>
__global__ void blockProductKernel(const T *A, const T *B, T *out, const int K,
                                   const int lda, const int ldb,
                                   const int ldc) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  int warpLane = threadIdx.x % 32;
  int rowsPerWarp = 32 / M;
  int m = warpLane % M;

  if (warpLane >= rowsPerWarp * M) {
    warpLane = rowsPerWarp * M - 1;
    m = warpLane % M;
  }

  __shared__ T blockStorage[BLOCKSIZE];

  zero(blockStorage[threadIdx.x]);

  T threadSum[N];
  for (int n = 0; n < N; n++) {
    zero(threadSum[n]);
  }

  for (int idx = (tidx / 32) * rowsPerWarp + warpLane / M; idx < K;
       idx += blockDim.x * gridDim.x / 32 * rowsPerWarp) {
    T av = A[idx * lda + m];
    if (!SELF) {
      blockStorage[threadIdx.x] = B[idx * ldb + m];
    } else {
      blockStorage[threadIdx.x] = av;
    }
    int localAddress = threadIdx.x - m;
    for (int n = 0; n < N; n++) {
      threadSum[n] =
          axpy(threadSum[n], condConj1<T, conjv, TRANSPOSE>(av),
               condConj2<T, conjv, TRANSPOSE>(blockStorage[localAddress + n]));
    }
  }

  for (int n = 0; n < N; n++) {
    __syncthreads();
    blockStorage[threadIdx.x] = threadSum[n];
    __syncthreads();

    if (threadIdx.x < M) {
      T blockSum;
      zero(blockSum);
      for (int w = 0; w < BLOCKSIZE / 32; w++) {
        for (int wp = threadIdx.x; wp < rowsPerWarp * M; wp += M) {
          blockSum = accu(blockSum, blockStorage[w * 32 + wp]);
        }
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


cublasHandle_t handle;
}

template <typename T, int M, int N, int conjv>
static void ghost_tsmttsm_cu_rm(T *const __restrict__ C,
                                const T *const __restrict__ A,
                                const T *const __restrict__ B, const T alpha,
                                const T beta, ghost_lidx K, ghost_lidx ldc,
                                ghost_lidx lda, ghost_lidx ldb) {
  if (M > 32 || N > 32) {
    if (temp_storage == NULL && temp_storage_bytes == 0) {
      temp_storage_bytes = 8;
      cublasCreate(&handle);
    }

    cublasStatus_t status;
    cublasOperation_t op = (conjv == 1) ? CUBLAS_OP_C : CUBLAS_OP_T;
    if (typeid(T) == typeid(double)) {
      status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K,
                           (double *)&alpha, (double *)A, lda, (double *)B, ldb,
                           (double *)&beta, (double *)C, ldc);
    } else if (typeid(T) == typeid(float)) {
      status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K,
                           (float *)&alpha, (float *)A, lda, (float *)B, ldb,
                           (float *)&beta, (float *)C, ldc);
    } else if (typeid(T) == typeid(cuDoubleComplex)) {
      status = cublasZgemm(handle, CUBLAS_OP_N, op, M, N, K,
                           (cuDoubleComplex *)&alpha, (cuDoubleComplex *)A, lda,
                           (cuDoubleComplex *)B, ldb, (cuDoubleComplex *)&beta,
                           (cuDoubleComplex *)C, ldc);
    } else if (typeid(T) == typeid(cuFloatComplex)) {
      status = cublasCgemm(handle, CUBLAS_OP_N, op, M, N, K,
                           (cuFloatComplex *)&alpha, (cuFloatComplex *)A, lda,
                           (cuFloatComplex *)B, ldb, (cuFloatComplex *)&beta,
                           (cuFloatComplex *)C, ldc);
    }
    if (status != CUBLAS_STATUS_SUCCESS) std::cerr << "cublasXgemm error\n";
    return;
  }

  const int threadsPerBlock = 256;
  int deviceUsed;
  cudaGetDevice(&deviceUsed);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, deviceUsed);
  int numBlocks;

  const void *func = NULL;

  if (N > M) {
    func = SPECSMALL::blockProductKernel<T, conjv, M, N, threadsPerBlock, true,
                                         false>;
  } else {
    if (M == N && A == B) {
      func = SPECSMALL::blockProductKernel<T, conjv, M, N, threadsPerBlock,
                                           false, true>;
    } else {
      func = SPECSMALL::blockProductKernel<T, conjv, M, N, threadsPerBlock,
                                           false, false>;
    }
  }
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, func,
                                                threadsPerBlock, 0);

  int blockCount = prop.multiProcessorCount * numBlocks;

  if (temp_storage_bytes == 0 || temp_storage == NULL) {
    temp_storage_bytes = blockCount * sizeof(T) * N * ldc;
    ghost_cu_malloc(&temp_storage, temp_storage_bytes);
    if (temp_storage == NULL) temp_storage_bytes = 0;
  }
  if (N > M) {
    SPECSMALL::blockProductKernel<T, conjv, N, M, threadsPerBlock, true,
                                  false><<<blockCount, threadsPerBlock>>>(
        B, A, (T *)temp_storage, K, ldb, lda, ldc);
  } else {
    if (M == N && A == B) {
      SPECSMALL::blockProductKernel<T, conjv, M, N, threadsPerBlock, false,
                                    true><<<blockCount, threadsPerBlock>>>(
          A, B, (T *)temp_storage, K, lda, ldb, ldc);
    } else {
      SPECSMALL::blockProductKernel<T, conjv, M, N, threadsPerBlock, false,
                                    false><<<blockCount, threadsPerBlock>>>(
          A, B, (T *)temp_storage, K, lda, ldb, ldc);
    }
  }

  SPECSMALL::deviceReduce<T, M, N><<<M * N / 256 + 1, 256>>>(
      (T *)temp_storage, C, alpha, beta, blockCount, lda, ldb, ldc);

  ghost_cu_free(temp_storage);
  temp_storage = NULL;
  temp_storage_bytes = 0;
}

#endif
