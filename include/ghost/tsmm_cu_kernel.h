/**
 * @file tsmm_cu_kernel.h
 * @brief TSMM CUDA kernels.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 * @author Dominik Ernst <dominik.ernst@fau.de>
 */
#ifndef GHOST_TSMM_CU_KERNEL_H
#define GHOST_TSMM_CU_KERNEL_H

#include <cublas_v2.h>
#include <iostream>
#include <typeinfo>
#include "ghost/config.h"
#include "ghost/cu_complex.h"
#include "ghost/cu_util.h"
#include "ghost/types.h"

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

template<typename T, int M, int N, int BLOCKSIZE, bool BETAISZERO>
static __global__ void tsmm_fallback_kernel(const T *__restrict__ A, const T *__restrict__ B, T *out, const int K, const int lda, const int ldb, const int ldc, T alpha, T beta)
{
    int tidx = blockIdx.x * BLOCKSIZE + threadIdx.x;
    int n = tidx % N;

    if (tidx / N == gridDim.x * BLOCKSIZE / N && !BETAISZERO)
        return;

    for (int row = tidx / N; row < K; row += gridDim.x * BLOCKSIZE / N) {
        T sum;
        zero(sum);
        for (int m = 0; m < M; m++) {
            sum = axpy(sum, A[row * lda + m], B[n * ldb + m]);
        }
        if (BETAISZERO) {
            out[row * ldc + n] = scale(alpha, sum);
        } else {
            out[row * ldc + n] = axpby(sum, out[row * ldc + n], alpha, beta);
        }
    }
}

template<typename T, int M, int N, int BLOCKSIZE, bool BETAISZERO>
static __global__ void tsmm_v1_kernel(const T *A, const T *__restrict__ B, T *out, const int K, const int lda, const int ldb, const int ldc, T alpha, T beta)
{
    int warpLane = threadIdx.x % 32;
    const int rowsPerWarp = 32 / N;
    const int n = warpLane % N;

    if (warpLane >= rowsPerWarp * N) {
        warpLane = rowsPerWarp * N - 1;
    }
    const int localRow = threadIdx.x / 32 * rowsPerWarp + warpLane / N;

    T __shared__ rowCache[(M / N <= 16) ? M * BLOCKSIZE / 32 * rowsPerWarp : 1];

    for (int row = blockIdx.x * BLOCKSIZE / 32 * rowsPerWarp + localRow; row < K;
         row += BLOCKSIZE * gridDim.x / 32 * rowsPerWarp) {
        for (int i = 0; i < M / N; i++) {
            rowCache[localRow * M + n + i * N] = A[row * lda + n + i * N];
        }

        T sum;
        zero(sum);
        for (int m = 0; m < M; m++) {
            sum = axpy(sum, rowCache[localRow * M + m], B[n * ldb + m]);
        }
        if (BETAISZERO) {
            out[row * ldc + n] = scale(alpha, sum);
        } else {
            out[row * ldc + n] = axpby(sum, out[row * ldc + n], alpha, beta);
        }
    }
}

template<typename T>
__device__ inline T __shfl_xor_t(T var, unsigned int srcLane, int width = 32)
{
    int *a = reinterpret_cast<int *>(&var);
    for (int i = 0; i < sizeof(T) / 4; i++) {
#if __CUDACC_VER_MAJOR__ < 9
    a[i] = __shfl_xor(a[i], srcLane, width);
#else
    a[i] = __shfl_xor_sync(0xFFFFFFFF, a[i], srcLane, width);
#endif
    }
    return *reinterpret_cast<T *>(a);
}

template<typename T>
__device__ inline T warpReduce(T lval, int width)
{
    for (int offset = width / 2; offset > 0; offset /= 2) {
        lval = accu(lval, __shfl_xor_t(lval, offset, width));
    }
    return lval;
}

template<typename T, int M, int N, int BLOCKSIZE, bool BETAISZERO>
static __global__ void tsmm_v2_kernel(const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ out, const int K, const int lda, const int ldb, const int ldc, T alpha, T beta)
{
    const int GANGSIZE = 32 / sizeof(T);

    int gId = threadIdx.x % GANGSIZE;
    int tidx = blockIdx.x * BLOCKSIZE + threadIdx.x;

    for (int row = tidx / GANGSIZE; row < K;
         row += gridDim.x * BLOCKSIZE / GANGSIZE) {
        for (int n = 0; n < N; n++) {
            T gval;
            zero(gval);
            for (int i = 0; i < (M - 1) / GANGSIZE + 1; i++) {
                int m = i * GANGSIZE + gId;
                if (m < M || M % GANGSIZE == 0)
                    gval = axpy(gval, A[row * lda + m], B[n * ldb + m]);
            }
            if (BETAISZERO) {
                out[row * ldc + n] = scale(alpha, warpReduce(gval, GANGSIZE));
            } else {
                out[row * ldc + n] =
                    axpby(warpReduce(gval, GANGSIZE), out[row * ldc + n], alpha, beta);
            }
        }
    }
}
template<typename T, int M, int N>
bool tsmm_fallback(const int K, const T alpha, const T *A, const int lda, const T *B, const int ldb, const T beta, T *C, const int ldc)
{
    const int threadsPerBlock = 256;
    int deviceUsed;
    cudaGetDevice(&deviceUsed);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceUsed);
    int numBlocks;
    ghost_error ret = GHOST_SUCCESS;

    CUDA_CALL(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocks, tsmm_fallback_kernel<T, M, N, threadsPerBlock, false>,
            threadsPerBlock, 0),
        ret);
    int blockCount = prop.multiProcessorCount * numBlocks;
    T Tzero;
    zero(Tzero);
    if (eq(beta, Tzero)) {
        tsmm_fallback_kernel<T, M, N, threadsPerBlock,
            true><<<blockCount, threadsPerBlock>>>(
            A, B, C, K, lda, ldb, ldc, alpha, beta);
    } else {
        tsmm_fallback_kernel<T, M, N, threadsPerBlock,
            false><<<blockCount, threadsPerBlock>>>(
            A, B, C, K, lda, ldb, ldc, alpha, beta);
    }
    CUDA_CALL(cudaGetLastError(), ret);
    if (ret != GHOST_SUCCESS)
        return false;
    return true;
}

template<typename T, int M, int N>
bool tsmm_v1(const int K, const T alpha, const T *A, const int lda, const T *B, const int ldb, const T beta, T *C, const int ldc)
{
    if (M % N == 0 && M / N <= 16) {
        const int threadsPerBlock = 256;
        int deviceUsed;
        cudaGetDevice(&deviceUsed);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceUsed);
        int numBlocks;
        ghost_error ret = GHOST_SUCCESS;

        CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                      &numBlocks, tsmm_v1_kernel<T, M, N, threadsPerBlock, false>,
                      threadsPerBlock, 0),
            ret);
        int blockCount = prop.multiProcessorCount * numBlocks;

        T Tzero;
        zero(Tzero);
        if (eq(beta, Tzero)) {
            tsmm_v1_kernel<T, M, N, threadsPerBlock,
                true><<<blockCount, threadsPerBlock>>>(
                A, B, C, K, lda, ldb, ldc, alpha, beta);
        } else {
            tsmm_v1_kernel<T, M, N, threadsPerBlock,
                false><<<blockCount, threadsPerBlock>>>(
                A, B, C, K, lda, ldb, ldc, alpha, beta);
        }
        CUDA_CALL(cudaGetLastError(), ret);
        if (ret != GHOST_SUCCESS)
            return false;
        return true;
    } else {
        return false;
    }
}

template<typename T, int M, int N>
bool tsmm_v2(const int K, const T alpha, const T *A, const int lda, const T *B, const int ldb, const T beta, T *C, const int ldc)
{
    if (M >= 32 / sizeof(T)) {
        const int threadsPerBlock = 256;
        int deviceUsed;
        cudaGetDevice(&deviceUsed);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceUsed);
        int numBlocks;
        ghost_error ret = GHOST_SUCCESS;

        CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                      &numBlocks, tsmm_v2_kernel<T, M, N, threadsPerBlock, false>,
                      threadsPerBlock, 0),
            ret);
        int blockCount = prop.multiProcessorCount * numBlocks;
        T Tzero;
        zero(Tzero);
        if (eq(beta, Tzero)) {
            tsmm_v2_kernel<T, M, N, threadsPerBlock,
                true><<<blockCount, threadsPerBlock>>>(
                A, B, C, K, lda, ldb, ldc, alpha, beta);
        } else {
            tsmm_v2_kernel<T, M, N, threadsPerBlock,
                false><<<blockCount, threadsPerBlock>>>(
                A, B, C, K, lda, ldb, ldc, alpha, beta);
        }
        CUDA_CALL(cudaGetLastError(), ret);
        if (ret != GHOST_SUCCESS)
            return false;
        return true;
    } else {
        return false;
    }
}
template<typename T, int M, int N>
bool tsmm_cublas(const int K, const T alpha, const T *A, const int lda, const T *B, const int ldb, const T beta, T *C, const int ldc)
{
    cublasHandle_t cublas_handle;
    ghost_cu_cublas_handle(&cublas_handle);

    ghost_error ret = GHOST_SUCCESS;

    if (typeid(T) == typeid(double)) {
        CUBLAS_CALL(cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, N, K, M,
                        (double *)&alpha, (double *)B, ldb, (double *)A,
                        lda, (double *)&beta, (double *)C, ldc),
            ret);
    } else if (typeid(T) == typeid(float)) {
        CUBLAS_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, K, M,
                        (float *)&alpha, (float *)B, ldb, (float *)A, lda,
                        (float *)&beta, (float *)C, ldc),
            ret);
    } else if (typeid(T) == typeid(cuDoubleComplex)) {
        CUBLAS_CALL(
            cublasZgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, K, M,
                (cuDoubleComplex *)&alpha, (cuDoubleComplex *)B, ldb,
                (cuDoubleComplex *)A, lda, (cuDoubleComplex *)&beta,
                (cuDoubleComplex *)C, ldc),
            ret);
    } else if (typeid(T) == typeid(cuComplex)) {
        CUBLAS_CALL(
            cublasCgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, K, M,
                (cuComplex *)&alpha, (cuComplex *)B, ldb, (cuComplex *)A,
                lda, (cuComplex *)&beta, (cuComplex *)C, ldc),
            ret);
    }
    if (ret != GHOST_SUCCESS)
        return false;
    return true;
}
}

template<typename T, int M, int N>
bool ghost_tsmm_cu_rm_cm(T *C, const T *A, const T *B, const T alpha, const T beta, const ghost_lidx K, const ghost_lidx ldc, const ghost_lidx lda, const ghost_lidx ldb)
{
    if (M >= 7 && N >= 4 && tsmm_v1<T, M, N>(K, alpha, A, lda, B, ldb, beta, C, ldc))
        return true;
    if (M >= 14 && N >= 14 && tsmm_cublas<T, M, N>(K, alpha, A, lda, B, ldb, beta, C, ldc))
        return true;
    if (M >= 7 && N <= 5 && tsmm_v2<T, M, N>(K, alpha, A, lda, B, ldb, beta, C, ldc))
        return true;
    if (tsmm_fallback<T, M, N>(K, alpha, A, lda, B, ldb, beta, C, ldc))
        return true;
    return tsmm_cublas<T, M, N>(K, alpha, A, lda, B, ldb, beta, C, ldc);
}

#endif
