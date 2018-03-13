#include <iostream>
#include "ghost/types.h"
#include "ghost/densemat.h"
#include "ghost/sparsemat.h"
#include "ghost/cu_complex.h"

#include "ghost/tsmtspmtsm_var2_cuda.h"

using namespace std;

namespace {


template<typename oT>
__global__ void deviceReduce(oT *blockResults, oT *result, oT alpha, oT beta, int blockCount,
    ghost_lidx M, ghost_lidx N, ghost_lidx ldx)
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

    result[n * ldx + m] = accu(scale(result[n * ldx + m], beta), scale2(sum, alpha));
}


template<typename T, typename oT, int BLOCKSIZE>
__global__ void tsmtspmtsm_kernel(oT *out, T *wval, T *vval, T *Aval, ghost_lidx *chunkStart,
    ghost_lidx *chunkLen, ghost_lidx *col, ghost_lidx nchunks, int C, ghost_lidx M, ghost_lidx N,
    ghost_lidx K, ghost_lidx ldv, ghost_lidx ldw)
{
    __shared__ oT blockStorage[BLOCKSIZE];
    zero(blockStorage[threadIdx.x]);
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int n = tidx % N;
    int m = (tidx / N) % M;

    if (blockDim.x * gridDim.x / N / M == tidx / N / M) return;


    oT threadSum;
    zero(threadSum);
    for (int chunk = tidx / N / M; chunk < nchunks; chunk += blockDim.x * gridDim.x / N / M) {
        for (int c = 0; c < C; c++) {
            oT wsum;
            zero(wsum);

            for (int j = 0; j < chunkLen[chunk]; j++) {
                ghost_gidx idx = chunkStart[chunk] + j * C + c;
                wsum = axpy(wsum, (oT)Aval[idx], (oT)wval[col[idx] * ldw + n]);
            }
            if (chunk * C + c < K) {
                threadSum = axpy(threadSum, wsum, (oT)vval[(chunk * C + c) * ldv + m]);
            }
        }
    }

    __syncthreads();
    blockStorage[threadIdx.x] = threadSum;
    __syncthreads();

    if (threadIdx.x < M * N) {
        oT blockSum;
        zero(blockSum);
        for (int i = threadIdx.x; i < BLOCKSIZE; i += M * N) {
          blockSum = accu(blockSum, blockStorage[i]);
        }

        out[blockIdx.x * N * M + n * M + m] = blockSum;
    }
}

template<typename T, typename oT>
ghost_error typed_tsmtspmtsm(ghost_densemat *x, ghost_densemat *v, ghost_densemat *w,
    ghost_sparsemat *A, void *pAlpha, void *pBeta)
{
    ghost_error ret;
    oT alpha = *(oT *)pAlpha;
    oT beta = *(oT *)pBeta;

    const ghost_lidx ldv = v->stride;
    const ghost_lidx ldw = w->stride;
    const ghost_lidx ldx = x->stride;

    ghost_lidx N = w->traits.ncols;
    ghost_lidx M = v->traits.ncols;


    int deviceUsed;
    cudaGetDevice(&deviceUsed);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceUsed);
    int numBlocks;

    int const blockSize = 256;
    CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &numBlocks, tsmtspmtsm_kernel<T, oT, blockSize>, blockSize, 0),
        ret);
    int blockCount = prop.multiProcessorCount * numBlocks;

    void *d_temp_storage = NULL;
    size_t required_temp_storage_bytes = M * N * blockCount * sizeof(oT);
    CUDA_CALL(cudaMalloc(&d_temp_storage, required_temp_storage_bytes), ret);
    CUDA_CALL(cudaMemset(d_temp_storage, 0, required_temp_storage_bytes), ret);

    tsmtspmtsm_kernel<T, oT, blockSize><<<blockCount, blockSize>>>((oT *)d_temp_storage,
        (T *)w->cu_val, (T *)v->cu_val, (T *)A->cu_val, A->cu_chunkStart, A->cu_chunkLen, A->cu_col,
        SPM_NCHUNKS(A), A->traits.C, M, N, SPM_NROWS(A), ldv, ldw);

    deviceReduce<oT><<<M * N / 256 + 1, 256>>>(
        (oT *)d_temp_storage, (oT *)x->cu_val, alpha, beta, blockCount, M, N, ldx);

    CUDA_CALL(cudaFree(d_temp_storage), ret);
    return ret;
}
}


ghost_error tsmtspmtsm_var2_cuda(ghost_densemat *x, ghost_densemat *v, ghost_densemat *w,
    ghost_sparsemat *A, void *pAlpha, void *pBeta)
{
    if (x->traits.datatype & GHOST_DT_REAL && w->traits.datatype & GHOST_DT_REAL) {
        if (x->traits.datatype & GHOST_DT_DOUBLE && w->traits.datatype & GHOST_DT_DOUBLE) {
            return typed_tsmtspmtsm<double, double>(x, v, w, A, pAlpha, pBeta);
        }
        if (x->traits.datatype & GHOST_DT_FLOAT && w->traits.datatype & GHOST_DT_FLOAT) {
            return typed_tsmtspmtsm<float, float>(x, v, w, A, pAlpha, pBeta);
        }
        if (x->traits.datatype & GHOST_DT_DOUBLE && w->traits.datatype & GHOST_DT_FLOAT) {
            return typed_tsmtspmtsm<float, double>(x, v, w, A, pAlpha, pBeta);
        }
    }
    if (x->traits.datatype & GHOST_DT_COMPLEX && w->traits.datatype & GHOST_DT_COMPLEX) {
        if (x->traits.datatype & GHOST_DT_DOUBLE && w->traits.datatype & GHOST_DT_DOUBLE) {
            return typed_tsmtspmtsm<cuDoubleComplex, cuDoubleComplex>(x, v, w, A, pAlpha, pBeta);
        }
        if (x->traits.datatype & GHOST_DT_FLOAT && w->traits.datatype & GHOST_DT_FLOAT) {
            return typed_tsmtspmtsm<cuFloatComplex, cuFloatComplex>(x, v, w, A, pAlpha, pBeta);
        }
    }
    return GHOST_SUCCESS;
}
