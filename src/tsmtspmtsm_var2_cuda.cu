#include <iostream>
#include "ghost/types.h"
#include "ghost/densemat.h"
#include "ghost/sparsemat.h"

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

    result[n * ldc + m] = accu(scale(result[n * ldc + m], beta), scale2(sum, alpha));
}


template<typename T, typename oT, int BLOCKSIZE>
__global__ void tsmtspmtsm_kernel(oT *out, T *wval, T *vval, T *Aval, ghost_lidx *chunkStart,
    ghost_lidx *chunkLen, ghost_lidx *col, ghost_lidx nchunks, int C, ghost_lidx M, ghost_lidx N,
    ghost_lidx K, ghost_lidx ldv, ghost_lidx ldw)
{
    __shared__ oT blockStorage[BLOCKSIZE];
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int n = tidx % N;

    if (blockDim.x * gridDim.x / N == tidx / N) return;


    oT threadSum[M];
    for (int chunk = tidx / N; chunk < nchunks; chunk += blockDim.x * gridDim.x / N) {
        for (int c = 0; c < C; c++) {
            oT wsum;
            zero(wsum);

            for (int j = 0; j < chunkLen[chunk]; j++) {
                ghost_gidx idx = chunkStart[chunk] + j * C + c;
                wsum += (oT)Aval[idx] * (oT)wval[A->col[idx] * ldw + n];
            }
            if (chunk * C + c < K) {
                for (int m = 0; m < M; m++) {
                    threadSum[m] += wsum * (oT)vval[(chunk * C + c) * ldv + m];
                }
            }
        }
    }

    for (int m = 0; m < M; m++) {
        __syncthreads();
        blockStorage[threadIdx.x] = threadSum[m];
        __syncthreads();

        if (threadIdx.x < N) {
            oT blockSum;
            zero(blockSum);
            for (int i = threadIdx.x; i < BLOCKSIZE; i += N) {
                blockSum += blockStorage[i];
            }
            out[blockIdx.x * N * M + n * M + m] = blockSum;
        }
    }
}

template<typename T, typename oT>
ghost_error typed_tsmtspmtsm(ghost_densemat *v, ghost_densemat *w, ghost_densemat *x,
    ghost_sparsemat *A, void *pAlpha, void *pBeta)
{

    T *wval = (T *)w->val;
    T *vval = (T *)v->val;
    oT *xval = (oT *)x->val;
    oT alpha = *(oT *)pAlpha;
    oT beta = *(oT *)pBeta;
    ghost_lidx C = A->traits.C;
    const ghost_lidx ldv = v->stride;
    const ghost_lidx ldw = w->stride;
    const ghost_lidx ldx = x->stride;

    ghost_lidx N = w->traits.ncols;
    ghost_lidx M = v->traits.ncols;
    int nchunks = SPM_NCHUNKS(A);

    void *d_temp_storage = NULL;
    size_t required_temp_storage_bytes = M * N * blockCount * sizeof(oT);
    GPU_ERROR(cudaMalloc(&d_temp_storage, required_temp_storage_bytes));


    tsmtspmtsm_kernel<T, oT>((oT *)d_temp_storage, wval, vval, (T *)A->cu_val, A->cu_chunkStart,
        A->cu_chunkLen, A->cu_col, A->nchunks, A->traits.C, M, N, SPM_NROWS(A), ldv, ldw);

    deviceReduce<oT>
        <<<M * N / 256 + 1, 256>>>((oT *)d_temp_storage, xval, alpha, beta, blockCount, M, N, ldx);

    GPU_ERROR(cudaFree(d_temp_storage));
    return GHOST_SUCCESS;
}
}


ghost_error tsmtspmtsm_var2_cuda(ghost_densemat *v, ghost_densemat *w, ghost_densemat *x,
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
    return GHOST_SUCCESS;
}
