#include <iostream>
#include "ghost/types.h"
#include "ghost/densemat.h"
#include "ghost/sparsemat.h"

#include "ghost/tsmtspmtsm_var2_plain.h"
#include <iostream>
#include <vector>
#include <complex>

using namespace std;

namespace {
template<typename T, typename oT>
ghost_error tsmtspmtsm_var2_plain_kernel(ghost_densemat *x, ghost_densemat *v, ghost_densemat *w,
    ghost_sparsemat *A, void *pAlpha, void *pBeta)
{
    T *wval = (T *)w->val;
    T *vval = (T *)v->val;
    oT *xval = (oT *)x->val;
    T *Aval = (T *)A->val;
    oT alpha = *(oT *)pAlpha;
    oT beta = *(oT *)pBeta;
    ghost_lidx C = A->traits.C;


    const ghost_lidx ldv = v->stride;
    const ghost_lidx ldw = w->stride;
    const ghost_lidx ldx = x->stride;

    ghost_lidx N = w->traits.ncols;
    ghost_lidx M = v->traits.ncols;


    vector<oT> wsums(N);
    vector<oT> result(N * M);
    for (int chunk = 0; chunk < A->nchunks; chunk++) {
        for (int c = 0; c < C; c++) {
            for (int n = 0; n < N; n++) {
                wsums[n] = 0;
            }
            for (int j = 0; j < A->chunkLen[chunk]; j++) {
                ghost_gidx idx = A->chunkStart[chunk] + j * C + c;
                for (int n = 0; n < N; n++) {
                    wsums[n] += (oT)Aval[idx] * (oT)wval[A->col[idx] * ldw + n];
                }
            }

            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    result[n * M + m] += wsums[n] * (oT)vval[(chunk * C + c) * ldv + m];
                }
            }
        }
    }

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            xval[n * ldx + m] = alpha * result[n * M + m] + beta * xval[n * ldx + m];
        }
    }

    return GHOST_SUCCESS;
}
}


ghost_error tsmtspmtsm_var2_plain(ghost_densemat *x, ghost_densemat *v, ghost_densemat *w,
    ghost_sparsemat *A, void *pAlpha, void *pBeta)
{
    if (x->traits.datatype & GHOST_DT_REAL && w->traits.datatype & GHOST_DT_REAL) {
        if (x->traits.datatype & GHOST_DT_DOUBLE && w->traits.datatype & GHOST_DT_DOUBLE) {
            return tsmtspmtsm_var2_plain_kernel<double, double>(x, v, w, A, pAlpha, pBeta);
        }
        if (x->traits.datatype & GHOST_DT_FLOAT && w->traits.datatype & GHOST_DT_FLOAT) {
            return tsmtspmtsm_var2_plain_kernel<float, float>(x, v, w, A, pAlpha, pBeta);
        }
        if (x->traits.datatype & GHOST_DT_DOUBLE && w->traits.datatype & GHOST_DT_FLOAT) {
            return tsmtspmtsm_var2_plain_kernel<float, double>(x, v, w, A, pAlpha, pBeta);
        }
    }
    if (x->traits.datatype & GHOST_DT_COMPLEX && w->traits.datatype & GHOST_DT_COMPLEX) {
        if (x->traits.datatype & GHOST_DT_DOUBLE && w->traits.datatype & GHOST_DT_DOUBLE) {
            return tsmtspmtsm_var2_plain_kernel<std::complex<double>, std::complex<double>>(
                x, v, w, A, pAlpha, pBeta);
        }
        if (x->traits.datatype & GHOST_DT_FLOAT && w->traits.datatype & GHOST_DT_FLOAT) {
            return tsmtspmtsm_var2_plain_kernel<std::complex<float>, std::complex<float>>(
                x, v, w, A, pAlpha, pBeta);
        }
    }
    GHOST_ERROR_LOG("Data type configuration not valid");
    return GHOST_ERR_DATATYPE;
}
