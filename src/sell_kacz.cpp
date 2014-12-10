#include "ghost/sell.h"
#include "ghost/complex.h"
#include "ghost/locality.h"
#include "ghost/util.h"
#include <complex>

template<typename m_t, typename v_t, bool forward>
static ghost_error_t sell_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *x, ghost_densemat_t *b, v_t *omega)
{
    if (!mat->color_ptr || mat->ncolors == 0) {
        WARNING_LOG("Matrix has not been colored!");
    }
    if (x->traits.ncols > 1) {
        ERROR_LOG("Multi-vec not implemented!");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
   
    ghost_lidx_t c;
    ghost_lidx_t row;
    ghost_lidx_t rowinchunk;
    ghost_lidx_t j;
    ghost_lidx_t color;
    ghost_sell_t *sellmat = SELL(mat);
    ghost_lidx_t fchunk, lchunk;
    v_t *bval = (v_t *)(b->val[0]);
    v_t *xval = (v_t *)(x->val[0]);
    m_t *mval = (m_t *)sellmat->val;


    int rank;
    ghost_rank(&rank,mat->context->mpicomm);

    ghost_lidx_t firstcolor, lastcolor, stride;
    
    if (forward) {
        firstcolor = 0;
        lastcolor = mat->ncolors;
        stride = 1;
    } else {
        firstcolor = mat->ncolors-1;
        lastcolor = -1;
        stride = -1;
    }

    
    for (color=firstcolor; color!=lastcolor; color+=stride) {
        fchunk = mat->color_ptr[color]/sellmat->chunkHeight;
        lchunk = mat->color_ptr[color+1]/sellmat->chunkHeight;
#pragma omp parallel
        { 
            m_t *rownorm;
            ghost_malloc((void **)&rownorm,sellmat->chunkHeight*sizeof(m_t));
#pragma omp for private(j,row,rowinchunk)
            for (c=fchunk; c<lchunk; c++) {
                for (rowinchunk = 0; rowinchunk < sellmat->chunkHeight; rowinchunk++) {
                    row = rowinchunk + c*sellmat->chunkHeight;
                    rownorm[rowinchunk] = 0.;

                    ghost_lidx_t idx = sellmat->chunkStart[c]+rowinchunk;
                    v_t scal = -bval[row];

                    for (j=0; j<sellmat->rowLen[row]; j++) {
                        scal += (v_t)mval[idx] * xval[sellmat->col[idx]];
                        rownorm[rowinchunk] += mval[idx]*mval[idx];
                        idx += sellmat->chunkHeight;
                    }

                    idx -= sellmat->chunkHeight*sellmat->rowLen[row];
                    scal /= (v_t)rownorm[rowinchunk];

                    for (j=0; j<sellmat->rowLen[row]; j++) {
                        xval[sellmat->col[idx]] = xval[sellmat->col[idx]] - (*omega) * scal * (v_t)mval[idx];
                        idx += sellmat->chunkHeight;
                    }
                }
            }
            free(rownorm);
            rownorm = NULL;
        }
    }
    
    return GHOST_SUCCESS;
}

ghost_error_t ghost_sell_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega, int forward)
{
    if (mat->traits->datatype != lhs->traits.datatype || lhs->traits.datatype != rhs->traits.datatype) {
        WARNING_LOG("Mixed data types not yet implemented!");
    }

    if (rhs->traits.datatype & GHOST_DT_COMPLEX) {
        if (rhs->traits.datatype & GHOST_DT_DOUBLE) {
            if (forward) {
                return sell_kacz<std::complex<double>, std::complex<double>, true>(mat,lhs,rhs,(std::complex<double> *)omega);
            } else {
                return sell_kacz<std::complex<double>, std::complex<double>, false>(mat,lhs,rhs,(std::complex<double> *)omega);
            }
        } else {
            if (forward) {
                return sell_kacz<std::complex<float>, std::complex<float>, true>(mat,lhs,rhs,(std::complex<float> *)omega);
            } else {
                return sell_kacz<std::complex<float>, std::complex<float>, false>(mat,lhs,rhs,(std::complex<float> *)omega);
            }
        }
    } else {
        if (rhs->traits.datatype & GHOST_DT_DOUBLE) {
            if (forward) {
                return sell_kacz<double, double, true>(mat,lhs,rhs,(double *)omega);
            } else {
                return sell_kacz<double, double, false>(mat,lhs,rhs,(double *)omega);
            }
        } else {
            if (forward) {
                return sell_kacz<float, float, true>(mat,lhs,rhs,(float *)omega);
            } else {
                return sell_kacz<float, float, false>(mat,lhs,rhs,(float *)omega);
            }
        }
    }
}
