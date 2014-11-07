#include "ghost/crs_kacz.h"
#include "ghost/crs.h"
#include "ghost/complex.h"
#include "ghost/locality.h"
#include <complex>

template<typename m_t, typename v_t, bool forward>
static ghost_error_t crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *x, ghost_densemat_t *b, v_t *omega, int forward)
{
    INFO_LOG("in kacz kernel");
    if (!mat->color_ptr) {
        WARNING_LOG("Matrix has not been colored!");
    }
   
    ghost_lidx_t i;
    ghost_lidx_t row;
    ghost_lidx_t j;
    ghost_lidx_t color;
    ghost_crs_t *crmat = CR(mat);
    v_t *bval = (v_t *)(b->val[0]);
    v_t *xval = (v_t *)(x->val[0]);
    m_t *mval = (m_t *)crmat->val;
    m_t rownorm;


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
#pragma omp parallel for private(i,j,row,rownorm)
        for (i=mat->color_ptr[color]; i<mat->color_ptr[color+1]; i++) {
            row = i;
            rownorm = 0.;
            v_t scal = -bval[row];

            for (j=crmat->rpt[row]; j<crmat->rpt[row+1]; j++) {
                scal += (v_t)mval[j] * xval[crmat->col[j]];
                rownorm += mval[j]*mval[j];
            }

            scal /= (v_t)rownorm;

            for (j=crmat->rpt[row]; j<crmat->rpt[row+1]; j++) {
                xval[crmat->col[j]] = xval[crmat->col[j]] - (*omega) * scal * (v_t)mval[j];
            }
        }
    }
    
    return GHOST_SUCCESS;
}

ghost_error_t ghost_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega, int forward)
{
    if (mat->traits->datatype != lhs->traits.datatype || lhs->traits.datatype != rhs->traits.datatype) {
        WARNING_LOG("Mixed data types not yet implemented!");
    }

    if (rhs->traits.datatype & GHOST_DT_COMPLEX) {
        if (rhs->traits.datatype & GHOST_DT_DOUBLE) {
            if (forward) {
                return crs_kacz<std::complex<double>, std::complex<double>, true>(mat,lhs,rhs,(std::complex<double> *)omega,forward);
            } else {
                return crs_kacz<std::complex<double>, std::complex<double>, false>(mat,lhs,rhs,(std::complex<double> *)omega,forward);
            }
        } else {
            if (forward) {
                return crs_kacz<std::complex<float>, std::complex<float>, true>(mat,lhs,rhs,(std::complex<float> *)omega,forward);
            } else {
                return crs_kacz<std::complex<float>, std::complex<float>, false>(mat,lhs,rhs,(std::complex<float> *)omega,forward);
            }
        }
    } else {
        if (rhs->traits.datatype & GHOST_DT_DOUBLE) {
            if (forward) {
                return crs_kacz<double, double, true>(mat,lhs,rhs,(double *)omega,forward);
            } else {
                return crs_kacz<double, double, false>(mat,lhs,rhs,(double *)omega,forward);
            }
        } else {
            if (forward) {
                return crs_kacz<float, float, true>(mat,lhs,rhs,(float *)omega,forward);
            } else {
                return crs_kacz<float, float, false>(mat,lhs,rhs,(float *)omega,forward);
            }
        }
    }
}
