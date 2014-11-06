#include "ghost/crs_kacz.h"
#include "ghost/crs.h"
#include "ghost/complex.h"
#include "ghost/locality.h"
#include <complex>

template<typename m_t, typename v_t>
static ghost_error_t crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *x, ghost_densemat_t *b, v_t *omega, int forward)
{
    INFO_LOG("in kacz kernel");
   
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
    
    if (forward) {
#pragma omp parallel for private(i,j,row,rownorm)
        for (color=0; color<mat->ncolors; color++) {
            for (i=mat->color_ptr[color]; i<mat->color_ptr[color+1]; i++) {
                row = mat->color_map[i];
                rownorm = 0.;
                v_t scal = -bval[row];

                for (j=crmat->rpt[row]; j<crmat->rpt[row+1]; j++) {
                    scal += (v_t)mval[j] * xval[crmat->col[j]];
                    rownorm += mval[j]*mval[j];
                }

                scal /= (v_t)rownorm;

                for (j=crmat->rpt[row]; j<crmat->rpt[row+1]; j++) {
                    //printf("rank %d xval[%d] = %f -= %f*%f = ",rank,xval[crmat->col[j]], crmat->col[j], scal, (v_t)mval[j]);
                    xval[crmat->col[j]] = xval[crmat->col[j]] - (*omega) * scal * (v_t)mval[j];
                    //printf("%f\n",xval[crmat->col[j]]);
                }
            }
        }
    } else {
#pragma omp parallel for private(i,j,row,rownorm)
        for (color=mat->ncolors-1; color>=0; color--) {
            for (i=mat->color_ptr[color+1]-1; i>=mat->color_ptr[color]; i--) {
                row = mat->color_map[i];
                rownorm = 0.;
                v_t scal = -bval[row];

                for (j=crmat->rpt[row]; j<crmat->rpt[row+1]; j++) {
                    scal += (v_t)mval[j] * xval[crmat->col[j]];
                    rownorm += mval[j]*mval[j];
                }

                scal /= (v_t)rownorm;

                for (j=crmat->rpt[row]; j<crmat->rpt[row+1]; j++) {
                    //printf("rank %d xval[%d] = %f -= %f*%f = ",rank,xval[crmat->col[j]], crmat->col[j], scal, (v_t)mval[j]);
                    xval[crmat->col[j]] = xval[crmat->col[j]] - (*omega) * scal * (v_t)mval[j];
                    //printf("%f\n",xval[crmat->col[j]]);
                }
            }
        }
    }
    
    return GHOST_SUCCESS;
}

ghost_error_t ghost_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega, int forward)
{
    WARNING_LOG("TODO: take into account matrix data type");

    if (rhs->traits.datatype & GHOST_DT_COMPLEX) {
        if (rhs->traits.datatype & GHOST_DT_DOUBLE) {
            return crs_kacz<std::complex<double>, std::complex<double>>(mat,lhs,rhs,(std::complex<double> *)omega,forward);
        } else {
            return crs_kacz<std::complex<float>, std::complex<float>>(mat,lhs,rhs,(std::complex<float> *)omega,forward);
        }
    } else {
        if (rhs->traits.datatype & GHOST_DT_DOUBLE) {
            return crs_kacz<double, double>(mat,lhs,rhs,(double *)omega,forward);
        } else {
            return crs_kacz<float, float>(mat,lhs,rhs,(float *)omega,forward);
        }
    }
}
