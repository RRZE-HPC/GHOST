#include "ghost/crs_kacz.h"
#include "ghost/crs.h"
#include "ghost/complex.h"

template<typename m_t, typename v_t>
static ghost_error_t crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *x, ghost_densemat_t *b, v_t *omega)
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

    for (color=0; color<mat->ncolors; color++) {
#pragma omp parallel for private(j,row,rownorm)
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
                xval[crmat->col[j]] = xval[crmat->col[j]] - (*omega) * scal * (v_t)mval[j];
            }
        }
    }
    
    return GHOST_SUCCESS;
} 

extern "C" ghost_error_t dd_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega)
{ return crs_kacz< double,double >(mat,lhs,rhs,(double *)omega); }

extern "C" ghost_error_t ds_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega)
{ return crs_kacz< double,float >(mat,lhs,rhs,(float *)omega); }

extern "C" ghost_error_t dc_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega)
{ return crs_kacz< double,ghost_complex<float> >(mat,lhs,rhs,(ghost_complex<float> *)omega); }

extern "C" ghost_error_t dz_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega)
{ return crs_kacz< double,ghost_complex<double> >(mat,lhs,rhs,(ghost_complex<double> *)omega); }

extern "C" ghost_error_t sd_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega)
{ return crs_kacz< float,double >(mat,lhs,rhs,(double *)omega); }

extern "C" ghost_error_t ss_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega)
{ return crs_kacz< float,float >(mat,lhs,rhs,(float *)omega); }

extern "C" ghost_error_t sc_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega)
{ return crs_kacz< float,ghost_complex<float> >(mat,lhs,rhs,(ghost_complex<float> *)omega); }

extern "C" ghost_error_t sz_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega)
{ return crs_kacz< float,ghost_complex<double> >(mat,lhs,rhs,(ghost_complex<double> *)omega); }

extern "C" ghost_error_t cd_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega)
{ return crs_kacz< ghost_complex<float>,double >(mat,lhs,rhs,(double *)omega); }

extern "C" ghost_error_t cs_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega)
{ return crs_kacz< ghost_complex<float>,float >(mat,lhs,rhs,(float *)omega); }

extern "C" ghost_error_t cc_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega)
{ return crs_kacz< ghost_complex<float>,ghost_complex<float> >(mat,lhs,rhs,(ghost_complex<float> *)omega); }

extern "C" ghost_error_t cz_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega)
{ return crs_kacz< ghost_complex<float>,ghost_complex<double> >(mat,lhs,rhs,(ghost_complex<double> *)omega); }

extern "C" ghost_error_t zd_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega)
{ return crs_kacz< ghost_complex<double>,double >(mat,lhs,rhs,(double *)omega); }

extern "C" ghost_error_t zs_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega)
{ return crs_kacz< ghost_complex<double>,float >(mat,lhs,rhs,(float *)omega); }

extern "C" ghost_error_t zc_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega)
{ return crs_kacz< ghost_complex<double>,ghost_complex<float> >(mat,lhs,rhs,(ghost_complex<float> *)omega); }

extern "C" ghost_error_t zz_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega)
{ return crs_kacz< ghost_complex<double>,ghost_complex<double> >(mat,lhs,rhs,(ghost_complex<double> *)omega); }
