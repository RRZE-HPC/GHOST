#include "ghost/config.h"

#if GHOST_HAVE_MPI
#include <mpi.h> //mpi.h has to be included before stdio.h
#endif
#include <stdio.h>

#include "ghost/complex.h"
#include "ghost/math.h"
#include "ghost/util.h"
#include "ghost/crs.h"
#include "ghost/constants.h"
#include <iostream>

// TODO shift, scale als templateparameter

template<typename m_t, typename v_t> void CRS_kernel_plain_tmpl(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{
    CR_TYPE *cr = CR(mat);
    v_t *rhsv;
    v_t *lhsv;
    v_t *local_dot_product, *partsums;
    m_t *mval = (m_t *)(cr->val);
    ghost_midx_t i, j;
    ghost_vidx_t v;
    int nthreads = 1;

    v_t hlp1 = 0.;
    v_t shift, scale, beta;
    if (options & GHOST_SPMVM_APPLY_SHIFT)
        shift = *((v_t *)(mat->traits->shift));
    if (options & GHOST_SPMVM_APPLY_SCALE)
        scale = *((v_t *)(mat->traits->scale));
    if (options & GHOST_SPMVM_AXPBY)
        beta = *((v_t *)(mat->traits->beta));
    if (options & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {
        local_dot_product = ((v_t *)(lhs->traits->localdot));

#pragma omp parallel
        nthreads = ghost_ompGetNumThreads();

        partsums = (v_t *)ghost_malloc(3*lhs->traits->nvecs*nthreads*sizeof(v_t));

        for (i=0; i<3*lhs->traits->nvecs*nthreads; i++) {
            partsums[i] = 0.;
        }
    }

#pragma omp parallel for schedule(runtime) private (hlp1, j, rhsv, lhsv,v) shared (partsums)
    for (i=0; i<cr->nrows; i++){
        for (v=0; v<MIN(lhs->traits->nvecs,rhs->traits->nvecs); v++)
        {
            rhsv = (v_t *)rhs->val[v];
            lhsv = (v_t *)lhs->val[v];
            hlp1 = (v_t)0.0;
            for (j=cr->rpt[i]; j<cr->rpt[i+1]; j++){
                hlp1 += ((v_t)(mval[j])) * rhsv[cr->col[j]];
            }

            if (options & GHOST_SPMVM_APPLY_SHIFT) {
                if (options & GHOST_SPMVM_APPLY_SCALE) {
                    if (options & GHOST_SPMVM_AXPY) {
                        lhsv[i] += scale*(v_t)(hlp1-shift*rhsv[i]);
                    } else if (options & GHOST_SPMVM_AXPBY) {
                        lhsv[i] = beta*lhsv[i] + scale*(v_t)(hlp1-shift*rhsv[i]);
                    } else {
                        lhsv[i] = scale*(v_t)(hlp1-shift*rhsv[i]);
                    }
                } else {
                    if (options & GHOST_SPMVM_AXPY) {
                        lhsv[i] += (hlp1-shift*rhsv[i]);
                    } else if (options & GHOST_SPMVM_AXPBY) {
                        lhsv[i] = beta*lhsv[i] + hlp1-shift*rhsv[i];
                    } else {
                        lhsv[i] = (hlp1-shift*rhsv[i]);
                    }
                }
            } else {
                if (options & GHOST_SPMVM_APPLY_SCALE) {
                    if (options & GHOST_SPMVM_AXPY) {
                        lhsv[i] += scale*(v_t)(hlp1);
                    } else if (options & GHOST_SPMVM_AXPBY) {
                        lhsv[i] = beta*lhsv[i] + scale*(v_t)hlp1;
                    } else {
                        lhsv[i] = scale*(v_t)(hlp1);
                    }
                } else {
                    if (options & GHOST_SPMVM_AXPY) {
                        lhsv[i] += (hlp1);
                    } else if (options & GHOST_SPMVM_AXPBY) {
                        lhsv[i] = beta*lhsv[i] + hlp1;
                    } else {
                        lhsv[i] = (hlp1);
                    }
                }

            }

            if (options & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {
                partsums[(v+0*lhs->traits->nvecs)*nthreads + ghost_ompGetThreadNum()] += conjugate(&lhsv[i])*lhsv[i];
                partsums[(v+1*lhs->traits->nvecs)*nthreads + ghost_ompGetThreadNum()] += conjugate(&lhsv[i])*rhsv[i];
                partsums[(v+2*lhs->traits->nvecs)*nthreads + ghost_ompGetThreadNum()] += conjugate(&rhsv[i])*rhsv[i];
            }
        }
    }
    if (options & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {
        for (v=0; v<MIN(lhs->traits->nvecs,rhs->traits->nvecs); v++) {
            for (i=0; i<nthreads; i++) {
                local_dot_product[v                       ] += partsums[(v+0*lhs->traits->nvecs)*nthreads + i];
                local_dot_product[v +   lhs->traits->nvecs] += partsums[(v+1*lhs->traits->nvecs)*nthreads + i];
                local_dot_product[v + 2*lhs->traits->nvecs] += partsums[(v+2*lhs->traits->nvecs)*nthreads + i];
            }
        }
    }
}

template<typename m_t, typename f_t> void CRS_castData_tmpl(void *matrixData, void *fileData, int nEnts)
{
    ghost_mnnz_t i;
    m_t *md = (m_t *)matrixData;
    f_t *fd = (f_t *)fileData;

    for (i = 0; i<nEnts; i++) {
        md[i] = (m_t)(fd[i]);
    }
}

template<typename m_t> void CRS_valToStr_tmpl(void *val, char *str, int n)
{
    if (val == NULL) {
        UNUSED(str);
        //str = "0.";
    } else {
        UNUSED(str);
        UNUSED(n);
        // TODO
        //snprintf(str,n,"%g",*((m_t *)(val)));
    }
}


extern "C" void dd_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl< double,double >(mat,lhs,rhs,options); }

extern "C" void ds_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl< double,float >(mat,lhs,rhs,options); }

extern "C" void dc_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl< double,ghost_complex<float> >(mat,lhs,rhs,options); }

extern "C" void dz_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl< double,ghost_complex<double> >(mat,lhs,rhs,options); }

extern "C" void sd_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl< float,double >(mat,lhs,rhs,options); }

extern "C" void ss_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl< float,float >(mat,lhs,rhs,options); }

extern "C" void sc_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl< float,ghost_complex<float> >(mat,lhs,rhs,options); }

extern "C" void sz_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl< float,ghost_complex<double> >(mat,lhs,rhs,options); }

extern "C" void cd_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl< ghost_complex<float>,double >(mat,lhs,rhs,options); }

extern "C" void cs_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl< ghost_complex<float>,float >(mat,lhs,rhs,options); }

extern "C" void cc_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl< ghost_complex<float>,ghost_complex<float> >(mat,lhs,rhs,options); }

extern "C" void cz_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl< ghost_complex<float>,ghost_complex<double> >(mat,lhs,rhs,options); }

extern "C" void zd_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl< ghost_complex<double>,double >(mat,lhs,rhs,options); }

extern "C" void zs_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl< ghost_complex<double>,float >(mat,lhs,rhs,options); }

extern "C" void zc_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl< ghost_complex<double>,ghost_complex<float> >(mat,lhs,rhs,options); }

extern "C" void zz_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl< ghost_complex<double>,ghost_complex<double> >(mat,lhs,rhs,options); }

extern "C" void dd_CRS_castData(void *matrixData, void *fileData, int nEnts)
{ return CRS_castData_tmpl< double,double >(matrixData, fileData, nEnts); }

extern "C" void ds_CRS_castData(void *matrixData, void *fileData, int nEnts)
{ return CRS_castData_tmpl< double,float >(matrixData, fileData, nEnts); }

extern "C" void dc_CRS_castData(void *matrixData, void *fileData, int nEnts)
{ return CRS_castData_tmpl< double,ghost_complex<float> >(matrixData, fileData, nEnts); }

extern "C" void dz_CRS_castData(void *matrixData, void *fileData, int nEnts)
{ return CRS_castData_tmpl< double,ghost_complex<double> >(matrixData, fileData, nEnts); }

extern "C" void sd_CRS_castData(void *matrixData, void *fileData, int nEnts)
{ return CRS_castData_tmpl< float,double >(matrixData, fileData, nEnts); }

extern "C" void ss_CRS_castData(void *matrixData, void *fileData, int nEnts)
{ return CRS_castData_tmpl< float,float >(matrixData, fileData, nEnts); }

extern "C" void sc_CRS_castData(void *matrixData, void *fileData, int nEnts)
{ return CRS_castData_tmpl< float,ghost_complex<float> >(matrixData, fileData, nEnts); }

extern "C" void sz_CRS_castData(void *matrixData, void *fileData, int nEnts)
{ return CRS_castData_tmpl< float,ghost_complex<double> >(matrixData, fileData, nEnts); }

extern "C" void cd_CRS_castData(void *matrixData, void *fileData, int nEnts)
{ return CRS_castData_tmpl< ghost_complex<float>,double >(matrixData, fileData, nEnts); }

extern "C" void cs_CRS_castData(void *matrixData, void *fileData, int nEnts)
{ return CRS_castData_tmpl< ghost_complex<float>,float >(matrixData, fileData, nEnts); }

extern "C" void cc_CRS_castData(void *matrixData, void *fileData, int nEnts)
{ return CRS_castData_tmpl< ghost_complex<float>,ghost_complex<float> >(matrixData, fileData, nEnts); }

extern "C" void cz_CRS_castData(void *matrixData, void *fileData, int nEnts)
{ return CRS_castData_tmpl< ghost_complex<float>,ghost_complex<double> >(matrixData, fileData, nEnts); }

extern "C" void zd_CRS_castData(void *matrixData, void *fileData, int nEnts)
{ return CRS_castData_tmpl< ghost_complex<double>,double >(matrixData, fileData, nEnts); }

extern "C" void zs_CRS_castData(void *matrixData, void *fileData, int nEnts)
{ return CRS_castData_tmpl< ghost_complex<double>,float >(matrixData, fileData, nEnts); }

extern "C" void zc_CRS_castData(void *matrixData, void *fileData, int nEnts)
{ return CRS_castData_tmpl< ghost_complex<double>,ghost_complex<float> >(matrixData, fileData, nEnts); }

extern "C" void zz_CRS_castData(void *matrixData, void *fileData, int nEnts)
{ return CRS_castData_tmpl< ghost_complex<double>,ghost_complex<double> >(matrixData, fileData, nEnts); }

extern "C" void d_CRS_valToStr(void *val, char *str, int n)
{ return CRS_valToStr_tmpl< double >(val,str,n); }

extern "C" void s_CRS_valToStr(void *val, char *str, int n)
{ return CRS_valToStr_tmpl< float >(val,str,n); }

extern "C" void c_CRS_valToStr(void *val, char *str, int n)
{ return CRS_valToStr_tmpl< ghost_complex<float> >(val,str,n); }

extern "C" void z_CRS_valToStr(void *val, char *str, int n)
{ return CRS_valToStr_tmpl< ghost_complex<double> >(val,str,n); }
