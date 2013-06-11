#include <ghost.h>
#include <ghost_util.h>
#include <ghost_mat.h>
#include <crs.h>
#include "ghost_complex.h"
#include <iostream>


template<typename m_t, typename v_t> void CRS_kernel_plain_tmpl(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options) 
{
	CR_TYPE *cr = CR(mat);
	v_t *rhsv = (v_t *)(rhs->val);	
	v_t *lhsv = (v_t *)(lhs->val);
	m_t *mval = (m_t *)(cr->val);	
	ghost_midx_t i, j;
	v_t hlp1 = 0.;

	if (options & GHOST_SPMVM_APPLY_SHIFT) {
		m_t shift = *((m_t *)(mat->traits->shift));
#pragma omp parallel for schedule(runtime) private (hlp1, j)
		for (i=0; i<cr->nrows; i++){
			hlp1 = (v_t)0.0;
			for (j=cr->rpt[i]; j<cr->rpt[i+1]; j++){
				hlp1 += ((v_t)(mval[j] + shift)) * rhsv[cr->col[j]];
			}
			if (options & GHOST_SPMVM_AXPY) { 
				lhsv[i] += hlp1;
			} else {
				lhsv[i] = hlp1;
			}
		}
	} else {
#pragma omp parallel for schedule(runtime) private (hlp1, j)
		for (i=0; i<cr->nrows; i++){
			hlp1 = (v_t)0.0;
			for (j=cr->rpt[i]; j<cr->rpt[i+1]; j++){
				hlp1 += (v_t)(mval[j]) * rhsv[cr->col[j]];
			}
			if (options & GHOST_SPMVM_AXPY) { 
				lhsv[i] += hlp1;
			} else {
				lhsv[i] = hlp1;
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
