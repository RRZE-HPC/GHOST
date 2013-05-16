#include <ghost.h>
#include <ghost_util.h>
#include <ghost_mat.h>
#include <crs.h>
	

template<typename m_t, typename v_t> void CRS_kernel_plain_tmpl(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options) 
{
	v_t *rhsv = (v_t *)(rhs->val);	
	v_t *lhsv = (v_t *)(lhs->val);	
	ghost_midx_t i, j;
	v_t hlp1;
	CR_TYPE *cr = CR(mat);

#pragma omp parallel for schedule(runtime) private (hlp1, j)
	for (i=0; i<cr->nrows; i++){
			hlp1 = 0.0;
			for (j=cr->rpt[i]; j<cr->rpt[i+1]; j++){
				//printf("[%d/%d] %f*%f\n",i,cr->col[j],cr->val[j],rhsv[cr->col[j]]);
				hlp1 = hlp1 + (v_t)(((m_t *)(cr->val))[j]) * rhsv[cr->col[j]];
			}
			if (options & GHOST_SPMVM_AXPY) 
				lhsv[i] += hlp1;
			else
				lhsv[i] = hlp1;
	}
}

extern "C" void dd_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl<double,double>(mat,lhs,rhs,options); }

extern "C" void ds_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl<double,float>(mat,lhs,rhs,options); }

extern "C" void dc_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl<double,complex float>(mat,lhs,rhs,options); }

extern "C" void dz_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl<double,complex double>(mat,lhs,rhs,options); }

extern "C" void sd_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl<float,double>(mat,lhs,rhs,options); }

extern "C" void ss_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl<float,float>(mat,lhs,rhs,options); }

extern "C" void sc_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl<float,complex float>(mat,lhs,rhs,options); }

extern "C" void sz_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl<float,complex double>(mat,lhs,rhs,options); }

extern "C" void cd_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl<complex float,double>(mat,lhs,rhs,options); }

extern "C" void cs_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl<complex float,float>(mat,lhs,rhs,options); }

extern "C" void cc_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl<complex float,complex float>(mat,lhs,rhs,options); }

extern "C" void cz_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl<complex float,complex double>(mat,lhs,rhs,options); }

extern "C" void zd_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl<complex double,double>(mat,lhs,rhs,options); }

extern "C" void zs_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl<complex double,float>(mat,lhs,rhs,options); }

extern "C" void zc_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl<complex double,complex float>(mat,lhs,rhs,options); }

extern "C" void zz_CRS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return CRS_kernel_plain_tmpl<complex double,complex double>(mat,lhs,rhs,options); }

