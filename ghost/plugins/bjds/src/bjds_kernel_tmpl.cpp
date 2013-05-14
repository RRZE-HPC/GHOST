#include <ghost.h>
#include <bjds.h>
#include <stdio.h>

template<typename m_t, typename v_t> void BJDS_kernel_plain_tmpl(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{
	printf("i'm here\n");

	v_t *rhsd = (v_t *)(rhs->val);
	v_t *lhsd = (v_t *)(lhs->val);
	ghost_midx_t i,j,c;
	v_t tmp[BJDS_LEN];
	BJDS_TYPE *bjds = (BJDS_TYPE *)(mat->data);
}

extern "C" void dd_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl<double,double>(mat,lhs,rhs,options); }

extern "C" void ds_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl<double,float>(mat,lhs,rhs,options); }

extern "C" void dc_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl<double,complex float>(mat,lhs,rhs,options); }

extern "C" void dz_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl<double,complex double>(mat,lhs,rhs,options); }

extern "C" void sd_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl<float,double>(mat,lhs,rhs,options); }

extern "C" void ss_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl<float,float>(mat,lhs,rhs,options); }

extern "C" void sc_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl<float,complex float>(mat,lhs,rhs,options); }

extern "C" void sz_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl<float,complex double>(mat,lhs,rhs,options); }

extern "C" void cd_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl<complex float,double>(mat,lhs,rhs,options); }

extern "C" void cs_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl<complex float,float>(mat,lhs,rhs,options); }

extern "C" void cc_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl<complex float,complex float>(mat,lhs,rhs,options); }

extern "C" void cz_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl<complex float,complex double>(mat,lhs,rhs,options); }

extern "C" void zd_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl<complex double,double>(mat,lhs,rhs,options); }

extern "C" void zs_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl<complex double,float>(mat,lhs,rhs,options); }

extern "C" void zc_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl<complex double,complex float>(mat,lhs,rhs,options); }

extern "C" void zz_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl<complex double,complex double>(mat,lhs,rhs,options); }
