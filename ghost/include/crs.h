#ifndef __GHOST_SPMFORMAT_CRS__
#define __GHOST_SPMFORMAT_CRS__

#include "ghost.h"

typedef struct
{
/*	ghost_midx_t len;
	ghost_midx_t idx;
	ghost_dt val;
	ghost_midx_t minRow;
	ghost_midx_t maxRow;*/
}
CONST_DIAG;

typedef struct 
{
#ifdef OPENCL
	ghost_cl_midx_t  nrows, ncols;
	ghost_cl_mnnz_t  nEnts;
	cl_mem rpt;
	cl_mem col;
	cl_mem val;
#endif
} 
CL_CR_TYPE;

typedef struct 
{
	ghost_midx_t  nrows, ncols;
	ghost_mnnz_t  nEnts;
	ghost_midx_t  *rpt;
	ghost_midx_t  *col;
	char *val;

	CL_CR_TYPE *clmat;
	ghost_midx_t nConstDiags;
	CONST_DIAG *constDiags;
} 
CR_TYPE;

/*typedef struct 
{
	ghost_midx_t row, col, nThEntryInRow;
	ghost_dt val;
} 
NZE_TYPE;*/


#define CR(mat) ((CR_TYPE *)(mat->data))

ghost_mat_t * ghost_CRS_init(ghost_mtraits_t *);
#ifdef __cplusplus
template<typename m_t, typename v_t> void CRS_kernel_plain_tmpl(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options);
extern "C" {
#endif
void dd_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void ds_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void dc_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void dz_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void sd_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void ss_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void sc_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void sz_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void cd_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void cs_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void cc_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void cz_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void zd_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void zs_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void zc_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void zz_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
#ifdef __cplusplus
}
#endif


#endif

