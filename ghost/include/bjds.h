
#ifndef __GHOST_SPMFORMAT_BJDS__
#define __GHOST_SPMFORMAT_BJDS__

#include "ghost.h"

#ifdef MIC
//#define BJDS_LEN 8
#define BJDS_LEN 16
#elif defined (AVX)
#define BJDS_LEN 4 // TODO single/double precision
#elif defined (SSE)
#define BJDS_LEN 2
#elif defined (OPENCL) || defined (CUDA)
#define BJDS_LEN 256
#elif defined (VSX)
#define BJDS_LEN 2
#else
#define BJDS_LEN 4
#endif

typedef struct 
{
#ifdef OPENCL
	cl_mem val;
	cl_mem col;
	cl_mem rowLen;
	cl_mem chunkStart;
	cl_mem chunkLen;
	ghost_cl_midx_t nrows;
	ghost_cl_midx_t nrowsPadded;
#endif
} 
CL_BJDS_TYPE;

typedef struct 
{
#ifdef CUDA
	char * val;
	ghost_midx_t * col;
	ghost_midx_t * rowLen;
	ghost_mnnz_t * chunkStart;
	ghost_midx_t * chunkLen;
	ghost_midx_t nrows;
	ghost_midx_t nrowsPadded;
#endif
} 
CU_BJDS_TYPE;

typedef struct 
{
	char *val;
	ghost_midx_t *col;
	ghost_mnnz_t *chunkStart;
	ghost_midx_t nrows;
	ghost_midx_t nrowsPadded;
	ghost_mnnz_t nnz;
	ghost_mnnz_t nEnts;
	double nu;
	double mu;
	double beta;
	ghost_midx_t *chunkMin; // for version with remainder loop
	ghost_midx_t *chunkLen; // for version with remainder loop
	ghost_midx_t *rowLen;   // for version with remainder loop
	
	CL_BJDS_TYPE *clmat;
	CU_BJDS_TYPE *cumat;
} 
BJDS_TYPE;

#define BJDS(mat) ((BJDS_TYPE *)(mat->data))

ghost_mat_t * ghost_BJDS_init(ghost_mtraits_t *);
#ifdef __cplusplus
template<typename m_t, typename v_t> void BJDS_kernel_plain_tmpl(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options);
template <typename m_t> void BJDS_fromCRS(ghost_mat_t *mat, void *crs);
extern "C" {
#endif
void dd_BJDS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void ds_BJDS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void dc_BJDS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void dz_BJDS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void sd_BJDS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void ss_BJDS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void sc_BJDS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void sz_BJDS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void cd_BJDS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void cs_BJDS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void cc_BJDS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void cz_BJDS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void zd_BJDS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void zs_BJDS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void zc_BJDS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void zz_BJDS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
#ifdef CUDA
void dd_BJDS_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void ds_BJDS_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void dc_BJDS_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void dz_BJDS_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void sd_BJDS_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void ss_BJDS_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void sc_BJDS_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void sz_BJDS_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void cd_BJDS_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void cs_BJDS_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void cc_BJDS_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void cz_BJDS_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void zd_BJDS_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void zs_BJDS_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void zc_BJDS_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void zz_BJDS_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
#endif
void d_BJDS_fromCRS(ghost_mat_t *mat, void *crs);
void s_BJDS_fromCRS(ghost_mat_t *mat, void *crs);
void c_BJDS_fromCRS(ghost_mat_t *mat, void *crs);
void z_BJDS_fromCRS(ghost_mat_t *mat, void *crs);
#ifdef __cplusplus
}
#endif

#endif
