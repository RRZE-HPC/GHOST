#ifndef __GHOST_SPMFORMAT_CRS__
#define __GHOST_SPMFORMAT_CRS__

#include "config.h"
#include "types.h"

#if GHOST_HAVE_CUDA
#include "cu_crs.h"
#endif

typedef struct
{
    ghost_midx_t len;
    ghost_midx_t idx;
    void *val;
    ghost_midx_t minRow;
    ghost_midx_t maxRow;
}
CONST_DIAG;

typedef struct 
{
#ifdef GHOST_HAVE_OPENCL
    ghost_cl_midx_t  nrows, ncols;
    ghost_cl_mnnz_t  nEnts;
    cl_mem rpt;
    cl_mem col;
    cl_mem val;
#else
    void *empty;
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

ghost_mat_t * ghost_CRS_init(ghost_context_t *, ghost_mtraits_t *);
#ifdef __cplusplus
template<typename, typename> void CRS_kernel_plain_tmpl(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int);
template<typename m_t, typename f_t> void CRS_castData_tmpl(void *matrixData, void *fileData, int nEnts);
template<typename> void CRS_valToStr_tmpl(void *, char *, int);
extern "C" {
#endif
    
ghost_midx_t * CRS_readRpt(ghost_midx_t nrpt, char *matrixPath);
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
void dd_CRS_castData(void *, void *, int);
void ds_CRS_castData(void *, void *, int);
void dc_CRS_castData(void *, void *, int);
void dz_CRS_castData(void *, void *, int);
void sd_CRS_castData(void *, void *, int);
void ss_CRS_castData(void *, void *, int);
void sc_CRS_castData(void *, void *, int);
void sz_CRS_castData(void *, void *, int);
void cd_CRS_castData(void *, void *, int);
void cs_CRS_castData(void *, void *, int);
void cc_CRS_castData(void *, void *, int);
void cz_CRS_castData(void *, void *, int);
void zd_CRS_castData(void *, void *, int);
void zs_CRS_castData(void *, void *, int);
void zc_CRS_castData(void *, void *, int);
void zz_CRS_castData(void *, void *, int);
void d_CRS_valToStr(void *, char *, int);
void s_CRS_valToStr(void *, char *, int);
void c_CRS_valToStr(void *, char *, int);
void z_CRS_valToStr(void *, char *, int);
#ifdef __cplusplus
}
#endif


#endif

