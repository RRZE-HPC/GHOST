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
    ghost_midx_t  *rpt;
    ghost_midx_t  *col;
    char *val;

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
ghost_error_t dd_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t options);
ghost_error_t ds_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t options);
ghost_error_t dc_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t options);
ghost_error_t dz_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t options);
ghost_error_t sd_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t options);
ghost_error_t ss_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t options);
ghost_error_t sc_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t options);
ghost_error_t sz_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t options);
ghost_error_t cd_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t options);
ghost_error_t cs_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t options);
ghost_error_t cc_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t options);
ghost_error_t cz_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t options);
ghost_error_t zd_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t options);
ghost_error_t zs_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t options);
ghost_error_t zc_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t options);
ghost_error_t zz_CRS_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t options);
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
const char * d_CRS_stringify(ghost_mat_t *mat, int dense);
const char * s_CRS_stringify(ghost_mat_t *mat, int dense);
const char * c_CRS_stringify(ghost_mat_t *mat, int dense);
const char * z_CRS_stringify(ghost_mat_t *mat, int dense);
#ifdef __cplusplus
}
#endif


#endif

