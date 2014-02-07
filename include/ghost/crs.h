#ifndef GHOST_CRS_H
#define GHOST_CRS_H

#include "config.h"
#include "types.h"
#include "mat.h"

#if GHOST_HAVE_CUDA
#include "cu_crs.h"
#endif

/**
 * @brief Struct defining a CRS matrix.
 */
typedef struct 
{
    ghost_mnnz_t  *rpt;
    ghost_midx_t  *col;
    char *val;
} 
ghost_crs_t;


#define CR(mat) ((ghost_crs_t *)((mat)->data))

ghost_error_t ghost_CRS_init(ghost_mat_t *mat);
#ifdef __cplusplus
extern "C" {
#endif
    
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
const char * d_CRS_stringify(ghost_mat_t *mat, int dense);
const char * s_CRS_stringify(ghost_mat_t *mat, int dense);
const char * c_CRS_stringify(ghost_mat_t *mat, int dense);
const char * z_CRS_stringify(ghost_mat_t *mat, int dense);

#ifdef __cplusplus
}
#endif


#endif

