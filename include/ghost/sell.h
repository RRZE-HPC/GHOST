#ifndef GHOST_SELL_H
#define GHOST_SELL_H

#include "config.h"
#include "types.h"
#include "mat.h"

#define GHOST_SELL_CHUNKHEIGHT_ELLPACK 0
#define GHOST_SELL_CHUNKHEIGHT_AUTO -1
#define GHOST_SELL_SORT_GLOBALLY -1
#define SELL_AUX_S 0
#define SELL_AUX_C 1
#define SELL_AUX_T 2

typedef struct 
{
#ifdef GHOST_HAVE_CUDA
    char * val;
    ghost_midx_t * col;
    ghost_midx_t * rowLen;
    ghost_midx_t * rowLenPadded;
    ghost_mnnz_t * chunkStart;
    ghost_midx_t * chunkLen;
    ghost_midx_t nrows;
    ghost_midx_t nrowsPadded;
    int T; // number of threads per row (if applicable)
    ghost_midx_t chunkHeight;
#else
    void *empty;
#endif
} 
CU_SELL_TYPE;

typedef struct 
{
    char *val;
    ghost_midx_t *col;
    ghost_mnnz_t *chunkStart;
    double beta; // chunk occupancy
    double variance; // row length variance
    double deviation; // row lenght standard deviation
    double cv; // row lenght coefficient of variation
    int T; // number of threads per row (if applicable)
    ghost_midx_t *chunkMin; // for version with remainder loop
    ghost_midx_t *chunkLen; // for version with remainder loop
    ghost_midx_t *chunkLenPadded; // for version with remainder loop
    ghost_midx_t *rowLen;   // for version with remainder loop
    ghost_midx_t *rowLenPadded; // for SELL-T 
    ghost_midx_t maxRowLen;
    ghost_midx_t nMaxRows;
    ghost_midx_t chunkHeight;
    ghost_midx_t scope;
    
    CU_SELL_TYPE *cumat;
} 
SELL_TYPE;

typedef struct 
{
    ghost_midx_t row, nEntsInRow;
} 
ghost_sorting_t;
#define SELL(mat) ((SELL_TYPE *)(mat->data))

#define SELL_CUDA_THREADSPERBLOCK 256

enum GHOST_SELL_C {GHOST_SELL_C_1 = 1, GHOST_SELL_C_2 = 2, GHOST_SELL_C_4 = 4, GHOST_SELL_C_256 = 256};
enum GHOST_SELL_T {GHOST_SELL_T_1 = 1, GHOST_SELL_T_2 = 2, GHOST_SELL_T_4 = 4, GHOST_SELL_T_256 = 256};

ghost_error_t ghost_SELL_init(ghost_context_t *ctx, ghost_mtraits_t * traits, ghost_mat_t **mat);
#ifdef __cplusplus
template<typename, typename, int> void SELL_kernel_plain_tmpl(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int);
template<typename, typename> void SELL_kernel_plain_ELLPACK_tmpl(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int);
template <typename> ghost_error_t SELL_fromCRS(ghost_mat_t *, ghost_mat_t *);
extern "C" {
#endif
ghost_error_t dd_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t ds_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t dc_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t dz_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t sd_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t ss_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t sc_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t sz_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t cd_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t cs_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t cc_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t cz_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t zd_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t zs_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t zc_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t zz_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);

ghost_error_t dd_SELL_kernel_SSE(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t dd_SELL_kernel_AVX(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t dd_SELL_kernel_AVX_32(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t dd_SELL_kernel_AVX_32_rich(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t dd_SELL_kernel_MIC_16(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t dd_SELL_kernel_MIC_32(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
#if GHOST_HAVE_CUDA
ghost_error_t dd_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t ds_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t dc_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t dz_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t sd_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t ss_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t sc_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t sz_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t cd_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t cs_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t cc_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t cz_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t zd_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t zs_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t zc_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
ghost_error_t zz_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
#endif
ghost_error_t d_SELL_fromCRS(ghost_mat_t *mat, ghost_mat_t *crs);
ghost_error_t s_SELL_fromCRS(ghost_mat_t *mat, ghost_mat_t *crs);
ghost_error_t c_SELL_fromCRS(ghost_mat_t *mat, ghost_mat_t *crs);
ghost_error_t z_SELL_fromCRS(ghost_mat_t *mat, ghost_mat_t *crs);

const char * d_SELL_stringify(ghost_mat_t *mat, int dense);
const char * s_SELL_stringify(ghost_mat_t *mat, int dense);
const char * c_SELL_stringify(ghost_mat_t *mat, int dense);
const char * z_SELL_stringify(ghost_mat_t *mat, int dense);
#ifdef __cplusplus
}
#endif

#endif
