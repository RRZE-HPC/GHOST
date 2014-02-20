#ifndef GHOST_SELL_H
#define GHOST_SELL_H

#include "config.h"
#include "types.h"
#include "sparsemat.h"

#define GHOST_SELL_CHUNKHEIGHT_ELLPACK 0
#define GHOST_SELL_CHUNKHEIGHT_AUTO -1

typedef struct 
{
    char * val;
    ghost_idx_t * col;
    ghost_idx_t * rowLen;
    ghost_idx_t * rowLenPadded;
    ghost_nnz_t * chunkStart;
    ghost_idx_t * chunkLen;
/*    ghost_idx_t nrows;
    ghost_idx_t nrowsPadded;
    int T; // number of threads per row (if applicable)
    ghost_idx_t chunkHeight;*/
}
ghost_cu_sell_t;

typedef struct
{
    int C;
    int T;
} 
ghost_sell_aux_t;

extern const ghost_sell_aux_t GHOST_SELL_AUX_INITIALIZER;

/**
 * @brief Struct defining a SELL-C-sigma-T matrix.
 */
typedef struct 
{
    char *val;
    ghost_idx_t *col;
    ghost_nnz_t *chunkStart;
    double beta; // chunk occupancy
    double variance; // row length variance
    double deviation; // row lenght standard deviation
    double cv; // row lenght coefficient of variation
    int T; // number of threads per row (if applicable)
    ghost_idx_t *chunkMin; // for version with remainder loop
    ghost_idx_t *chunkLen; // for version with remainder loop
    ghost_idx_t *chunkLenPadded; // for version with remainder loop
    ghost_idx_t *rowLen;   // for version with remainder loop
    ghost_idx_t *rowLenPadded; // for SELL-T 
    ghost_idx_t maxRowLen;
    ghost_idx_t nMaxRows;
    ghost_idx_t chunkHeight;
    ghost_idx_t scope;
    
    ghost_cu_sell_t *cumat;
}
ghost_sell_t;


#define SELL(mat) ((ghost_sell_t *)(mat->data))

#define SELL_CUDA_THREADSPERBLOCK 256

enum GHOST_SELL_C {GHOST_SELL_C_1 = 1, GHOST_SELL_C_2 = 2, GHOST_SELL_C_4 = 4, GHOST_SELL_C_256 = 256};
enum GHOST_SELL_T {GHOST_SELL_T_1 = 1, GHOST_SELL_T_2 = 2, GHOST_SELL_T_4 = 4, GHOST_SELL_T_256 = 256};

ghost_error_t ghost_SELL_init(ghost_sparsemat_t *mat);
#ifdef __cplusplus
template<typename, typename, int> void SELL_kernel_plain_tmpl(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, int);
template<typename, typename> void SELL_kernel_plain_ELLPACK_tmpl(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, int);
template <typename> ghost_error_t SELL_fromCRS(ghost_sparsemat_t *, ghost_sparsemat_t *);
extern "C" {
#endif
ghost_error_t dd_SELL_kernel_plain(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t ds_SELL_kernel_plain(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t dc_SELL_kernel_plain(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t dz_SELL_kernel_plain(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t sd_SELL_kernel_plain(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t ss_SELL_kernel_plain(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t sc_SELL_kernel_plain(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t sz_SELL_kernel_plain(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t cd_SELL_kernel_plain(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t cs_SELL_kernel_plain(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t cc_SELL_kernel_plain(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t cz_SELL_kernel_plain(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t zd_SELL_kernel_plain(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t zs_SELL_kernel_plain(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t zc_SELL_kernel_plain(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t zz_SELL_kernel_plain(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);

ghost_error_t dd_SELL_kernel_SSE(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t dd_SELL_kernel_AVX(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t dd_SELL_kernel_AVX_32(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t dd_SELL_kernel_AVX_32_rich(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t dd_SELL_kernel_MIC_16(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t dd_SELL_kernel_MIC_32(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
#ifdef GHOST_HAVE_CUDA
ghost_error_t dd_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t ds_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t dc_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t dz_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t sd_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t ss_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t sc_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t sz_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t cd_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t cs_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t cc_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t cz_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t zd_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t zs_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t zc_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
ghost_error_t zz_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
#endif
ghost_error_t d_SELL_fromCRS(ghost_sparsemat_t *mat, ghost_sparsemat_t *crs);
ghost_error_t s_SELL_fromCRS(ghost_sparsemat_t *mat, ghost_sparsemat_t *crs);
ghost_error_t c_SELL_fromCRS(ghost_sparsemat_t *mat, ghost_sparsemat_t *crs);
ghost_error_t z_SELL_fromCRS(ghost_sparsemat_t *mat, ghost_sparsemat_t *crs);

const char * d_SELL_stringify(ghost_sparsemat_t *mat, int dense);
const char * s_SELL_stringify(ghost_sparsemat_t *mat, int dense);
const char * c_SELL_stringify(ghost_sparsemat_t *mat, int dense);
const char * z_SELL_stringify(ghost_sparsemat_t *mat, int dense);
#ifdef __cplusplus
}
#endif

#endif
