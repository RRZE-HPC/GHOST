/**
 * @file sell.h
 * @brief Macros and functions for the SELL sparse matrix implementation.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
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
    ghost_lidx_t * col;
    ghost_lidx_t * rowLen;
    ghost_lidx_t * rowLenPadded;
    ghost_lidx_t * chunkStart;
    ghost_lidx_t * chunkLen;
}
ghost_cu_sell_t;

typedef struct
{
    int C;
    int T;
} 
ghost_sell_aux_t;

#define GHOST_SELL_AUX_INITIALIZER (ghost_sell_aux_t) {.C = 32, .T = 1};

/**
 * @brief Struct defining a SELL-C-sigma-T matrix.
 */
typedef struct 
{
    char *val;
    ghost_lidx_t *col;
    ghost_lidx_t *chunkStart;
    double beta; // chunk occupancy
    int T; // number of threads per row (if applicable)
    ghost_lidx_t *chunkMin; // for version with remainder loop
    ghost_lidx_t *chunkLen; // for version with remainder loop
    ghost_lidx_t *chunkLenPadded; // for version with remainder loop
    ghost_lidx_t *rowLen;   // for version with remainder loop
    ghost_lidx_t *rowLenPadded; // for SELL-T 
    ghost_lidx_t chunkHeight;
    
    ghost_cu_sell_t *cumat;
}
ghost_sell_t;


typedef struct 
{
    ghost_implementation_t impl;
    ghost_datatype_t mdt;
    ghost_datatype_t vdt;
    int blocksz;
    int chunkheight;
    ghost_densemat_storage_t storage;

}
ghost_sellspmv_parameters_t;
typedef ghost_error_t (*sellspmv_kernel)(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list);

#define SELL(mat) ((ghost_sell_t *)(mat->data))


ghost_error_t ghost_sell_init(ghost_sparsemat_t *mat);

#ifdef __cplusplus
extern "C" {
#endif
ghost_error_t ghost_sell_spmv_selector(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t options, va_list argp);
ghost_error_t ghost_sell_stringify_selector(ghost_sparsemat_t *mat, char **str, int dense);

#ifdef GHOST_HAVE_CUDA
ghost_error_t dd_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list);
ghost_error_t ds_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list);
ghost_error_t dc_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list);
ghost_error_t dz_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list);
ghost_error_t sd_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list);
ghost_error_t ss_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list);
ghost_error_t sc_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list);
ghost_error_t sz_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list);
ghost_error_t cd_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list);
ghost_error_t cs_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list);
ghost_error_t cc_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list);
ghost_error_t cz_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list);
ghost_error_t zd_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list);
ghost_error_t zs_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list);
ghost_error_t zc_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list);
ghost_error_t zz_SELL_kernel_CU(ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list);
#endif
#ifdef __cplusplus
}
#endif

#endif
