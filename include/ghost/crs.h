/**
 * @file crs.h
 * @brief Macros and functions for the CRS sparse matrix implementation.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_CRS_H
#define GHOST_CRS_H

#include "config.h"
#include "types.h"
#include "sparsemat.h"

#ifdef GHOST_HAVE_CUDA
#include "cu_crs.h"
#endif

/**
 * @brief Struct defining a CRS matrix on CUDA.
 */
typedef struct 
{
    ghost_lidx_t  *rpt;
    ghost_lidx_t  *col;
    char *val;
} 
ghost_cu_crs_t;

/**
 * @brief Struct defining a CRS matrix.
 */
typedef struct 
{
    ghost_lidx_t  *rpt;
    ghost_lidx_t  *col;
    char *val;
    
    ghost_cu_crs_t *cumat;
} 
ghost_crs_t;


#define CR(mat) ((ghost_crs_t *)((mat)->data))

ghost_error_t ghost_crs_init(ghost_sparsemat_t *mat);
#ifdef __cplusplus
extern "C" {
#endif
    
ghost_error_t CRS_kernel_plain_selector(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t options, va_list argp);
ghost_error_t CRS_stringify_selector(ghost_sparsemat_t *mat, char **str, int dense);

#ifdef __cplusplus
}
#endif


#endif

