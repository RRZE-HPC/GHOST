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

/**
 * @brief Struct defining a CRS matrix on CUDA.
 */
typedef struct 
{
    /**
     * @brief The row pointers.
     */
    ghost_lidx_t  *rpt;
    /**
     * @brief The column indices
     */
    ghost_lidx_t  *col;
    /**
     * @brief The values.
     */
    char *val;
} 
ghost_cu_crs_t;

/**
 * @brief Struct defining a CRS matrix.
 */
typedef struct 
{
    /**
     * @brief The row pointers.
     */
    ghost_lidx_t  *rpt;
    /**
     * @brief The column indices
     */
    ghost_lidx_t  *col;
    /**
     * @brief The values.
     */
    char *val;
    /**
     * @brief The CUDA matrix.
     */
    ghost_cu_crs_t *cumat;
} 
ghost_crs_t;

/**
 * @brief Get the CRS data of a general sparsemat.
 *
 * @param mat The sparsemat.
 *
 * @return Pointer to the CRS data.
 */
#define CR(mat) ((ghost_crs_t *)((mat)->data))

/**
 * @brief Initialize a sparsemat as a CRS matrix.
 *
 *
 * The major part of this function is setting the function pointers 
 * for all sparsemat functions to the corresponding CRS functions.
 *
 * @param mat The sparsemat.
 *
 * @return ::GHOST_SUCCESS on success or an error indicator.
 */
ghost_error_t ghost_crs_init(ghost_sparsemat_t *mat);

#ifdef __cplusplus
extern "C" {
#endif
    
    /**
     * @brief Select and call the right CRS SpMV kernel. 
     *
     * @param mat The matrix.
     * @param lhs The result densemat.
     * @param rhs The input densemat.
     * @param options Options to the SpMV.
     * @param argp The varargs. 
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_crs_spmv_selector(ghost_sparsemat_t *mat, 
            ghost_densemat_t *lhs, ghost_densemat_t *rhs, 
            ghost_spmv_flags_t options, va_list argp);
    
    /**
     * @brief Select and call the right CRS stringification function.
     *
     * @param mat The matrix.
     * @param str Where to store the string.
     * @param dense Print in a dense or sparse manner.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_crs_stringify_selector(ghost_sparsemat_t *mat, 
            char **str, int dense);

    /**
     * @brief Select and call the right CUDA CRS SpMV kernel. 
     *
     * @param mat The matrix.
     * @param lhs The result densemat.
     * @param rhs The input densemat.
     * @param options Options to the SpMV.
     * @param argp The varargs. 
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_cu_crs_spmv_selector(ghost_sparsemat_t *mat, 
            ghost_densemat_t * lhs, ghost_densemat_t * rhs, 
            ghost_spmv_flags_t options, va_list argp);

    /**
     * @brief Perform a Kaczmarz sweep with the CRS matrix. 
     *
     * @param mat The matrix.
     * @param lhs Output densemat.
     * @param rhs Input densemat.
     * @param omega The scaling factor omega.
     * @param forward 1 if forward, 0 if backward sweep should be done.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, 
            ghost_densemat_t *rhs, void *omega, int forward);

#ifdef __cplusplus
}
#endif


#endif

