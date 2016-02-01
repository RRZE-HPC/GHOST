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

/**
 * @brief A CUDA SELL-C-sigma matrix.
 */
typedef struct 
{
    /**
     * @brief The values.
     */
    char * val;
    /**
     * @brief The column indices.
     */
    ghost_lidx * col;
    /**
     * @brief The length of each row.
     */
    ghost_lidx * rowLen;
    /**
     * @brief Needed if T>1.
     */
    ghost_lidx * rowLenPadded;
    /**
     * @brief Pointer to start of each chunk.
     */
    ghost_lidx * chunkStart;
    /**
     * @brief The length of each chunk.
     */
    ghost_lidx * chunkLen;
}
ghost_cu_sell;

/**
 * @brief Struct defining a SELL-C-sigma matrix.
 */
typedef struct 
{
    /**
     * @brief The values.
     */
    char *val;
    /**
     * @brief The column indices.
     */
    ghost_lidx *col;
    /**
     * @brief Pointer to start of each chunk.
     */
    ghost_lidx *chunkStart;
    /**
     * @brief Minimal row length in a chunk.
     */
    ghost_lidx *chunkMin;
    /**
     * @brief The length of each chunk.
     */
    ghost_lidx *chunkLen;
    /**
     * @brief Needed if T>1.
     */
    ghost_lidx *chunkLenPadded;
    /**
     * @brief Length of each row.
     *
     * Especially useful in SELL-1 kernels.
     */
    ghost_lidx *rowLen;
    /**
     * @brief Needed if T>1.
     */
    ghost_lidx *rowLenPadded; 
    /**
     * @brief The CUDA matrix.
     */
    ghost_cu_sell *cumat;
}
ghost_sell;

/**
 * @brief The parameters to identify a SELL SpMV kernel.
 *
 * On kernel execution, GHOST will try to find an auto-generated kernel which
 * matches all of these parameters.
 */
typedef struct 
{
    /**
     * @brief The data access alignment.
     */
    ghost_alignment alignment;
    /**
     * @brief The implementation.
     */
    ghost_implementation impl;
    /**
     * @brief The matrix data type.
     */
    ghost_datatype mdt;
    /**
     * @brief The densemat data type.
     */
    ghost_datatype vdt;
    /**
     * @brief The densemat width.
     */
    int blocksz;
    /**
     * @brief The SELL matrix chunk height.
     */
    int chunkheight;
    /**
     * @brief The densemat storage order.
     */
    ghost_densemat_storage storage;

}
ghost_sellspmv_parameters;

/**
 * @brief Get the SELL data of a general sparsemat.
 *
 * @param mat The sparsemat.
 *
 * @return Pointer to the SELL data.
 */
#define SELL(mat) ((ghost_sell *)(mat->data))

/**
 * @brief Create only a single chunk, i.e., use the ELLPACK storage format.
 */

#define GHOST_SELL_CHUNKHEIGHT_ELLPACK 0
/**
 * @brief A chunkheight should automatically be determined.
 */
#define GHOST_SELL_CHUNKHEIGHT_AUTO -1

/**
 * @brief Initialize a sparsemat as a SELL matrix.
 *
 * The major part of this function is setting the function pointers 
 * for all sparsemat functions to the corresponding SELL functions.
 *
 * @param mat The sparsemat.
 *
 * @return ::GHOST_SUCCESS on success or an error indicator.
 */
ghost_error ghost_sell_init(ghost_sparsemat *mat);

#ifdef __cplusplus
extern "C" {
#endif
    /**
     * @brief Select and call the right SELL SpMV kernel. 
     *
     * @param mat The matrix.
     * @param lhs The result densemat.
     * @param rhs The input densemat.
     * @param options Options to the SpMV.
     * @param argp The varargs. 
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_sell_spmv_selector(ghost_sparsemat *mat, 
            ghost_densemat *lhs, ghost_densemat *rhs, 
            ghost_spmv_flags options, va_list argp);
    
    /**
     * @brief Select and call the right SELL stringification function.
     *
     * @param mat The matrix.
     * @param str Where to store the string.
     * @param dense Print in a dense or sparse manner.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_sell_stringify_selector(ghost_sparsemat *mat, 
            char **str, int dense);
    
    /**
     * @brief Select and call the right CUDA SELL SpMV kernel. 
     *
     * @param mat The matrix.
     * @param lhs The result densemat.
     * @param rhs The input densemat.
     * @param flags Options to the SpMV.
     * @param argp The varargs. 
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_cu_sell_spmv_selector(ghost_sparsemat *mat, 
            ghost_densemat *lhs, ghost_densemat *rhs, 
            ghost_spmv_flags flags, va_list argp);

    ghost_error ghost_cu_sell1_spmv_selector(ghost_sparsemat *mat, ghost_densemat * lhs_in, ghost_densemat * rhs_in, ghost_spmv_flags options, va_list argp);

    /**
     * @brief Perform a Kaczmarz sweep with the SELL matrix. 
     *
     * @param mat The matrix.
     * @param lhs Output densemat.
     * @param rhs Input densemat.
     * @param omega The scaling factor omega.
     * @param forward 1 if forward, 0 if backward sweep should be done.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_sell_kacz(ghost_sparsemat *mat, ghost_densemat *lhs, 
            ghost_densemat *rhs, void *omega, int forward);

    /**
     * @brief Get the largest SELL chunk height of auto-generated kernels.
     *
     * @return The largest configured SELL chunk height or 0 if none has been 
     * configured.
     */
    int ghost_sell_max_cfg_chunkheight();
#ifdef __cplusplus
}
#endif

#endif
