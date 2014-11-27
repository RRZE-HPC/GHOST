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
    ghost_lidx_t * col;
    /**
     * @brief The length of each row.
     */
    ghost_lidx_t * rowLen;
    /**
     * @brief Needed if T>1.
     */
    ghost_lidx_t * rowLenPadded;
    /**
     * @brief Pointer to start of each chunk.
     */
    ghost_lidx_t * chunkStart;
    /**
     * @brief The length of each chunk.
     */
    ghost_lidx_t * chunkLen;
}
ghost_cu_sell_t;

/**
 * @brief Auxiliary information to the SELL format.
 */
typedef struct
{
    /**
     * @brief The chunk height.
     */
    int C;
    /**
     * @brief Number of threads per row.
     */
    int T;
} 
ghost_sell_aux_t;

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
    ghost_lidx_t *col;
    /**
     * @brief Pointer to start of each chunk.
     */
    ghost_lidx_t *chunkStart;
    /**
     * @brief Chunk occupancy.
     *
     * This gets computed at matrix construction.
     */
    double beta;
    /**
     * @brief Number of threads per row.
     */
    int T;
    /**
     * @brief Minimal row length in a chunk.
     */
    ghost_lidx_t *chunkMin;
    /**
     * @brief The length of each chunk.
     */
    ghost_lidx_t *chunkLen;
    /**
     * @brief Needed if T>1.
     */
    ghost_lidx_t *chunkLenPadded;
    /**
     * @brief Length of each row.
     *
     * Especially useful in SELL-1 kernels.
     */
    ghost_lidx_t *rowLen;
    /**
     * @brief Needed if T>1.
     */
    ghost_lidx_t *rowLenPadded; 
    /**
     * @brief The chunk height C.
     */
    ghost_lidx_t chunkHeight;
    /**
     * @brief The CUDA matrix.
     */
    ghost_cu_sell_t *cumat;
}
ghost_sell_t;


/**
 * @brief The parameters to identify a SELL SpMV kernel.
 *
 * On kernel execution, GHOST will try to find an auto-generated kernel which
 * matches all of these parameters.
 */
typedef struct 
{
    /**
     * @brief The implementation.
     */
    ghost_implementation_t impl;
    /**
     * @brief The matrix data type.
     */
    ghost_datatype_t mdt;
    /**
     * @brief The densemat data type.
     */
    ghost_datatype_t vdt;
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
    ghost_densemat_storage_t storage;

}
ghost_sellspmv_parameters_t;

/**
 * @brief Get the SELL data of a general sparsemat.
 *
 * @param mat The sparsemat.
 *
 * @return Pointer to the SELL data.
 */
#define SELL(mat) ((ghost_sell_t *)(mat->data))

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
ghost_error_t ghost_sell_init(ghost_sparsemat_t *mat);

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
    ghost_error_t ghost_sell_spmv_selector(ghost_sparsemat_t *mat, 
            ghost_densemat_t *lhs, ghost_densemat_t *rhs, 
            ghost_spmv_flags_t options, va_list argp);
    
    /**
     * @brief Select and call the right SELL stringification function.
     *
     * @param mat The matrix.
     * @param str Where to store the string.
     * @param dense Print in a dense or sparse manner.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_sell_stringify_selector(ghost_sparsemat_t *mat, 
            char **str, int dense);
    
    /**
     * @brief Select and call the right CUDA SELL SpMV kernel. 
     *
     * @param mat The matrix.
     * @param lhs The result densemat.
     * @param rhs The input densemat.
     * @param options Options to the SpMV.
     * @param argp The varargs. 
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_cu_sell_spmv_selector(ghost_sparsemat_t *mat, 
            ghost_densemat_t *lhs, ghost_densemat_t *rhs, 
            ghost_spmv_flags_t flags, va_list argp);

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
    ghost_error_t ghost_sell_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, 
            ghost_densemat_t *rhs, void *omega, int forward);
#ifdef __cplusplus
}
#endif

/**
 * @brief Initializer for SELL aux information.
 */
extern const ghost_sell_aux_t GHOST_SELL_AUX_INITIALIZER;

#endif
