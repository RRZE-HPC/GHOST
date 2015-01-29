/**
 * @file tsmm_inplace.h
 * @brief The specialized GEMM function tsmm (in-place).
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_TSMM_INPLACE_H
#define GHOST_TSMM_INPLACE_H

#include "config.h"
#include "types.h"
#include "densemat.h"

typedef struct
{
    /**
     * @brief The data type of the densemats.
     */
    ghost_datatype_t dt;
    /**
     * @brief The first configured block size M.
     */
    int xcols;

    ghost_implementation_t impl;
} ghost_tsmm_inplace_parameters_t;

/**
 * @brief A tsmm-inplace kernel function. 
 */
typedef ghost_error_t (*ghost_tsmm_inplace_kernel_t)(ghost_densemat_t *, ghost_densemat_t *, void *, void *);


#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @ingroup locops
     *
     * @brief Multiply a distributed dense tall skinny matrix with a redundant dense matrix in-place. 
     *
     * @param[inout] x
     * @param[in] w
     * @param[in] alpha
     * @param[in] beta
     *
     *
     * Compute \f$ x = \alpha \cdot x \cdot w  + \beta \cdot x\f$.
     *
     * w is MxM, redundant, col-major.
     *
     * x is NxM, distributed, row-major.
     *
     * M<<N
     *
     * This kernel is auto-generated at compile time for given values of M.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_tsmm_inplace(ghost_densemat_t *x, ghost_densemat_t *w, void *alpha, void *beta);

    /**
     * @brief Check whether TSMM-inplace can be applied instead of GEMM with the given arguments.
     *
     * @param x
     * @param v
     * @param transv
     * @param w
     * @param transw
     * @param alpha
     * @param beta
     * @param reduce
     * @param printerror Print an error message if application is not possible.
     *
     * @return 
     */
    ghost_error_t ghost_tsmm_inplace_valid(ghost_densemat_t *x, ghost_densemat_t *v, const char * transv, 
        ghost_densemat_t *w, const char *transw, void *alpha, void *beta, int reduce, int printerror);

#ifdef __cplusplus
}
#endif
#endif
