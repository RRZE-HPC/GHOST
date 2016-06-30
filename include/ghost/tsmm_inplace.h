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
    ghost_datatype dt;
    /**
     * @brief The number of columns for the input densemat.
     */
    int ncolsin;
    /**
     * @brief The number of columns for the output densemat.
     */
    int ncolsout;

    ghost_implementation impl;
    ghost_alignment alignment;
} ghost_tsmm_inplace_parameters;

/**
 * @brief A tsmm-inplace kernel function. 
 */
typedef ghost_error (*ghost_tsmm_inplace_kernel)(ghost_densemat *, ghost_densemat *, void *, void *);


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
     * Compute \f$ x(:,1:K) = \alpha \cdot x(:,1:M) \cdot w  + \beta \cdot x(:,1:M)\f$.
     *
     * w is MxK, M>K, redundant, col-major.
     *
     * x is NxM, distributed, row-major.
     *
     * M,K<<N
     *
     * This kernel is auto-generated at compile time for given values of M and K.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_tsmm_inplace(ghost_densemat *x, ghost_densemat *w, void *alpha, void *beta);

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
    ghost_error ghost_tsmm_inplace_valid(ghost_densemat *x, ghost_densemat *v, const char * transv, 
        ghost_densemat *w, const char *transw, void *alpha, void *beta, int reduce, int printerror);

#ifdef __cplusplus
}
#endif
#endif
