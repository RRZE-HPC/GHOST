/**
 * @file tsmtspmtsm.h
 * @brief A fused x = V^T x A x W operation
 * @author Dominik Ernst <dominik.ernst@fau.de>
 */
#ifndef GHOST_TSMTSPMTSM_H
#define GHOST_TSMTSPMTSM_H

#include "config.h"
#include "types.h"
#include "densemat.h"
#include "math.h"


#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup globops
 *
 * @brief Multiply a transposed distributed dense tall skinny matrix with a sparse matrix and
 * another distributed dense tall skinny matrix and Allreduce the result.
 *
 * @param[inout] x
 * @param[in] v
 * @param[in] w
 * @param[in] A
 * @param[in] alpha
 * @param[in] beta
 *
 *
 * \f$ x =  \alpha \cdot v^T \cdot A \cdot w + \beta \cdot x \f$.
 *
 * x is MxN, redundant.
 *
 * v is KxM, distributed.
 *
 * w is KxM, distributed.
 *
 * A is a KxK sparse matrix
 *
 * M,N << K
 *
 * alpha and beta are pointers to values in host memory, that have the same data type as x
 *
 * The data types of x, v, w and A have to be the same, except for the special case that v, w, and A
 * are float, and x is double. In that case, all calculations are peformed in double precision.
 *
 * This kernel is auto-generated at compile time for given values of M and N.
 *
 * @return ::GHOST_SUCCESS on success or an error indicator.
 */
ghost_error ghost_tsmtspmtsm(ghost_densemat *x, ghost_densemat *v, ghost_densemat *w,
    ghost_sparsemat *A, void *alpha, void *beta, int reduce);


#ifdef __cplusplus
}
#endif
#endif
