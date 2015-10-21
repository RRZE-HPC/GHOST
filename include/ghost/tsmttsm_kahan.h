/**
 * @file tsmttsm_kahan.h
 * @brief The specialized GEMM function Kahan-TSMTTSM.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_TSMTTSM_KAHAN_H
#define GHOST_TSMTTSM_KAHAN_H

#include "config.h"
#include "types.h"
#include "densemat.h"
#include "math.h"

typedef struct
{
    ghost_alignment_t alignment;
    ghost_datatype_t dt;
    int wcols;
    int vcols;
    ghost_implementation_t impl;
    ghost_densemat_storage_t xstor;
    ghost_densemat_storage_t wstor;
} ghost_tsmttsm_kahan_parameters_t;

typedef ghost_error_t (*ghost_tsmttsm_kahan_kernel_t)(ghost_densemat_t *, ghost_densemat_t *, ghost_densemat_t *, void *, void *, int);


#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @ingroup globops
     *
     * @brief Multiple a transposed distributed dense tall skinny matrix with another distributed dense tall skinny matrix and Allreduce the result.
     *
     * @param[inout] x
     * @param[in] v
     * @param[in] w
     * @param[in] alpha
     * @param[in] beta
     * @param[in] reduce
     * @param[in] conjv If v should be conjugated, set this to != 1.
     *
     *
     * \f$ x = \alpha \cdot v^T \cdot w + \beta \cdot x \f$.
     *
     * v is NxM, distributed, row-major.
     *
     * w is NxK, distributed, row-major.
     *
     * x is NxK, distributed, row- or col-major.
     *
     * M<<N
     * 
     * This kernel is auto-generated at compile time for given values of K and M.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_tsmttsm_kahan(ghost_densemat_t *x, ghost_densemat_t *v, ghost_densemat_t *w, void *alpha, void *beta, int reduce, int conjv);

    /**
     * @brief Check whether Kahan-TSMTTSM can be applied instead of GEMM with the given arguments.
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
    ghost_error_t ghost_tsmttsm_kahan_valid(ghost_densemat_t *x, ghost_densemat_t *v, const char * transv, 
        ghost_densemat_t *w, const char *transw, void *alpha, void *beta, int reduce, int printerror);

#ifdef __cplusplus
}
#endif
#endif

