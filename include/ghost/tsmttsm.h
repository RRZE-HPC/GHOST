/**
 * @file tsmttsm.h
 * @brief The specialized GEMM function tsmttsm.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_TSMTTSM_H
#define GHOST_TSMTTSM_H

#include "config.h"
#include "types.h"
#include "densemat.h"

typedef struct
{
    ghost_datatype_t dt;
    int blocksz;
} ghost_tsmttsm_parameters_t;

typedef ghost_error_t (*ghost_tsmttsm_kernel_t)(ghost_densemat_t *, ghost_densemat_t *, ghost_densemat_t *, void *, void *);


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
     *
     *
     * \f$ x = \alpha \cdot v^T \cdot w + \beta \cdot x \f$.
     *
     * v is NxM, distributed, row-major.
     *
     * w is MxK, distributed, row-major.
     *
     * x is NxK, distributed, row- or col-major.
     *
     * M<<N
     * 
     * This kernel is auto-generated at compile time for given values of K.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_tsmttsm(ghost_densemat_t *x, ghost_densemat_t *v, ghost_densemat_t *w, void *alpha, void *beta);
    void ghost_tsmttsm_kernelmap_generate();
    ghost_tsmttsm_kernel_t ghost_tsmttsm_kernel(ghost_tsmttsm_parameters_t p, ghost_densemat_t *x, ghost_densemat_t *v, ghost_densemat_t *w, int reduce);

#ifdef __cplusplus
}
#endif
#endif

