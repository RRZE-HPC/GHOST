/**
 * @file tsmm.h
 * @brief The specialized GEMM function tsmm.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_TSMM_H
#define GHOST_TSMM_H

#include "config.h"
#include "types.h"
#include "densemat.h"

typedef struct
{
    ghost_datatype_t dt;
    int blocksz1;
    int blocksz2;
} ghost_tsmm_parameters_t;

typedef ghost_error_t (*tsmm_kernel)(ghost_densemat_t *, ghost_densemat_t *, ghost_densemat_t *, void *);


#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @ingroup locops
     *
     * @brief Multiply a distributed dense tall skinny matrix with a redundant dense matrix. 
     *
     * @param[inout] x
     * @param[in] v
     * @param[in] w
     * @param[in] alpha
     *
     *
     * Compute \f$ x = \alpha \cdot v \cdot w \f$.
     *
     * v is NxM, distributed, row-major.
     *
     * w is MxK, redundant, col-major.
     *
     * x is NxK, distributed, row-major.
     * M<<N
     *
     * This kernel is auto-generated at compile time for given values of K and M.
     * Additionally, a version for given K but arbitrary M is being generated.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_tsmm(ghost_densemat_t *x, ghost_densemat_t *v, ghost_densemat_t *w, void *alpha);
    void ghost_tsmm_kernelmap_generate();
    tsmm_kernel ghost_tsmm_kernel(ghost_tsmm_parameters_t p);

#ifdef __cplusplus
}
#endif
#endif
