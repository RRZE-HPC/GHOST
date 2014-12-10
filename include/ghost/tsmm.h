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
    /**
     * @brief The data type of the densemats.
     */
    ghost_datatype_t dt;
    /**
     * @brief The first configured block size K.
     */
    int blocksz1;
    /**
     * @brief The second configure block size M.
     */
    int blocksz2;
} ghost_tsmm_parameters_t;

/**
 * @brief A tsmm kernel function. 
 */
typedef ghost_error_t (*ghost_tsmm_kernel_t)(ghost_densemat_t *, ghost_densemat_t *, ghost_densemat_t *, void *, void *);


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
     * @param[in] beta
     *
     *
     * Compute \f$ x = \alpha \cdot v \cdot w  + \beta \cdot x\f$.
     *
     * v is NxM, distributed, row-major.
     *
     * w is MxK, redundant, col-major.
     *
     * x is NxK, distributed, row-major.
     *
     * M<<N
     *
     * This kernel is auto-generated at compile time for given values of K and M.
     * Additionally, a version for given K but arbitrary M is being generated.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_tsmm(ghost_densemat_t *x, ghost_densemat_t *v, ghost_densemat_t *w, void *alpha, void *beta);
    /**
     * @brief Generate the map of auto-generated tsmm kernels.
     */
    void ghost_tsmm_kernelmap_generate();
    /**
     * @brief Get the auto-generated tsmm kernel which fits the given parameters or, if not found, a fallback kernel. 
     *
     * @param[in] p The tsmm kernel parameters
     *
     * @return The according kernel. 
     */
    ghost_tsmm_kernel_t ghost_tsmm_kernel(ghost_tsmm_parameters_t p, ghost_densemat_t *x, ghost_densemat_t *v, ghost_densemat_t *w, int reduce);

#ifdef __cplusplus
}
#endif
#endif
