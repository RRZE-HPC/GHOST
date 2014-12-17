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
    int blocksz;
} ghost_tsmm_inplace_parameters_t;

/**
 * @brief A tsmm kernel function. 
 */
typedef ghost_error_t (*ghost_tsmm_inplace_kernel_t)(ghost_densemat_t *, ghost_densemat_t *, void *);


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
     *
     *
     * Compute \f$ x = \alpha \cdot x \cdot w \f$.
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
    ghost_error_t ghost_tsmm_inplace(ghost_densemat_t *x, ghost_densemat_t *w, void *alpha);
    /**
     * @brief Generate the map of auto-generated tsmm kernels.
     */
    void ghost_tsmm_inplace_kernelmap_generate();
    /**
     * @brief Get the auto-generated tsmm kernel which fits the given parameters or, if not found, a fallback kernel. 
     *
     * @param[in] p The tsmm kernel parameters
     * @param[in] x The densemat x
     * @param[in] v The densemat v
     * @param[in] w The densemat w
     * @param[in] reduce The reduce argument which may comm from a ghost_gemm() call.
     *
     * @return The according kernel or NULL if no suiting kernel found. 
     */
    ghost_tsmm_inplace_kernel_t ghost_tsmm_inplace_kernel(ghost_tsmm_inplace_parameters_t p, ghost_densemat_t *x, ghost_densemat_t *v, ghost_densemat_t *w, int reduce);

#ifdef __cplusplus
}
#endif
#endif
