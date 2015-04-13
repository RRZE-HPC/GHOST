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
#include "math.h"

typedef struct
{
    /**
     * @brief The data type of the densemats.
     */
    ghost_datatype_t dt;
    /**
     * @brief The first configured block size K.
     */
    int xcols;
    /**
     * @brief The second configure block size M.
     */
    int vcols;
    ghost_implementation_t impl;
    ghost_alignment_t alignment;
    ghost_densemat_storage_t xstor;
    ghost_densemat_storage_t wstor;
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
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_tsmm(ghost_densemat_t *x, ghost_densemat_t *v, ghost_densemat_t *w, void *alpha, void *beta);

    /**
     * @brief Check whether TSMM can be applied instead of GEMM with the given arguments.
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
    ghost_error_t ghost_tsmm_valid(ghost_densemat_t *x, ghost_densemat_t *v, const char * transv, 
        ghost_densemat_t *w, const char *transw, void *alpha, void *beta, int reduce, int printerror);

    int ghost_tsmm_perf_GBs(double *perf, double time, void *varg);
    int ghost_tsmm_perf_GFs(double *perf, double time, void *varg);

#ifdef __cplusplus
}
#endif
#endif
