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
    ghost_datatype dt;
    /**
     * @brief The first configured block size K.
     */
    int xcols;
    /**
     * @brief The second configure block size M.
     */
    int vcols;
    ghost_implementation impl;
    ghost_alignment alignment;
    ghost_densemat_storage xstor;
    ghost_densemat_storage wstor;
    int unroll;
} ghost_tsmm_parameters;

/**
 * @brief A tsmm kernel function. 
 */
typedef ghost_error (*ghost_tsmm_kernel)(ghost_densemat *, ghost_densemat *, ghost_densemat *, void *, void *);


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
     * v is NxM, distributed.
     *
     * w is MxK, redundant.
     *
     * x is NxK, distributed.
     *
     * M<<N
     *
     * This kernel is auto-generated at compile time for given values of K and M.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_tsmm(ghost_densemat *x, ghost_densemat *v, ghost_densemat *w, void *alpha, void *beta);

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
    ghost_error ghost_tsmm_valid(ghost_densemat *x, ghost_densemat *v, const char * transv, 
        ghost_densemat *w, const char *transw, void *alpha, void *beta, int reduce, int printerror);

    int ghost_tsmm_perf_GBs(double *perf, double time, void *varg);
    int ghost_tsmm_perf_GFs(double *perf, double time, void *varg);

#ifdef __cplusplus
}
#endif
#endif
