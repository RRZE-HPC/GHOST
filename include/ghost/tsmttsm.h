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
#include "math.h"

typedef struct
{
    ghost_alignment alignment;
    ghost_datatype dt;
    int wcols;
    int vcols;
    ghost_implementation impl;
    ghost_densemat_storage wstor;
    int unroll;
} ghost_tsmttsm_parameters;

typedef ghost_error (*ghost_tsmttsm_kernel)(ghost_densemat *, ghost_densemat *, ghost_densemat *, void *, void *, int);


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
     * @param[in] context In which context to do the reduction if one is specified.
     * @param[in] conjv If v should be conjugated, set this to != 1.
     * @param[in] flags Flags. Currently, they are only checked for ::GHOST_GEMM_KAHAN.
     *
     *
     * \f$ x = \alpha \cdot v^T \cdot w + \beta \cdot x \f$.
     *
     * v is NxM, distributed.
     *
     * w is NxK, distributed.
     *
     * x is NxK, redundant.
     *
     * M<<N
     * 
     * This kernel is auto-generated at compile time for given values of K and M.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_tsmttsm(ghost_densemat *x, ghost_densemat *v, ghost_densemat *w, void *alpha, void *beta, int reduce, int conjv,ghost_gemm_flags flags);
    
    /**
     * @brief Check whether TSMTTSM can be applied instead of GEMM with the given arguments.
     *
     * @param x
     * @param v
     * @param transv
     * @param w
     * @param transw
     * @param alpha
     * @param beta
     * @param reduce
     * @param flags
     * @param printerror Print an error message if application is not possible.
     *
     * @return 
     */
    ghost_error ghost_tsmttsm_valid(ghost_densemat *x, ghost_densemat *v, const char * transv, 
        ghost_densemat *w, const char *transw, void *alpha, void *beta, int reduce, ghost_gemm_flags flags, int printerror);

#ifdef __cplusplus
}
#endif
#endif

