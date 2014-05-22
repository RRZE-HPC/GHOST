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

typedef ghost_error_t (*tsmttsm_kernel)(ghost_densemat_t *, ghost_densemat_t *, ghost_densemat_t *, void *, void *);


#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @brief 
     *
     * @param x
     * @param v
     * @param w
     * @param alpha
     *
     * x = alpha*v*w
     * v is NxM, distributed, row-major
     * w is MxK, redundant, col-mjaor
     * x is NxK, distributed, row-major
     * M<<N
     * K=4,8,...
     *
     * @return 
     */
    ghost_error_t ghost_tsmttsm(ghost_densemat_t *x, ghost_densemat_t *v, ghost_densemat_t *w, void *alpha, void *beta);
    void ghost_tsmttsm_kernelmap_generate();
    tsmttsm_kernel ghost_tsmttsm_kernel(ghost_tsmttsm_parameters_t p);

#ifdef __cplusplus
}
#endif
#endif

