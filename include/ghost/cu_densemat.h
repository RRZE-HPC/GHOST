/**
 * @file cu_densemat.h
 * @brief Functions for dense matrices/vectors with CUDA.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_CU_DENSEMAT_H
#define GHOST_CU_DENSEMAT_H

#include "config.h"
#include "types.h"
#include "densemat.h"
#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

ghost_error_t ghost_vec_cu_dotprod(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *res);
ghost_error_t ghost_vec_cu_vaxpy(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a);
ghost_error_t ghost_vec_cu_vaxpby(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a, void *b);
ghost_error_t ghost_vec_cu_axpy(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a);
ghost_error_t ghost_vec_cu_axpby(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a, void *b);
ghost_error_t ghost_vec_cu_scale(ghost_densemat_t *v, void *a);
ghost_error_t ghost_vec_cu_vscale(ghost_densemat_t *v, void *a);
ghost_error_t ghost_vec_cu_fromScalar(ghost_densemat_t *vec, void *val);
ghost_error_t ghost_vec_cu_fromRand(ghost_densemat_t *vec);


#ifdef __cplusplus
}z
#endif

#endif
