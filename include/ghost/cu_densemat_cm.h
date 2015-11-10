/**
 * @file cu_densemat_cm.h
 * @brief Functions for col-major dense matrices/vectors with CUDA.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_CU_DENSEMAT_CM_H
#define GHOST_CU_DENSEMAT_CM_H

#include "config.h"
#include "types.h"
#include "densemat.h"
#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

ghost_error_t ghost_densemat_cm_cu_dotprod(ghost_densemat_t *vec, void *res, ghost_densemat_t *vec2);
ghost_error_t ghost_densemat_cm_cu_vaxpy(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a);
ghost_error_t ghost_densemat_cm_cu_vaxpby(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a, void *b);
ghost_error_t ghost_densemat_cm_cu_vaxpbypcz(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a, void *b, ghost_densemat_t *v3, void *c);
ghost_error_t ghost_densemat_cm_cu_axpy(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a);
ghost_error_t ghost_densemat_cm_cu_axpby(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a, void *b);
ghost_error_t ghost_densemat_cm_cu_axpbypcz(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a, void *b, ghost_densemat_t *v3, void *c);
ghost_error_t ghost_densemat_cm_cu_scale(ghost_densemat_t *v, void *a);
ghost_error_t ghost_densemat_cm_cu_vscale(ghost_densemat_t *v, void *a);
ghost_error_t ghost_densemat_cm_cu_fromScalar(ghost_densemat_t *vec, void *val);
ghost_error_t ghost_densemat_cm_cu_fromRand(ghost_densemat_t *vec);
ghost_error_t ghost_densemat_cm_cu_communicationassembly(void * work, ghost_lidx_t *dueptr, ghost_densemat_t *vec, ghost_lidx_t *perm);
ghost_error_t ghost_densemat_cm_cu_conj(ghost_densemat_t *vec);


#ifdef __cplusplus
}
#endif

#endif
