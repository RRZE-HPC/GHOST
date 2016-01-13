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

ghost_error_t ghost_densemat_cu_cm_dotprod(ghost_densemat_t *vec, void *res, ghost_densemat_t *vec2);
ghost_error_t ghost_densemat_cu_cm_vaxpy(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a);
ghost_error_t ghost_densemat_cu_cm_vaxpby(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a, void *b);
ghost_error_t ghost_densemat_cu_cm_vaxpbypcz(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a, void *b, ghost_densemat_t *v3, void *c);
ghost_error_t ghost_densemat_cu_cm_axpy(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a);
ghost_error_t ghost_densemat_cu_cm_axpby(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a, void *b);
ghost_error_t ghost_densemat_cu_cm_axpbypcz(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a, void *b, ghost_densemat_t *v3, void *c);
ghost_error_t ghost_densemat_cu_cm_scale(ghost_densemat_t *v, void *a);
ghost_error_t ghost_densemat_cu_cm_vscale(ghost_densemat_t *v, void *a);
ghost_error_t ghost_densemat_cu_cm_fromScalar(ghost_densemat_t *vec, void *val);
ghost_error_t ghost_densemat_cu_cm_fromRand(ghost_densemat_t *vec);
ghost_error_t ghost_densemat_cu_cm_communicationassembly(void * work, ghost_lidx_t *dueptr, ghost_lidx_t totaldues, ghost_densemat_t *vec, ghost_gidx_t *perm);
ghost_error_t ghost_densemat_cu_cm_conj(ghost_densemat_t *vec);


#ifdef __cplusplus
}
#endif

#endif
