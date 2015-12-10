/**
 * @file cu_densemat_rm.h
 * @brief Functions for row-major dense matrices/vectors with CUDA.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_CU_DENSEMAT_RM_H
#define GHOST_CU_DENSEMAT_RM_H

#include "config.h"
#include "types.h"
#include "densemat.h"
#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

ghost_error_t ghost_densemat_cu_rm_dotprod(ghost_densemat_t *vec, void *res, ghost_densemat_t *vec2);
ghost_error_t ghost_densemat_cu_rm_vaxpy(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a);
ghost_error_t ghost_densemat_cu_rm_vaxpby(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a, void *b);
ghost_error_t ghost_densemat_cu_rm_vaxpbypcz(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a, void *b, ghost_densemat_t *v3, void *c);
ghost_error_t ghost_densemat_cu_rm_axpy(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a);
ghost_error_t ghost_densemat_cu_rm_axpby(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a, void *b);
ghost_error_t ghost_densemat_cu_rm_axpbypcz(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a, void *b, ghost_densemat_t *v3, void *c);
ghost_error_t ghost_densemat_cu_rm_scale(ghost_densemat_t *v, void *a);
ghost_error_t ghost_densemat_cu_rm_vscale(ghost_densemat_t *v, void *a);
ghost_error_t ghost_densemat_cu_rm_fromScalar(ghost_densemat_t *vec, void *val);
ghost_error_t ghost_densemat_cu_rm_fromRand(ghost_densemat_t *vec);
ghost_error_t ghost_densemat_cu_rm_communicationassembly(void * work, ghost_lidx_t *dueptr, ghost_lidx_t totaldues, ghost_densemat_t *vec, ghost_lidx_t *perm);
ghost_error_t ghost_densemat_cu_rm_conj(ghost_densemat_t *vec);


#ifdef __cplusplus
}
#endif

#endif
