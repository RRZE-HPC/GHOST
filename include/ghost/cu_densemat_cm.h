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

ghost_error ghost_densemat_cu_cm_dotprod(ghost_densemat *vec, void *res, ghost_densemat *vec2);
ghost_error ghost_densemat_cu_cm_vaxpy(ghost_densemat *v1, ghost_densemat *v2, void *a);
ghost_error ghost_densemat_cu_cm_vaxpby(ghost_densemat *v1, ghost_densemat *v2, void *a, void *b);
ghost_error ghost_densemat_cu_cm_vaxpbypcz(ghost_densemat *v1, ghost_densemat *v2, void *a, void *b, ghost_densemat *v3, void *c);
ghost_error ghost_densemat_cu_cm_axpy(ghost_densemat *v1, ghost_densemat *v2, void *a);
ghost_error ghost_densemat_cu_cm_axpby(ghost_densemat *v1, ghost_densemat *v2, void *a, void *b);
ghost_error ghost_densemat_cu_cm_axpbypcz(ghost_densemat *v1, ghost_densemat *v2, void *a, void *b, ghost_densemat *v3, void *c);
ghost_error ghost_densemat_cu_cm_scale(ghost_densemat *v, void *a);
ghost_error ghost_densemat_cu_cm_vscale(ghost_densemat *v, void *a);
ghost_error ghost_densemat_cu_cm_fromScalar(ghost_densemat *vec, void *val);
ghost_error ghost_densemat_cu_cm_fromRand(ghost_densemat *vec);
ghost_error ghost_densemat_cu_cm_communicationassembly(void * work, ghost_lidx *dueptr, ghost_lidx totaldues, ghost_densemat *vec, ghost_context *ctx, ghost_gidx *perm);
ghost_error ghost_densemat_cu_cm_conj(ghost_densemat *vec);


#ifdef __cplusplus
}
#endif

#endif
