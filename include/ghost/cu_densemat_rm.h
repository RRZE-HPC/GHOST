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

ghost_error ghost_densemat_cu_rm_dotprod(ghost_densemat *vec, void *res, ghost_densemat *vec2);
ghost_error ghost_densemat_cu_rm_vaxpy(ghost_densemat *v1, ghost_densemat *v2, void *a);
ghost_error ghost_densemat_cu_rm_vaxpby(ghost_densemat *v1, ghost_densemat *v2, void *a, void *b);
ghost_error ghost_densemat_cu_rm_vaxpbypcz(ghost_densemat *v1, ghost_densemat *v2, void *a, void *b, ghost_densemat *v3, void *c);
ghost_error ghost_densemat_cu_rm_axpy(ghost_densemat *v1, ghost_densemat *v2, void *a);
ghost_error ghost_densemat_cu_rm_axpby(ghost_densemat *v1, ghost_densemat *v2, void *a, void *b);
ghost_error ghost_densemat_cu_rm_axpbypcz(ghost_densemat *v1, ghost_densemat *v2, void *a, void *b, ghost_densemat *v3, void *c);
ghost_error ghost_densemat_cu_rm_scale(ghost_densemat *v, void *a);
ghost_error ghost_densemat_cu_rm_vscale(ghost_densemat *v, void *a);
ghost_error ghost_densemat_cu_rm_fromScalar(ghost_densemat *vec, void *val);
ghost_error ghost_densemat_cu_rm_fromRand(ghost_densemat *vec);
ghost_error ghost_densemat_cu_rm_communicationassembly(void * work, ghost_lidx *dueptr, ghost_lidx totaldues, ghost_densemat *vec, ghost_context *ctx, ghost_lidx *perm);
ghost_error ghost_densemat_cu_rm_conj(ghost_densemat *vec);


#ifdef __cplusplus
}
#endif

#endif
