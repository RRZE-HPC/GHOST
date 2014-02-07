#ifndef GHOST_CU_VEC_H
#define GHOST_CU_VEC_H

#include "config.h"
#include "types.h"
#include "vec.h"
#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

ghost_error_t ghost_vec_cu_dotprod(ghost_vec_t *vec, ghost_vec_t *vec2, void *res);
ghost_error_t ghost_vec_cu_vaxpy(ghost_vec_t *v1, ghost_vec_t *v2, void *a);
ghost_error_t ghost_vec_cu_vaxpby(ghost_vec_t *v1, ghost_vec_t *v2, void *a, void *b);
ghost_error_t ghost_vec_cu_axpy(ghost_vec_t *v1, ghost_vec_t *v2, void *a);
ghost_error_t ghost_vec_cu_axpby(ghost_vec_t *v1, ghost_vec_t *v2, void *a, void *b);
ghost_error_t ghost_vec_cu_scale(ghost_vec_t *v, void *a);
ghost_error_t ghost_vec_cu_vscale(ghost_vec_t *v, void *a);
ghost_error_t ghost_vec_cu_fromScalar(ghost_vec_t *vec, void *val);
ghost_error_t ghost_vec_cu_fromRand(ghost_vec_t *vec);


#ifdef __cplusplus
}
#endif

#endif
