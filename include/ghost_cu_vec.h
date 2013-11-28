#ifndef __GHOST_CU_VEC_H__
#define __GHOST_CU_VEC_H__

#include <ghost_config.h>
#include <ghost_types.h>

#ifdef __cplusplus
extern "C" {
#endif

void ghost_vec_cu_dotprod(ghost_vec_t *vec, ghost_vec_t *vec2, void *res);
void ghost_vec_cu_vaxpy(ghost_vec_t *v1, ghost_vec_t *v2, void *a);
void ghost_vec_cu_vaxpby(ghost_vec_t *v1, ghost_vec_t *v2, void *a, void *b);;
void ghost_vec_cu_axpy(ghost_vec_t *v1, ghost_vec_t *v2, void *a);
void ghost_vec_cu_axpby(ghost_vec_t *v1, ghost_vec_t *v2, void *a, void *b);;
void ghost_vec_cu_scale(ghost_vec_t *v, void *a);;
void ghost_vec_cu_vscale(ghost_vec_t *v, void *a);;

#ifdef __cplusplus
}
#endif

#endif
