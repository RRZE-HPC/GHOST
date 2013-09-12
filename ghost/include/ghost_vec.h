#ifndef __GHOST_VEC_H__
#define __GHOST_VEC_H__

#include "ghost.h"

#ifdef MIC
//#define SELL_LEN 8
#define VEC_PAD 16
#elif defined (AVX)
#define VEC_PAD 4 // TODO single/double precision
#elif defined (SSE)
#define VEC_PAD 2
#elif defined (OPENCL) || defined (CUDA)
#define VEC_PAD 256
#elif defined (VSX)
#define VEC_PAD 2
#else
#define VEC_PAD 16
#endif

#define VAL(vec,k) ((char *)(vec->val))[(k)*ghost_sizeofDataType(vec->traits->datatype)]

ghost_vec_t *ghost_createVector(ghost_context_t *ctx, ghost_vtraits_t *traits);

#ifdef __cplusplus
template <typename v_t> void ghost_normalizeVector_tmpl(ghost_vec_t *vec);
template <typename v_t> void ghost_vec_dotprod_tmpl(ghost_vec_t *vec, ghost_vec_t *vec2, void *res);
template <typename v_t> void ghost_vec_axpy_tmpl(ghost_vec_t *vec, ghost_vec_t *vec2, void *);
template <typename v_t> void ghost_vec_axpby_tmpl(ghost_vec_t *vec, ghost_vec_t *vec2, void *, void *);
template<typename v_t> void ghost_vec_scale_tmpl(ghost_vec_t *vec, void *scale);
template <typename v_t> void ghost_vec_fromRand_tmpl(ghost_vec_t *vec);
template<typename v_t> int ghost_vecEquals_tmpl(ghost_vec_t *a, ghost_vec_t *b);

extern "C" {
#endif
void getNrowsFromContext(ghost_vec_t *vec);
void vec_malloc(ghost_vec_t *vec);
void d_ghost_normalizeVector(ghost_vec_t *vec); 
void s_ghost_normalizeVector(ghost_vec_t *vec); 
void z_ghost_normalizeVector(ghost_vec_t *vec);
void c_ghost_normalizeVector(ghost_vec_t *vec);
void d_ghost_vec_dotprod(ghost_vec_t *vec1, ghost_vec_t *vec2, void *res); 
void s_ghost_vec_dotprod(ghost_vec_t *vec1, ghost_vec_t *vec2, void *res); 
void z_ghost_vec_dotprod(ghost_vec_t *vec1, ghost_vec_t *vec2, void *res);
void c_ghost_vec_dotprod(ghost_vec_t *vec1, ghost_vec_t *vec2, void *res);
void d_ghost_vec_scale(ghost_vec_t *vec1, void *scale); 
void s_ghost_vec_scale(ghost_vec_t *vec1, void *scale); 
void z_ghost_vec_scale(ghost_vec_t *vec1, void *scale);
void c_ghost_vec_scale(ghost_vec_t *vec1, void *scale);
void d_ghost_vec_axpy(ghost_vec_t *vec1, ghost_vec_t *vec2, void *); 
void s_ghost_vec_axpy(ghost_vec_t *vec1, ghost_vec_t *vec2, void *); 
void z_ghost_vec_axpy(ghost_vec_t *vec1, ghost_vec_t *vec2, void *);
void c_ghost_vec_axpy(ghost_vec_t *vec1, ghost_vec_t *vec2, void *);
void d_ghost_vec_axpby(ghost_vec_t *vec1, ghost_vec_t *vec2, void *, void *); 
void s_ghost_vec_axpby(ghost_vec_t *vec1, ghost_vec_t *vec2, void *, void *); 
void z_ghost_vec_axpby(ghost_vec_t *vec1, ghost_vec_t *vec2, void *, void *);
void c_ghost_vec_axpby(ghost_vec_t *vec1, ghost_vec_t *vec2, void *, void *);
void d_ghost_vec_fromRand(ghost_vec_t *vec); 
void s_ghost_vec_fromRand(ghost_vec_t *vec); 
void z_ghost_vec_fromRand(ghost_vec_t *vec); 
void c_ghost_vec_fromRand(ghost_vec_t *vec); 
int d_ghost_vecEquals(ghost_vec_t *vec, ghost_vec_t *vec2); 
int s_ghost_vecEquals(ghost_vec_t *vec, ghost_vec_t *vec2); 
int z_ghost_vecEquals(ghost_vec_t *vec, ghost_vec_t *vec2); 
int c_ghost_vecEquals(ghost_vec_t *vec, ghost_vec_t *vec2); 
#ifdef __cplusplus
}
#endif

#endif
