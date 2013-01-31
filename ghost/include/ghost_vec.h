#ifndef __GHOST_VEC_H__
#define __GHOST_VEC_H__

#include "ghost.h"

#ifdef MIC
//#define BJDS_LEN 8
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

void         ghost_zeroVector(ghost_vec_t *vec);
ghost_vec_t *ghost_newVector( const int nrows, unsigned int flags );
void         ghost_swapVectors(ghost_vec_t *v1, ghost_vec_t *v2);
void         ghost_normalizeVector( ghost_vec_t *vec);
ghost_vec_t *ghost_distributeVector(ghost_comm_t *lcrp, ghost_vec_t *vec);
void         ghost_collectVectors(ghost_context_t *context, ghost_vec_t *vec,	ghost_vec_t *totalVec, int kernel);
void         ghost_freeVector( ghost_vec_t* const vec );
void         ghost_permuteVector( ghost_vdat_t* vec, ghost_vidx_t* perm, ghost_vidx_t len);
int          ghost_vecEquals(ghost_vec_t *a, ghost_vec_t *b, double tol);
ghost_vec_t *ghost_cloneVector(ghost_vec_t *src);

#endif
