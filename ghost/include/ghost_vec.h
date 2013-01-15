#ifndef __GHOST_VEC_H__
#define __GHOST_VEC_H__

#include "ghost.h"


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
