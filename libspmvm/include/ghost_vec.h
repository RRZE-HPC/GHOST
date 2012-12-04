#ifndef __GHOST_VEC__H__
#define __GHOST_VEC__H__

#include "ghost.h"


void         ghost_zeroVector(ghost_vec_t *vec);
ghost_vec_t *ghost_newVector( const int nrows, unsigned int flags );
void         ghost_swapVectors(ghost_vec_t *v1, ghost_vec_t *v2);
void         ghost_normalizeVector( ghost_vec_t *vec);
ghost_vec_t *ghost_distributeVector(ghost_comm_t *lcrp, ghost_vec_t *vec);
void         ghost_collectVectors(ghost_setup_t *setup, ghost_vec_t *vec,	ghost_vec_t *totalVec, int kernel);
void         ghost_freeVector( ghost_vec_t* const vec );
void         ghost_permuteVector( ghost_mdat_t* vec, mat_idx_t* perm, mat_idx_t len);
int          ghost_vecEquals(ghost_vec_t *a, ghost_vec_t *b, double tol);

#endif
