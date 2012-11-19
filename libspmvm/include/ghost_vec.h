#ifndef __GHOST_VEC_H__
#define __GHOST_VEC_H__

#include "spmvm.h"


void         SpMVM_zeroVector(ghost_vec_t *vec);
ghost_vec_t *SpMVM_newVector( const int nrows, unsigned int flags );
void         SpMVM_swapVectors(ghost_vec_t *v1, ghost_vec_t *v2);
void         SpMVM_normalizeVector( ghost_vec_t *vec);
ghost_vec_t *SpMVM_distributeVector(ghost_comm_t *lcrp, ghost_vec_t *vec);
void         SpMVM_collectVectors(ghost_setup_t *setup, ghost_vec_t *vec,	ghost_vec_t *totalVec, int kernel);
void         SpMVM_freeVector( ghost_vec_t* const vec );
void         SpMVM_permuteVector( mat_data_t* vec, mat_idx_t* perm, mat_idx_t len);

#endif
