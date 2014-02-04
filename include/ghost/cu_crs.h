#ifndef GHOST_CU_CRS_H
#define GHOST_CU_CRS_H

#include "config.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

void ghost_cu_crsspmv(ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options);

#ifdef __cplusplus
}
#endif

#endif
