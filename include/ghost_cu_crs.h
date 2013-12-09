#ifndef __GHOST_CU_CRS_H__
#define __GHOST_CU_CRS_H__

#include <ghost_config.h>
#include <ghost_types.h>

#ifdef __cplusplus
extern "C" {
#endif

void ghost_cu_crsspmv(ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options);

#ifdef __cplusplus
}
#endif

#endif
