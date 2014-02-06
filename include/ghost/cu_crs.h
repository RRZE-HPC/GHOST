#ifndef GHOST_CU_CRS_H
#define GHOST_CU_CRS_H

#include "config.h"
#include "types.h"
#include "error.h"
#include "vec.h"
#include "mat.h"

#ifdef __cplusplus
extern "C" {
#endif

ghost_error_t ghost_cu_crsspmv(ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options);

#ifdef __cplusplus
}
#endif

#endif
