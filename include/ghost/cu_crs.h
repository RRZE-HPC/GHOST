#ifndef GHOST_CU_CRS_H
#define GHOST_CU_CRS_H

#include "config.h"
#include "types.h"
#include "error.h"
#include "densemat.h"
#include "sparsemat.h"

#ifdef __cplusplus
extern "C" {
#endif

ghost_error_t ghost_cu_crsspmv(ghost_sparsemat_t *mat, ghost_densemat_t * lhs, ghost_densemat_t * rhs, int options);

#ifdef __cplusplus
}
#endif

#endif
