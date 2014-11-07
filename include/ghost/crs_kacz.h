#ifndef GHOST_CRS_KACZ_H
#define GHOST_CRS_KACZ_H

#include "error.h"
#include "sparsemat.h"
#include "densemat.h"


#ifdef __cplusplus
extern "C" {
#endif
ghost_error_t ghost_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega, int forward);
#ifdef __cplusplus
}
#endif

#endif
