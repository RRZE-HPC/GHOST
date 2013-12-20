#ifndef __GHOST_MAT_H__
#define __GHOST_MAT_H__

#include "config.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

ghost_mat_t     * ghost_createMatrix(ghost_context_t *, ghost_mtraits_t *, int);
ghost_mnnz_t ghost_getMatNnz(ghost_mat_t *mat);
ghost_mnnz_t ghost_getMatNrows(ghost_mat_t *mat);

#ifdef __cplusplus
} extern "C"
#endif

#endif
