/**
 * @file cu_crs.h
 * @brief CUDA CRS functions.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_CU_CRS_H
#define GHOST_CU_CRS_H

#include "spmv.h"
#include "densemat.h"
#include "sparsemat.h"
#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

ghost_error_t ghost_cu_crsspmv(ghost_sparsemat_t *mat, ghost_densemat_t * lhs, ghost_densemat_t * rhs, ghost_spmv_flags_t options, va_list argp);

#ifdef __cplusplus
}
#endif

#endif
