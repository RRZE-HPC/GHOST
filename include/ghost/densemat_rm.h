/**
 * @file densemat_rm.h
 * @brief Types and functions related to row major dense matrices/vectors.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_DENSEMAT_RM_H
#define GHOST_DENSEMAT_RM_H

#include "config.h"
#include "types.h"
#include "context.h"
#include "perm.h"
#include "densemat.h"
#ifdef GHOST_HAVE_CUDA
#include "cu_densemat_rm.h"
#endif

#ifdef __cplusplus

extern "C" {
#endif

    /**
     * @brief Set the function pointers of a row-major densemat.
     *
     * @param[inout] vec The dense matrix.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_densemat_rm_setfuncs(ghost_densemat_t *vec);
    ghost_error_t ghost_densemat_rm_malloc(ghost_densemat_t *vec);
    ghost_error_t ghost_densemat_rm_string_selector(ghost_densemat_t *vec, char **str);
    ghost_error_t ghost_densemat_rm_normalize_selector(ghost_densemat_t *vec);
    ghost_error_t ghost_densemat_rm_dotprod_selector(ghost_densemat_t *vec, void *, ghost_densemat_t *);
    ghost_error_t ghost_densemat_rm_vscale_selector(ghost_densemat_t *vec1, void *); 
    ghost_error_t ghost_densemat_rm_vaxpy_selector(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *); 
    ghost_error_t ghost_densemat_rm_vaxpby_selector(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *, void *); 
    ghost_error_t ghost_densemat_rm_fromRand_selector(ghost_densemat_t *vec);
    ghost_error_t ghost_densemat_rm_fromScalar_selector(ghost_densemat_t *vec, void *);
    ghost_error_t ghost_densemat_rm_averagehalo_selector(ghost_densemat_t *vec);
#ifdef __cplusplus
}
#endif

#endif
