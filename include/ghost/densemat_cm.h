/**
 * @file densemat_cm.h
 * @brief Types and functions related to column major dense matrices/vectors.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_DENSEMAT_CM_H
#define GHOST_DENSEMAT_CM_H

#include "config.h"
#include "types.h"
#include "context.h"
#include "perm.h"
#include "densemat.h"
#ifdef GHOST_HAVE_CUDA
#include "cu_densemat_cm.h"
#endif

#ifdef __cplusplus

extern "C" {
#endif

    /**
     * @brief Set the function pointers of a column-major densemat.
     *
     * @param[inout] vec The dense matrix.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_densemat_cm_setfuncs(ghost_densemat_t *vec);


    ghost_error_t ghost_densemat_cm_malloc(ghost_densemat_t *vec);
    ghost_error_t ghost_densemat_cm_string_selector(ghost_densemat_t *vec, char **str);
    ghost_error_t ghost_densemat_cm_normalize_selector(ghost_densemat_t *vec);
    ghost_error_t ghost_densemat_cm_dotprod_selector(ghost_densemat_t *vec, void *, ghost_densemat_t *);
    ghost_error_t ghost_densemat_cm_vscale_selector(ghost_densemat_t *vec1, void *); 
    ghost_error_t ghost_densemat_cm_vaxpy_selector(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *); 
    ghost_error_t ghost_densemat_cm_vaxpby_selector(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *, void *); 
    ghost_error_t ghost_densemat_cm_fromScalar_selector(ghost_densemat_t *vec, void *);
    ghost_error_t ghost_densemat_cm_fromRand_selector(ghost_densemat_t *vec);
    ghost_error_t ghost_densemat_cm_fromVec_selector(ghost_densemat_t *vec1, ghost_densemat_t *vec2, ghost_lidx_t, ghost_lidx_t); 
    ghost_error_t ghost_densemat_cm_permute_selector(ghost_densemat_t *vec, ghost_permutation_direction_t dir);
    ghost_error_t ghost_densemat_cm_norm_selector(ghost_densemat_t *vec, void *res, void *p);
    ghost_error_t ghost_densemat_cm_averagehalo_selector(ghost_densemat_t *vec);
#ifdef __cplusplus
}
#endif

#endif
