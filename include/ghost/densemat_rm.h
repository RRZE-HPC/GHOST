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

#define VECVAL(vec,val,__x,__y) &(val[__x][(__y)*vec->elSize])
#define CUVECVAL(vec,val,__x,__y) &(val[((__x)*vec->traits.nrowspadded+(__y))*vec->elSize])

#ifdef __cplusplus

extern "C" {
#endif

    /**
     * @ingroup types
     *
     * @brief Create a dense matrix/vector. 
     *
     * @param vec Where to store the matrix.
     * @param ctx The context the matrix lives in or NULL.
     * @param traits The matrix traits.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_densemat_rm_create(ghost_densemat_t *vec);
    ghost_error_t ghost_densemat_rm_malloc(ghost_densemat_t *vec);
    ghost_error_t d_ghost_densemat_rm_string(char **str, ghost_densemat_t *vec); 
    ghost_error_t s_ghost_densemat_rm_string(char **str, ghost_densemat_t *vec); 
    ghost_error_t z_ghost_densemat_rm_string(char **str, ghost_densemat_t *vec);
    ghost_error_t c_ghost_densemat_rm_string(char **str, ghost_densemat_t *vec);
    ghost_error_t d_ghost_densemat_rm_normalize(ghost_densemat_t *vec); 
    ghost_error_t s_ghost_densemat_rm_normalize(ghost_densemat_t *vec); 
    ghost_error_t z_ghost_densemat_rm_normalize(ghost_densemat_t *vec);
    ghost_error_t c_ghost_densemat_rm_normalize(ghost_densemat_t *vec);
    ghost_error_t d_ghost_densemat_rm_dotprod(ghost_densemat_t *vec1, void *res, ghost_densemat_t *vec2); 
    ghost_error_t s_ghost_densemat_rm_dotprod(ghost_densemat_t *vec1, void *res, ghost_densemat_t *vec2); 
    ghost_error_t z_ghost_densemat_rm_dotprod(ghost_densemat_t *vec1, void *res, ghost_densemat_t *vec2);
    ghost_error_t c_ghost_densemat_rm_dotprod(ghost_densemat_t *vec1, void *res, ghost_densemat_t *vec2);
    ghost_error_t d_ghost_densemat_rm_vscale(ghost_densemat_t *vec1, void *vscale); 
    ghost_error_t s_ghost_densemat_rm_vscale(ghost_densemat_t *vec1, void *vscale); 
    ghost_error_t z_ghost_densemat_rm_vscale(ghost_densemat_t *vec1, void *vscale);
    ghost_error_t c_ghost_densemat_rm_vscale(ghost_densemat_t *vec1, void *vscale);
    ghost_error_t d_ghost_densemat_rm_vaxpy(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *); 
    ghost_error_t s_ghost_densemat_rm_vaxpy(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *); 
    ghost_error_t z_ghost_densemat_rm_vaxpy(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *);
    ghost_error_t c_ghost_densemat_rm_vaxpy(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *);
    ghost_error_t d_ghost_densemat_rm_vaxpby(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *, void *); 
    ghost_error_t s_ghost_densemat_rm_vaxpby(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *, void *); 
    ghost_error_t z_ghost_densemat_rm_vaxpby(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *, void *);
    ghost_error_t c_ghost_densemat_rm_vaxpby(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *, void *);
    ghost_error_t d_ghost_densemat_rm_fromRand(ghost_densemat_t *vec); 
    ghost_error_t s_ghost_densemat_rm_fromRand(ghost_densemat_t *vec); 
    ghost_error_t z_ghost_densemat_rm_fromRand(ghost_densemat_t *vec); 
    ghost_error_t c_ghost_densemat_rm_fromRand(ghost_densemat_t *vec); 
#ifdef __cplusplus
}
#endif

#endif
