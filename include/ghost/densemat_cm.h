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

#define VECVAL(vec,val,__x,__y) &(val[__x][(__y)*vec->elSize])
#define CUVECVAL(vec,val,__x,__y) &(val[((__x)*vec->traits.nrowspadded+(__y))*vec->elSize])

#define ITER_ROWS_BEGIN(vec,row,rowidx)\
    _Pragma("omp for schedule(runtime)")\
    for (row=0; row<vec->traits.nrowsorig; row++) {\
        if (hwloc_bitmap_isset(vec->mask,row)) {

#define ITER_ROWS_END(rowidx)\
            rowidx++;\
        }\
    }

#define ITER_BEGIN_CM(vec,col,row,rowidx)\
    rowidx = 0;\
    ITER_ROWS_BEGIN(vec,row,rowidx)\
    for (col=0; col<vec->traits.ncols; col++) {

#define ITER_END_CM(rowidx)\
    }\
    ITER_ROWS_END(rowidx)

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
    ghost_error_t ghost_densemat_cm_create(ghost_densemat_t *vec);

    ghost_error_t ghost_densemat_cm_malloc(ghost_densemat_t *vec);
    ghost_error_t d_ghost_densemat_cm_string(char **str, ghost_densemat_t *vec); 
    ghost_error_t s_ghost_densemat_cm_string(char **str, ghost_densemat_t *vec); 
    ghost_error_t z_ghost_densemat_cm_string(char **str, ghost_densemat_t *vec);
    ghost_error_t c_ghost_densemat_cm_string(char **str, ghost_densemat_t *vec);
    ghost_error_t d_ghost_densemat_cm_normalize(ghost_densemat_t *vec); 
    ghost_error_t s_ghost_densemat_cm_normalize(ghost_densemat_t *vec); 
    ghost_error_t z_ghost_densemat_cm_normalize(ghost_densemat_t *vec);
    ghost_error_t c_ghost_densemat_cm_normalize(ghost_densemat_t *vec);
    ghost_error_t d_ghost_densemat_cm_dotprod(ghost_densemat_t *vec1, void *res, ghost_densemat_t *vec2); 
    ghost_error_t s_ghost_densemat_cm_dotprod(ghost_densemat_t *vec1, void *res, ghost_densemat_t *vec2); 
    ghost_error_t z_ghost_densemat_cm_dotprod(ghost_densemat_t *vec1, void *res, ghost_densemat_t *vec2);
    ghost_error_t c_ghost_densemat_cm_dotprod(ghost_densemat_t *vec1, void *res, ghost_densemat_t *vec2);
    ghost_error_t d_ghost_densemat_cm_vscale(ghost_densemat_t *vec1, void *vscale); 
    ghost_error_t s_ghost_densemat_cm_vscale(ghost_densemat_t *vec1, void *vscale); 
    ghost_error_t z_ghost_densemat_cm_vscale(ghost_densemat_t *vec1, void *vscale);
    ghost_error_t c_ghost_densemat_cm_vscale(ghost_densemat_t *vec1, void *vscale);
    ghost_error_t d_ghost_densemat_cm_vaxpy(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *); 
    ghost_error_t s_ghost_densemat_cm_vaxpy(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *); 
    ghost_error_t z_ghost_densemat_cm_vaxpy(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *);
    ghost_error_t c_ghost_densemat_cm_vaxpy(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *);
    ghost_error_t d_ghost_densemat_cm_vaxpby(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *, void *); 
    ghost_error_t s_ghost_densemat_cm_vaxpby(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *, void *); 
    ghost_error_t z_ghost_densemat_cm_vaxpby(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *, void *);
    ghost_error_t c_ghost_densemat_cm_vaxpby(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *, void *);
    ghost_error_t d_ghost_densemat_cm_fromRand(ghost_densemat_t *vec); 
    ghost_error_t s_ghost_densemat_cm_fromRand(ghost_densemat_t *vec); 
    ghost_error_t z_ghost_densemat_cm_fromRand(ghost_densemat_t *vec); 
    ghost_error_t c_ghost_densemat_cm_fromRand(ghost_densemat_t *vec); 
#ifdef __cplusplus
}
#endif

#endif
