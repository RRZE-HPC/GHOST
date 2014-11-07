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

#define VECVAL_CM(vec,val,__x,__y) &(val[__x][(__y)*vec->elSize])
#define CUVECVAL_CM(vec,val,__x,__y) &(val[((__x)*vec->traits.nrowspadded+(__y))*vec->elSize])

#define ITER_ROWS_BEGIN(vec,row,rowidx)\
    for (row=0; row<vec->traits.nrowsorig; row++) {\
        if (hwloc_bitmap_isset(vec->ldmask,row)) {\


#define ITER_ROWS_END(rowidx)\
            rowidx++;\
        }\
    }

#define ITER_BEGIN_CM_INPAR(vec,col,row,rowidx)\
    for (col=0; col<vec->traits.ncols; col++) {\
        ITER_ROWS_BEGIN(vec,row,rowidx)

#define ITER_END_CM_INPAR(rowidx)\
        ITER_ROWS_END(rowidx)\
    }\

#define ITER_BEGIN_CM(vec,col,row,rowidx)\
    for (col=0; col<vec->traits.ncols; col++) {\
        row = 0;\
        rowidx = 0;\
        ITER_ROWS_BEGIN(vec,row,rowidx)

#define ITER_END_CM(rowidx)\
        ITER_ROWS_END(rowidx)\
    }\

#define ITER_COMPACT_ROWS_BEGIN(vec,row,rowidx)\
    int offs = hwloc_bitmap_first(vec->ldmask);\
    _Pragma("omp for schedule(runtime)")\
    for (row=offs; row<vec->traits.nrows+offs; row++) {\
        rowidx = row-offs;\

#define ITER_COMPACT_ROWS_END()\
    }

#define ITER_COMPACT_BEGIN_CM(vec,col,row,rowidx)\
    for (col=0; col<vec->traits.ncols; col++) {\
        _Pragma("omp parallel private(row,rowidx)")\
        {\
            ITER_COMPACT_ROWS_BEGIN(vec,row,rowidx)

#define ITER_COMPACT_END_CM()\
            ITER_COMPACT_ROWS_END()\
        }\
    }\


/*
   No-OpenMP variant. TODO: check whether the below variant is really faster than this one.
#define ITER2_ROWS_BEGIN(vec1,vec2,row1,row2,rowidx)\
    row1 = -1;\
    row2 = -1;\
    for (rowidx=0; rowidx<vec1->traits.nrows; rowidx++) {\
        row1 = hwloc_bitmap_next(vec1->ldmask,row1);\
        row2 = hwloc_bitmap_next(vec2->ldmask,row2);\

#define ITER2_ROWS_END(rowidx)\
    }*/

#define ITER2_ROWS_BEGIN(vec1,vec2,row1,row2,rowidx)\
    rowidx = 0;\
    row2 = hwloc_bitmap_next(vec2->ldmask,-1);\
    _Pragma("omp for schedule(runtime)")\
    for (row1=0; row1<vec->traits.nrowsorig; row1++) {\
        if (hwloc_bitmap_isset(vec->ldmask,row1)) {

#define ITER2_ROWS_END(rowidx)\
        _Pragma("omp critical")\
        row2 = hwloc_bitmap_next(vec2->ldmask,row2);\
        rowidx++;\
        }\
    }

#define ITER2_BEGIN_CM(vec1,vec2,col,row1,row2,rowidx)\
    for (col=0; col<vec->traits.ncols; col++) {\
        _Pragma("omp parallel private(row1,row2)")\
        ITER2_ROWS_BEGIN(vec1,vec2,row1,row2,rowidx)

#define ITER2_END_CM(rowidx)\
        ITER2_ROWS_END(rowidx)\
    }

#define ITER2_COMPACT_ROWS_BEGIN(vec1,vec2,row1,row2,rowidx)\
    rowidx = 0;\
    row1 = 0;\
    row2 = 0;\
    const int offs1 = hwloc_bitmap_first(vec1->ldmask);\
    const int offs2 = hwloc_bitmap_first(vec2->ldmask);\
    _Pragma("omp for schedule(runtime)")\
    for (rowidx=0; rowidx<vec->traits.nrows; rowidx++) {\
        row1 = offs1+rowidx;\
        row2 = offs2+rowidx;

#define ITER2_COMPACT_ROWS_END()\
    }

#define ITER2_COMPACT_BEGIN_CM(vec1,vec2,col,row1,row2,rowidx)\
    for (col=0; col<vec->traits.ncols; col++) {\
        _Pragma("omp parallel private(row1,row2,rowidx)")\
        {\
            ITER2_COMPACT_ROWS_BEGIN(vec1,vec2,row1,row2,rowidx)

#define ITER2_COMPACT_END_CM()\
            ITER2_COMPACT_ROWS_END()\
        }\
    }
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

    ghost_error_t ghost_densemat_cm_averagehalo(ghost_densemat_t *vec);

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
