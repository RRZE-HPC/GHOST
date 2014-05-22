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

#define VECVAL_RM(vec,val,__x,__y) &(val[__x][(__y)*vec->elSize])
#define CUVECVAL_RM(vec,val,__x,__y) &(val[((__x)*vec->traits.nrowspadded+(__y))*vec->elSize])

#define ITER_COLS_BEGIN(vec,col,colidx)\
    colidx = 0;\
    for (col=0; col<vec->traits.ncolsorig; col++) {\
        if (hwloc_bitmap_isset(vec->ldmask,col)) {


#define ITER_COLS_END(colidx)\
            colidx++;\
        }\
    }

#define ITER_BEGIN_RM_INPAR(vec,col,row,colidx)\
    _Pragma("omp for schedule(runtime)")\
    for (row=0; row<vec->traits.nrows; row++) {\
        ITER_COLS_BEGIN(vec,col,colidx)\

#define ITER_BEGIN_RM(vec,col,row,colidx)\
    _Pragma("omp parallel private(col,colidx)")\
    ITER_BEGIN_RM_INPAR(vec,col,row,colidx)\

#define ITER_END_RM(colidx)\
        ITER_COLS_END(colidx)\
    }

#define ITER_COMPACT_BEGIN_RM(vec,col,row,colidx)\
    int offs = hwloc_bitmap_first(vec->ldmask);\
    _Pragma("omp parallel private(col,colidx)")\
        {\
        _Pragma("omp for schedule(runtime)")\
        for (row=0; row<vec->traits.nrows; row++) {\
            colidx = 0;\
            for (col=offs; col<offs+vec->traits.ncols; col++, colidx++) {

#define ITER_COMPACT_END_RM()\
            }\
        }\
    }



#define ITER2_COLS_BEGIN(vec1,vec2,col1,col2,colidx)\
    colidx = 0;\
    col2 = hwloc_bitmap_next(vec2->ldmask,-1);\
    for (col1=0; col1<vec1->traits.ncolsorig; col1++) {\
        if (hwloc_bitmap_isset(vec1->ldmask,col1)) {\


#define ITER2_COLS_END(colidx)\
            col2 = hwloc_bitmap_next(vec2->ldmask,col2);\
            colidx++;\
        }\
    }

#define ITER2_BEGIN_RM_INPAR(vec1,vec2,col1,col2,row,colidx)\
    _Pragma("omp for schedule(runtime)")\
    for (row=0; row<vec->traits.nrows; row++) {\
        ITER2_COLS_BEGIN(vec1,vec2,col1,col2,colidx)\

#define ITER2_BEGIN_RM(vec1,vec2,col1,col2,row,colidx)\
    _Pragma("omp parallel private(col1,col2,colidx)")\
    ITER2_BEGIN_RM_INPAR(vec1,vec2,col1,col2,row,colidx)\

#define ITER2_END_RM(colidx)\
        ITER2_COLS_END(colidx)\
    }

#define ITER2_COMPACT_COLS_BEGIN(vec1,vec2,col1,col2,colidx)\
    colidx = 0;\
    col1 = 0;\
    col2 = 0;\
    for (colidx=0, col1=hwloc_bitmap_first(vec1->ldmask), col2=hwloc_bitmap_first(vec2->ldmask); colidx<vec1->traits.ncols; colidx++, col1++, col2++) {\


#define ITER2_COMPACT_COLS_END()\
    }

#define ITER2_COMPACT_BEGIN_RM_INPAR(vec1,vec2,col1,col2,row,colidx)\
    _Pragma("omp for schedule(runtime)")\
    for (row=0; row<vec->traits.nrows; row++) {\
        ITER2_COMPACT_COLS_BEGIN(vec1,vec2,col1,col2,colidx)\

#define ITER2_COMPACT_BEGIN_RM(vec1,vec2,col1,col2,row,colidx)\
    _Pragma("omp parallel private(col1,col2,colidx)")\
    ITER2_COMPACT_BEGIN_RM_INPAR(vec1,vec2,col1,col2,row,colidx)\

#define ITER2_COMPACT_END_RM()\
        ITER2_COMPACT_COLS_END()\
    }

#define ITER2_COMPACT_END_RM_INPAR()



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
