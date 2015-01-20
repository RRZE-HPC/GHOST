/**
 * @file densemat_iter_macros.h
 * @brief Macros for iterating through densemats.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_DENSEMAT_ITER_MACROS_H
#define GHOST_DENSEMAT_ITER_MACROS_H

#include "ghost/omp.h"

#define DENSEMAT_COMPACT(vec) (!(vec->traits.flags & GHOST_DENSEMAT_SCATTERED))

/**
 * @brief Iterate over a densemats and execute a statement for each entry. 
 *
 * This macro sets the following variables: 
 * row,col,memrow,memcol
 *
 * @param vec The densemat.
 * @param call The statement to call for each entry.
 *
 * @return 
 */
#define DENSEMAT_ITER(vec,call)\
    ghost_lidx_t row=0,col=0,memrow=0,memcol=0;\
    if (DENSEMAT_COMPACT(vec)) {\
        if (ghost_omp_in_parallel()) {\
            DENSEMAT_ITER_BEGIN_COMPACT(vec,row,col,memrow,memcol);\
            call;\
            DENSEMAT_ITER_END();\
        } else {\
            _Pragma("omp parallel")\
            {\
                DENSEMAT_ITER_BEGIN_COMPACT(vec,row,col,memrow,memcol)\
                call;\
                DENSEMAT_ITER_END()\
            }\
        }\
    } else {\
        _Pragma("omp single")\
        {\
            WARNING_LOG("Serialized operation for scattered densemat!");\
            DENSEMAT_ITER_BEGIN_SCATTERED(vec,row,col,memrow,memcol);\
            call;\
            DENSEMAT_ITER_END();\
        }\
    }\
    /* Trick the compiler to not produce warnings about unused variables */\
    if (row+col+memrow+memcol < 0) {printf("Never happens\n");}

/**
 * @see #DENSEMAT_ITER2_OFFS with offsets set to (0,0).
 */
#define DENSEMAT_ITER2(vec1,vec2,call) DENSEMAT_ITER2_OFFS(vec1,vec2,0,0,call)

/**
 * @brief Iterate over two densemats synchronously and execute a statement for
 * each entry. An offset may be given for the second input densemat.
 *
 * This macro sets the following variables: 
 * row,col,memrow1,memcol1,memrow2,memcol2
 *
 * @param vec1 The first densemat
 * @param vec2 The second densemat
 * @param vec2roffs The row offset to the second densemat.
 * @param vec2coffs The col offset to the second densemat.
 * @param call The statement to call for each entry.
 *
 * @return 
 */
#define DENSEMAT_ITER2_OFFS(vec1,vec2,vec2roffs,vec2coffs,call)\
    ghost_lidx_t row=0,col=0,memrow1=0,memcol1=0,memrow2=0,memcol2=0;\
    if (DENSEMAT_COMPACT(vec1) && DENSEMAT_COMPACT(vec2)) {\
        if (ghost_omp_in_parallel()) {\
            DENSEMAT_ITER2_BEGIN_COMPACT_OFFS(vec1,vec2,row,col,memrow1,memrow2,memcol1,memcol2,vec2roffs,vec2coffs);\
            call;\
            DENSEMAT_ITER_END();\
        } else {\
            _Pragma("omp parallel")\
            {\
                DENSEMAT_ITER2_BEGIN_COMPACT_OFFS(vec1,vec2,row,col,memrow1,memrow2,memcol1,memcol2,vec2roffs,vec2coffs);\
                call;\
                DENSEMAT_ITER_END();\
            }\
        }\
    } else {\
        _Pragma("omp single")\
        {\
            WARNING_LOG("Serialized operation for scattered densemat!");\
            DENSEMAT_ITER2_BEGIN_SCATTERED_OFFS(vec1,vec2,row,col,memrow1,memrow2,memcol1,memcol2,vec2roffs,vec2coffs);\
            call;\
            DENSEMAT_ITER_END();\
        }\
    }\
    /* Trick the compiler to not produce warnings about unused variables */\
    if (row+col+memrow1+memcol1+memrow2+memcol2 < 0) {printf("Never happens\n");}



#ifdef ROWMAJOR
#ifdef COLMAJOR
#error "Only one of COLMAJOR or ROWMAJOR has to be defined for this header!"
#endif

#define DENSEMAT_VAL(vec,row,col) &vec->val[row][col*vec->elSize]
#define DENSEMAT_CUVAL(vec,row,col) &((char *)(vec->cu_val))[(row*vec->traits.ncolspadded+col)*vec->elSize]

#define DENSEMAT_ITER_BEGIN_COMPACT(vec,row,col,memrow,memcol)\
    ghost_lidx_t rowoffs = ghost_bitmap_first(vec->trmask);\
    ghost_lidx_t coloffs = ghost_bitmap_first(vec->ldmask);\
    _Pragma("omp for private(col,memcol,memrow) schedule(runtime)")\
    for (row=0; row<vec->traits.nrows; row++) {\
        memrow = row+rowoffs;\
        for (col = 0, memcol = coloffs; col<vec->traits.ncols; col++, memcol++) {\

#define DENSEMAT_ITER_END()\
        }\
    }

#define DENSEMAT_ITER2_BEGIN_COMPACT(vec1,vec2,row,col,memrow1,memrow2,memcol1,memcol2)\
    DENSEMAT_ITER2_BEGIN_COMPACT_OFFS(vec1,vec2,row,col,memrow1,memrow2,memcol1,memcol2,0,0)


#define DENSEMAT_ITER2_BEGIN_COMPACT_OFFS(vec1,vec2,row,col,memrow1,memrow2,memcol1,memcol2,vec2roffs,vec2coffs)\
    ghost_lidx_t coloffs1 = ghost_bitmap_first(vec1->ldmask);\
    ghost_lidx_t coloffs2 = ghost_bitmap_first(vec2->ldmask)+vec2coffs;\
    ghost_lidx_t rowoffs1 = ghost_bitmap_first(vec1->trmask);\
    ghost_lidx_t rowoffs2 = ghost_bitmap_first(vec2->trmask)+vec2roffs;\
    _Pragma("omp for private(col,memcol1,memcol2,memrow1,memrow2) schedule(runtime)")\
    for (row=0; row<vec1->traits.nrows; row++) {\
        memrow1 = row+rowoffs1;\
        memrow2 = row+rowoffs2;\
        for (col = 0, memcol1 = coloffs1, memcol2 = coloffs2;\
                col<vec1->traits.ncols;\
                col++, memcol1++, memcol2++) {



#define DENSEMAT_ITER_BEGIN_SCATTERED(vec,row,col,memrow,memcol)\
    memrow = -1;\
    for (row=0; row<vec->traits.nrows; row++) {\
        memrow = ghost_bitmap_next(vec->trmask,memrow);\
        memcol = -1;\
        for (col = 0; col<vec->traits.ncols; col++) {\
            memcol = ghost_bitmap_next(vec->ldmask,memcol);\

#define DENSEMAT_ITER2_BEGIN_SCATTERED(vec1,vec2,row,col,memrow1,memrow2,memcol1,memcol2)\
    DENSEMAT_ITER2_BEGIN_SCATTERED_OFFS(vec1,vec2,row,col,memrow1,memrow2,memcol1,memcol2,0,0)

#define DENSEMAT_ITER2_BEGIN_SCATTERED_OFFS(vec1,vec2,row,col,memrow1,memrow2,memcol1,memcol2,vec2roffs,vec2coffs)\
    memrow1 = -1;\
    memrow2 = -1;\
    for (row=0; row<vec2roffs; row++) { /* go to offset */\
        memrow2 = ghost_bitmap_next(vec2->trmask,memrow2);\
    }\
    for (row=0; row<vec1->traits.nrows; row++) {\
        memrow1 = ghost_bitmap_next(vec1->trmask,memrow1);\
        memrow2 = ghost_bitmap_next(vec2->trmask,memrow2);\
        memcol1 = -1;\
        memcol2 = -1;\
        for (col=0; col<vec2coffs; col++) { /* go to offset */\
            memcol2 = ghost_bitmap_next(vec2->ldmask,memcol2);\
        }\
        for (col=0; col<vec1->traits.ncols; col++) {\
            memcol1 = ghost_bitmap_next(vec1->ldmask,memcol1);\
            memcol2 = ghost_bitmap_next(vec2->ldmask,memcol2);\


#elif defined(COLMAJOR)

#define DENSEMAT_VAL(vec,row,col) &vec->val[col][row*vec->elSize]
#define DENSEMAT_CUVAL(vec,row,col) &((char *)(vec->cu_val))[(col*vec->traits.nrowspadded+row)*vec->elSize]

#define DENSEMAT_ITER_BEGIN_COMPACT(vec,row,col,memrow,memcol)\
    ghost_lidx_t rowoffs = ghost_bitmap_first(vec->ldmask);\
    ghost_lidx_t coloffs = ghost_bitmap_first(vec->trmask);\
    _Pragma("omp for private(memrow,col,memcol) schedule(runtime)")\
    for (row = 0; row<vec->traits.nrows; row++) {\
        memrow = rowoffs+row;\
        for (col=0; col<vec->traits.ncols; col++) {\
            memcol = coloffs+col;\

#define DENSEMAT_ITER_END()\
        }\
    }

#define DENSEMAT_ITER2_BEGIN_COMPACT(vec1,vec2,row,col,memrow1,memrow2,memcol1,memcol2)\
    DENSEMAT_ITER2_BEGIN_COMPACT_OFFS(vec1,vec2,row,col,memrow1,memrow2,memcol1,memcol2,0,0)


#define DENSEMAT_ITER2_BEGIN_COMPACT_OFFS(vec1,vec2,row,col,memrow1,memrow2,memcol1,memcol2,vec2roffs,vec2coffs)\
    ghost_lidx_t coloffs1 = ghost_bitmap_first(vec1->trmask);\
    ghost_lidx_t coloffs2 = ghost_bitmap_first(vec2->trmask)+vec2coffs;\
    ghost_lidx_t rowoffs1 = ghost_bitmap_first(vec1->ldmask);\
    ghost_lidx_t rowoffs2 = ghost_bitmap_first(vec2->ldmask)+vec2roffs;\
    _Pragma("omp for private(memrow1,memrow2,col,memcol1,memcol2) schedule(runtime)")\
    for (row=0; row<vec1->traits.nrows; row++) {\
        memrow1 = row+rowoffs1;\
        memrow2 = row+rowoffs2;\
        for (col = 0, memcol1 = coloffs1, memcol2 = coloffs2;\
                col<vec1->traits.ncols;\
                col++, memcol1++, memcol2++) {\



#define DENSEMAT_ITER_BEGIN_SCATTERED(vec,row,col,memrow,memcol)\
    memrow = -1;\
    for (row=0; row<vec->traits.nrows; row++) {\
        memrow = ghost_bitmap_next(vec->ldmask,memrow);\
        memcol = -1;\
        for (col = 0; col<vec->traits.ncols; col++) {\
            memcol = ghost_bitmap_next(vec->trmask,memcol);\

#define DENSEMAT_ITER2_BEGIN_SCATTERED(vec1,vec2,row,col,memrow1,memrow2,memcol1,memcol2)\
    DENSEMAT_ITER2_BEGIN_SCATTERED_OFFS(vec1,vec2,row,col,memrow1,memrow2,memcol1,memcol2,0,0)

#define DENSEMAT_ITER2_BEGIN_SCATTERED_OFFS(vec1,vec2,row,col,memrow1,memrow2,memcol1,memcol2,vec2roffs,vec2coffs)\
    memrow1 = -1;\
    memrow2 = -1;\
    for (row=0; row<vec2roffs; row++) { /* go to offset */\
        memrow2 = ghost_bitmap_next(vec2->ldmask,memrow2);\
    }\
    for (row=0; row<vec1->traits.nrows; row++) {\
        memrow1 = ghost_bitmap_next(vec1->ldmask,memrow1);\
        memrow2 = ghost_bitmap_next(vec2->ldmask,memrow2);\
        memcol1 = -1;\
        memcol2 = -1;\
        for (col=0; col<vec2coffs; col++) { /* go to offset */\
            memcol2 = ghost_bitmap_next(vec2->trmask,memcol2);\
        }\
        for (col=0; col<vec1->traits.ncols; col++) {\
            memcol1 = ghost_bitmap_next(vec1->trmask,memcol1);\
            memcol2 = ghost_bitmap_next(vec2->trmask,memcol2);\


#else
#error "Either COLMAJOR or ROWMAJOR has to be defined for this header!"

#endif

#endif
