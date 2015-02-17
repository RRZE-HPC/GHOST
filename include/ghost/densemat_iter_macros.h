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
 * row,col,memrow,memcol,valptr
 *
 * @param vec The densemat.
 * @param call The statement to call for each entry.
 *
 * @return 
 */
#define DENSEMAT_ITER(vec,call)\
    ghost_lidx_t row=0,col=0,memrow=0,memcol=0;\
    char *valptr = NULL, *cuvalptr = NULL;\
    if (DENSEMAT_COMPACT(vec)) {\
        if (ghost_omp_in_parallel()) {\
            DENSEMAT_ITER_BEGIN_COMPACT(vec,valptr,row,col,memrow,memcol);\
            valptr = DENSEMAT_VAL(vec,row,col);\
            cuvalptr = DENSEMAT_CUVAL(vec,row,col);\
            call;\
            DENSEMAT_ITER_END();\
        } else {\
            _Pragma("omp parallel")\
            {\
                DENSEMAT_ITER_BEGIN_COMPACT(vec,valptr,row,col,memrow,memcol)\
                valptr = DENSEMAT_VAL(vec,row,col);\
                cuvalptr = DENSEMAT_CUVAL(vec,row,col);\
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
    if ((row+col+memrow+memcol < 0) || \
            (valptr == (char *)0xbeef) || (cuvalptr == (char*)0xbeef)) \
            {printf("Never happens\n");}

/**
 * @see #DENSEMAT_ITER2_OFFS with offsets set to (0,0).
 */
#define DENSEMAT_ITER2(vec1,vec2,call) DENSEMAT_ITER2_OFFS(vec1,vec2,0,0,call)

/**
 * @brief Iterate over two densemats synchronously and execute a statement for
 * each entry. An offset may be given for the second input densemat.
 *
 * This macro sets the following variables: 
 * row,col,memrow1,memcol1,memrow2,memcol2,valptr1,valptr2
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
    char *valptr1 = NULL, *valptr2 = NULL, *cuvalptr1 = NULL, *cuvalptr2 = NULL;\
    if (DENSEMAT_COMPACT(vec1) && DENSEMAT_COMPACT(vec2)) {\
        if (ghost_omp_in_parallel()) {\
            DENSEMAT_ITER2_BEGIN_COMPACT_OFFS(vec1,vec2,valptr1,valptr2,row,col,memrow1,memrow2,memcol1,memcol2,vec2roffs,vec2coffs);\
            valptr1 = DENSEMAT_VAL(vec1,row,col);\
            valptr2 = DENSEMAT_VAL(vec2,row+vec2roffs,col+vec2coffs);\
            cuvalptr1 = DENSEMAT_CUVAL(vec1,row,col);\
            cuvalptr2 = DENSEMAT_CUVAL(vec2,row+vec2roffs,col+vec2coffs);\
            call;\
            DENSEMAT_ITER_END();\
        } else {\
            _Pragma("omp parallel")\
            {\
                DENSEMAT_ITER2_BEGIN_COMPACT_OFFS(vec1,vec2,valptr1,valptr2,row,col,memrow1,memrow2,memcol1,memcol2,vec2roffs,vec2coffs);\
                valptr1 = DENSEMAT_VAL(vec1,row,col);\
                valptr2 = DENSEMAT_VAL(vec2,row+vec2roffs,col+vec2coffs);\
                cuvalptr1 = DENSEMAT_CUVAL(vec1,row,col);\
                cuvalptr2 = DENSEMAT_CUVAL(vec2,row+vec2roffs,col+vec2coffs);\
                call;\
                DENSEMAT_ITER_END();\
            }\
        }\
    } else {\
        if (DENSEMAT_COMPACT(vec1)) {\
            _Pragma("omp single")\
            {\
                WARNING_LOG("Serialized operation for scattered densemat! vec1 compact, vec2 scattered");\
                DENSEMAT_ITER2_BEGIN_SCATTERED2_OFFS(vec1,vec2,row,col,memrow1,memrow2,memcol1,memcol2,vec2roffs,vec2coffs);\
                call;\
                DENSEMAT_ITER_END();\
            }\
        } else if (DENSEMAT_COMPACT(vec2)) {\
            _Pragma("omp single")\
            {\
                WARNING_LOG("Serialized operation for scattered densemat! vec1 scattered, vec2 compact");\
                DENSEMAT_ITER2_BEGIN_SCATTERED1_OFFS(vec1,vec2,row,col,memrow1,memrow2,memcol1,memcol2,vec2roffs,vec2coffs);\
                call;\
                DENSEMAT_ITER_END();\
            }\
        } else {\
            _Pragma("omp single")\
            {\
                WARNING_LOG("Serialized operation for scattered densemat! both scattered");\
                DENSEMAT_ITER2_BEGIN_SCATTERED_OFFS(vec1,vec2,row,col,memrow1,memrow2,memcol1,memcol2,vec2roffs,vec2coffs);\
                call;\
                DENSEMAT_ITER_END();\
            }\
        }\
    }\
    /* Trick the compiler to not produce warnings about unused variables */\
    if ((row+col+memrow1+memcol1+memrow2+memcol2 < 0) || \
            (valptr1 == (char *)0xbeef) || (valptr2 == (char *)0xbeef) || \
            (cuvalptr1 == (char *)0xbeef) || (cuvalptr2 == (char *)0xbeef))\
            {printf("Never happens\n");}

/**
 * @brief Iterate over two densemats synchronously and execute a statement for
 * each entry. The second densemat is stored transposed (compared to the first).
 * An offset may be given for the second input densemat.
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
#define DENSEMAT_ITER2_COMPACT_OFFS_TRANSPOSED(vec1,vec2,vec2roffs,vec2coffs,call)\
    ghost_lidx_t row=0,col=0,memrow1=0,memcol1=0,memrow2=0,memcol2=0;\
    char *valptr1 = NULL, *valptr2 = NULL, *cuvalptr1 = NULL, *cuvalptr2 = NULL;\
    if (ghost_omp_in_parallel()) {\
        DENSEMAT_ITER2_BEGIN_COMPACT_OFFS_TRANSPOSED(vec1,vec2,row,col,memrow1,memrow2,memcol1,memcol2,vec2roffs,vec2coffs);\
        valptr1 = DENSEMAT_VAL(vec1,row,col);\
        valptr2 = DENSEMAT_VAL_TRANSPOSED(vec2,row+vec2roffs,col+vec2coffs);\
        cuvalptr1 = DENSEMAT_CUVAL(vec1,row,col);\
        cuvalptr2 = DENSEMAT_CUVAL_TRANSPOSED(vec2,row+vec2roffs,col+vec2coffs);\
        call;\
        DENSEMAT_ITER_END();\
    } else {\
        _Pragma("omp parallel")\
        {\
            DENSEMAT_ITER2_BEGIN_COMPACT_OFFS_TRANSPOSED(vec1,vec2,row,col,memrow1,memrow2,memcol1,memcol2,vec2roffs,vec2coffs);\
            valptr1 = DENSEMAT_VAL(vec1,row,col);\
            valptr2 = DENSEMAT_VAL_TRANSPOSED(vec2,row+vec2roffs,col+vec2coffs);\
            cuvalptr1 = DENSEMAT_CUVAL(vec1,row,col);\
            cuvalptr2 = DENSEMAT_CUVAL_TRANSPOSED(vec2,row+vec2roffs,col+vec2coffs);\
            call;\
            DENSEMAT_ITER_END();\
        }\
    }\
    /* Trick the compiler to not produce warnings about unused variables */\
    if ((row+col+memrow1+memcol1+memrow2+memcol2 < 0) || \
            (valptr1 == (char *)0xbeef) || (valptr2 == (char *)0xbeef) || \
            (cuvalptr1 == (char *)0xbeef) || (cuvalptr2 == (char *)0xbeef)) \
            {printf("Never happens\n");}


#define DENSEMAT_ITER_BEGIN_COMPACT(vec,valptr,row,col,memrow,memcol)\
    _Pragma("omp for private(valptr,col,memcol) schedule(runtime)")\
    for (row = 0; row<vec->traits.nrows; row++) {\
        memrow = row;\
        for (col = 0, memcol = 0; col<vec->traits.ncols; col++, memcol++) {\

#define DENSEMAT_ITER_END()\
        }\
    }

#define DENSEMAT_ITER2_BEGIN_COMPACT(vec1,vec2,row,col,memrow1,memrow2,memcol1,memcol2)\
    DENSEMAT_ITER2_BEGIN_COMPACT_OFFS(vec1,vec2,row,col,memrow1,memrow2,memcol1,memcol2,0,0)

#define DENSEMAT_ITER2_BEGIN_COMPACT_OFFS(vec1,vec2,valptr1,valptr2,row,col,memrow1,memrow2,memcol1,memcol2,vec2roffs,vec2coffs)\
    _Pragma("omp for private(col,memcol1,memcol2,memrow1,memrow2,valptr1,valptr2) schedule(runtime)")\
    for (row=0; row<vec1->traits.nrows; row++) {\
        memrow1 = row;\
        memrow2 = row;\
        for (col = 0, memcol1 = 0, memcol2 = 0;\
                col<vec1->traits.ncols;\
                col++, memcol1++, memcol2++) {

#define DENSEMAT_ITER2_BEGIN_COMPACT_OFFS_TRANSPOSED(vec1,vec2,row,col,memrow1,memrow2,memcol1,memcol2,vec2roffs,vec2coffs)\
    _Pragma("omp for private(memrow1,memrow2,col,memcol1,memcol2,valptr1,valptr2) schedule(runtime)")\
    for (row=0; row<vec1->traits.nrows; row++) {\
        memrow1 = row;\
        memrow2 = row;\
        for (col = 0, memcol1 = 0, memcol2 = 0;\
                col<vec1->traits.ncols;\
                col++, memcol1++, memcol2++) {\


#ifdef ROWMAJOR
#ifdef COLMAJOR
#error "Only one of COLMAJOR or ROWMAJOR has to be defined for this header!"
#endif

//#define DENSEMAT_VAL(vec,row,col) &vec->val[row][(col)*vec->elSize]
//#define DENSEMAT_VAL(vec,row,col) &vec->val[0][((row)*(vec->stride)+(col))*vec->elSize]
#define DENSEMAT_VAL(vec,row,col) &vec->val[((row)*(vec->stride)+(col))*vec->elSize]
//#define DENSEMAT_VAL_TRANSPOSED(vec,row,col) &vec->val[col][(row)*vec->elSize]
#define DENSEMAT_VAL_TRANSPOSED(vec,row,col) &vec->val[((col)*vec->stride+(row))*vec->elSize]
#define DENSEMAT_CUVAL(vec,row,col) vec->cu_val?(&((char *)(vec->cu_val))[((row)*vec->stride+(col))*vec->elSize]):NULL
#define DENSEMAT_CUVAL_TRANSPOSED(vec,row,col) &((char *)(vec->cu_val))[((col)*vec->stride+(row))*vec->elSize]


#define DENSEMAT_ITER_BEGIN_SCATTERED(vec,row,col,memrow,memcol)\
    memrow = -1;\
    for (row=0; row<vec->traits.nrows; row++) {\
        memrow = ghost_bitmap_next(vec->trmask,memrow);\
        memcol = -1;\
        for (col = 0; col<vec->traits.ncols; col++) {\
            memcol = ghost_bitmap_next(vec->ldmask,memcol);\
            valptr = DENSEMAT_VAL(vec,memrow,memcol);\
            cuvalptr = DENSEMAT_CUVAL(vec,memrow,memcol);\

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
            valptr1 = DENSEMAT_VAL(vec1,memrow1,memcol1);\
            valptr2 = DENSEMAT_VAL(vec2,memrow2+vec2roffs,memcol2+vec2coffs);\
            cuvalptr1 = DENSEMAT_CUVAL(vec1,memrow1,memcol1);\
            cuvalptr2 = DENSEMAT_CUVAL(vec2,memrow2+vec2roffs,memcol2+vec2coffs);\

#define DENSEMAT_ITER2_BEGIN_SCATTERED1_OFFS(vec1,vec2,row,col,memrow1,memrow2,memcol1,memcol2,vec2roffs,vec2coffs)\
    memrow1 = -1;\
    for (row=0; row<vec1->traits.nrows; row++) {\
        memrow1 = ghost_bitmap_next(vec1->trmask,memrow1);\
        memrow2 = row;\
        memcol1 = -1;\
        for (col=0; col<vec1->traits.ncols; col++) {\
            memcol1 = ghost_bitmap_next(vec1->ldmask,memcol1);\
            memcol2 = col;\
            valptr1 = DENSEMAT_VAL(vec1,memrow1,memcol1);\
            valptr2 = DENSEMAT_VAL(vec2,row+vec2roffs,col+vec2coffs);\
            cuvalptr1 = DENSEMAT_CUVAL(vec1,memrow1,memcol1);\
            cuvalptr2 = DENSEMAT_CUVAL(vec2,row+vec2roffs,col+vec2coffs);\

#define DENSEMAT_ITER2_BEGIN_SCATTERED2_OFFS(vec1,vec2,row,col,memrow1,memrow2,memcol1,memcol2,vec2roffs,vec2coffs)\
    memrow2 = -1;\
    for (row=0; row<vec2roffs; row++) { /* go to offset */\
        memrow2 = ghost_bitmap_next(vec2->trmask,memrow2);\
    }\
    for (row=0; row<vec1->traits.nrows; row++) {\
        memrow1 = row;\
        memrow2 = ghost_bitmap_next(vec2->trmask,memrow2);\
        memcol2 = -1;\
        for (col=0; col<vec2coffs; col++) { /* go to offset */\
            memcol2 = ghost_bitmap_next(vec2->ldmask,memcol2);\
        }\
        for (col=0; col<vec1->traits.ncols; col++) {\
            memcol1 = col;\
            memcol2 = ghost_bitmap_next(vec2->ldmask,memcol2);\
            valptr1 = DENSEMAT_VAL(vec1,row,col);\
            valptr2 = DENSEMAT_VAL(vec2,memrow2+vec2roffs,memcol2+vec2coffs);\
            cuvalptr1 = DENSEMAT_CUVAL(vec1,row,col);\
            cuvalptr2 = DENSEMAT_CUVAL(vec2,memrow2+vec2roffs,memcol2+vec2coffs);\

#elif defined(COLMAJOR)

//#define DENSEMAT_VAL(vec,row,col) &vec->val[col][(row)*vec->elSize]
#define DENSEMAT_VAL(vec,row,col) &vec->val[((col)*vec->stride+(row))*vec->elSize]
//#define DENSEMAT_VAL_TRANSPOSED(vec,row,col) &vec->val[row][(col)*vec->elSize]
#define DENSEMAT_VAL_TRANSPOSED(vec,row,col) &vec->val[((row)*vec->stride+(col))*vec->elSize]
#define DENSEMAT_CUVAL(vec,row,col) &((char *)(vec->cu_val))[((col)*vec->stride+(row))*vec->elSize]
#define DENSEMAT_CUVAL_TRANSPOSED(vec,row,col) &((char *)(vec->cu_val))[((row)*vec->stride+(col))*vec->elSize]

#define DENSEMAT_ITER_BEGIN_SCATTERED(vec,row,col,memrow,memcol)\
    memrow = -1;\
    for (row=0; row<vec->traits.nrows; row++) {\
        memrow = ghost_bitmap_next(vec->ldmask,memrow);\
        memcol = -1;\
        for (col = 0; col<vec->traits.ncols; col++) {\
            memcol = ghost_bitmap_next(vec->trmask,memcol);\
            valptr = DENSEMAT_VAL(vec,memrow,memcol);\
            cuvalptr = DENSEMAT_CUVAL(vec,memrow,memcol);\

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
            valptr1 = DENSEMAT_VAL(vec1,memrow1,memcol1);\
            valptr2 = DENSEMAT_VAL(vec2,memrow2+vec2roffs,memcol2+vec2coffs);\
            cuvalptr1 = DENSEMAT_CUVAL(vec1,memrow1,memcol1);\
            cuvalptr2 = DENSEMAT_CUVAL(vec2,memrow2+vec2roffs,memcol2+vec2coffs);\

#define DENSEMAT_ITER2_BEGIN_SCATTERED1_OFFS(vec1,vec2,row,col,memrow1,memrow2,memcol1,memcol2,vec2roffs,vec2coffs)\
    memrow1 = -1;\
    for (row=0; row<vec1->traits.nrows; row++) {\
        memrow1 = ghost_bitmap_next(vec1->ldmask,memrow1);\
        memrow2 = row;\
        memcol1 = -1;\
        for (col=0; col<vec1->traits.ncols; col++) {\
            memcol1 = ghost_bitmap_next(vec1->trmask,memcol1);\
            memcol2 = col;\
            valptr1 = DENSEMAT_VAL(vec1,memrow1,memcol1);\
            valptr2 = DENSEMAT_VAL(vec2,row+vec2roffs,col+vec2coffs);\
            cuvalptr1 = DENSEMAT_CUVAL(vec1,memrow1,memcol1);\
            cuvalptr2 = DENSEMAT_CUVAL(vec2,row+vec2roffs,col+vec2coffs);\

#define DENSEMAT_ITER2_BEGIN_SCATTERED2_OFFS(vec1,vec2,row,col,memrow1,memrow2,memcol1,memcol2,vec2roffs,vec2coffs)\
    memrow2 = -1;\
    for (row=0; row<vec2roffs; row++) { /* go to offset */\
        memrow2 = ghost_bitmap_next(vec2->ldmask,memrow2);\
    }\
    for (row=0; row<vec1->traits.nrows; row++) {\
        memrow1 = row;\
        memrow2 = ghost_bitmap_next(vec2->ldmask,memrow2);\
        memcol2 = -1;\
        for (col=0; col<vec2coffs; col++) { /* go to offset */\
            memcol2 = ghost_bitmap_next(vec2->trmask,memcol2);\
        }\
        for (col=0; col<vec1->traits.ncols; col++) {\
            memcol1 = col;\
            memcol2 = ghost_bitmap_next(vec2->trmask,memcol2);\
            valptr1 = DENSEMAT_VAL(vec1,row,col);\
            valptr2 = DENSEMAT_VAL(vec2,memrow2+vec2roffs,memcol2+vec2coffs);\
            cuvalptr1 = DENSEMAT_CUVAL(vec1,row,col);\
            cuvalptr2 = DENSEMAT_CUVAL(vec2,memrow2+vec2roffs,memcol2+vec2coffs);\


#else
#error "Either COLMAJOR or ROWMAJOR has to be defined for this header!"

#endif

#endif
