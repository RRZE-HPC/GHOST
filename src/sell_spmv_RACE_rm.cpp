#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/instr.h"
#include "ghost/omp.h"
#include "ghost/machine.h"
#include "ghost/math.h"
#include "ghost/sparsemat.h"
#include "ghost/densemat.h"
#include "ghost/locality.h"
#include <complex>
#include <complex.h>
#include "ghost/timing.h"

//#define ENABLE_COMPRESSION

//#GHOST_SUBST NVECS ${BLOCKDIM1}
//#GHOST_SUBST CHUNKHEIGHT ${CHUNKHEIGHT}
#define CHUNKHEIGHT 1
#define NVECS 1

#if (NVECS==1 && CHUNKHEIGHT==1)

#ifdef ENABLE_COMPRESSION

/*
#define LOOP(start,end,MT,VT) \
_Pragma("omp parallel for schedule(static)") \
for (ghost_lidx row=start; row<end; ++row){ \
bval[row] = 0; \
ghost_lidx col_idx;\
_Pragma("unroll")\
_Pragma("simd")\
for (ghost_lidx j=mat->chunkStart[row]; j<mat->chunkStart[row+1]; j++) { \*/
/*        decodeCol(mat, idx+j, col_idx);\*/
/*        int idx_decode = (int)(j*(mat->compress->invNumBlockPerEl));\
          int rem = (int) (3*(j - (idx_decode)*mat->compress->numBlockPerEl));\
          col_idx =   ((mat->compress->block[idx_decode]>>(rem))<<16) + mat->compress->compressedCol[j];\
          bval[row] = bval[row] + (MT)mval[j] * (VT)xval[col_idx];\
          }\
          }\

#define LOOP(start,end,MT,VT) \
_Pragma("omp parallel for schedule(static)") \
for (ghost_lidx row=start; row<end; ++row){ \
bval[row] = 0; \
ghost_lidx col_idx, j, nnzCtr;\
for (j=mat->compress->blockPtr[row], nnzCtr=mat->chunkStart[row]; j<mat->compress->blockPtr[row+1]; ++j) { \
ghost_lidx k_end = (j!=(mat->compress->blockPtr[row+1]-1))?(mat->compress->numBlockPerEl):(mat->chunkStart[row+1]-nnzCtr);\
for(ghost_lidx k=0; k<k_end; ++k) {\
col_idx =  (((ghost_bit) (mat->compress->block[j]>>(3*k) ) & 7)<<16) + mat->compress->compressedCol[nnzCtr];\
bval[row] = bval[row] + (MT)mval[nnzCtr] * (VT)xval[col_idx];\
++nnzCtr;\
}\
}\
}\
*/

//Ivy Bridge needs this
#define LOOP(start,end,MT,VT) \
    _Pragma("omp parallel for schedule(static)") \
for (ghost_lidx row=start; row<end; ++row){ \
    bval[row] = 0; \
    ghost_lidx col_idx, j, nnzCtr;\
    for (j=mat->compress->blockPtr[row], nnzCtr=mat->chunkStart[row]; j<mat->compress->blockPtr[row+1]-1; ++j) { \
        col_idx =  (((ghost_bit) (mat->compress->block[j]>>(0) ) & 7)<<16) + mat->compress->compressedCol[nnzCtr];\
        bval[row] = bval[row] + (MT)mval[nnzCtr] * (VT)xval[col_idx];\
        ++nnzCtr;\
        col_idx =  (((ghost_bit) (mat->compress->block[j]>>(3) ) & 7)<<16) + mat->compress->compressedCol[nnzCtr];\
        bval[row] = bval[row] + (MT)mval[nnzCtr] * (VT)xval[col_idx];\
        ++nnzCtr;\
        col_idx =  (((ghost_bit) (mat->compress->block[j]>>(6) ) & 7)<<16) + mat->compress->compressedCol[nnzCtr];\
        bval[row] = bval[row] + (MT)mval[nnzCtr] * (VT)xval[col_idx];\
        ++nnzCtr;\
        col_idx =  (((ghost_bit) (mat->compress->block[j]>>(9) ) & 7)<<16) + mat->compress->compressedCol[nnzCtr];\
        bval[row] = bval[row] + (MT)mval[nnzCtr] * (VT)xval[col_idx];\
        ++nnzCtr;\
        col_idx =  (((ghost_bit) (mat->compress->block[j]>>(12) ) & 7)<<16) + mat->compress->compressedCol[nnzCtr];\
        bval[row] = bval[row] + (MT)mval[nnzCtr] * (VT)xval[col_idx];\
        ++nnzCtr;\
        col_idx =  (((ghost_bit) (mat->compress->block[j]>>(15) ) & 7)<<16) + mat->compress->compressedCol[nnzCtr];\
        bval[row] = bval[row] + (MT)mval[nnzCtr] * (VT)xval[col_idx];\
        ++nnzCtr;\
        col_idx =  (((ghost_bit) (mat->compress->block[j]>>(18) ) & 7)<<16) + mat->compress->compressedCol[nnzCtr];\
        bval[row] = bval[row] + (MT)mval[nnzCtr] * (VT)xval[col_idx];\
        ++nnzCtr;\
        col_idx =  (((ghost_bit) (mat->compress->block[j]>>(21) ) & 7)<<16) + mat->compress->compressedCol[nnzCtr];\
        bval[row] = bval[row] + (MT)mval[nnzCtr] * (VT)xval[col_idx];\
        ++nnzCtr;\
        col_idx =  (((ghost_bit) (mat->compress->block[j]>>(24) ) & 7)<<16) + mat->compress->compressedCol[nnzCtr];\
        bval[row] = bval[row] + (MT)mval[nnzCtr] * (VT)xval[col_idx];\
        ++nnzCtr;\
        col_idx =  (((ghost_bit) (mat->compress->block[j]>>(27) ) & 7)<<16) + mat->compress->compressedCol[nnzCtr];\
        bval[row] = bval[row] + (MT)mval[nnzCtr] * (VT)xval[col_idx];\
        ++nnzCtr;\
        col_idx =  (((ghost_bit) (mat->compress->block[j]>>(30) ) & 7)<<16) + mat->compress->compressedCol[nnzCtr];\
        bval[row] = bval[row] + (MT)mval[nnzCtr] * (VT)xval[col_idx];\
        ++nnzCtr;\
        col_idx =  (((ghost_bit) (mat->compress->block[j]>>(33) ) & 7)<<16) + mat->compress->compressedCol[nnzCtr];\
        bval[row] = bval[row] + (MT)mval[nnzCtr] * (VT)xval[col_idx];\
        ++nnzCtr;\
        col_idx =  (((ghost_bit) (mat->compress->block[j]>>(36) ) & 7)<<16) + mat->compress->compressedCol[nnzCtr];\
        bval[row] = bval[row] + (MT)mval[nnzCtr] * (VT)xval[col_idx];\
        ++nnzCtr;\
        col_idx =  (((ghost_bit) (mat->compress->block[j]>>(39) ) & 7)<<16) + mat->compress->compressedCol[nnzCtr];\
        bval[row] = bval[row] + (MT)mval[nnzCtr] * (VT)xval[col_idx];\
        ++nnzCtr;\
        col_idx =  (((ghost_bit) (mat->compress->block[j]>>(42) ) & 7)<<16) + mat->compress->compressedCol[nnzCtr];\
        bval[row] = bval[row] + (MT)mval[nnzCtr] * (VT)xval[col_idx];\
        ++nnzCtr;\
        col_idx =  (((ghost_bit) (mat->compress->block[j]>>(45) ) & 7)<<16) + mat->compress->compressedCol[nnzCtr];\
        bval[row] = bval[row] + (MT)mval[nnzCtr] * (VT)xval[col_idx];\
        ++nnzCtr;\
        col_idx =  (((ghost_bit) (mat->compress->block[j]>>(48) ) & 7)<<16) + mat->compress->compressedCol[nnzCtr];\
        bval[row] = bval[row] + (MT)mval[nnzCtr] * (VT)xval[col_idx];\
        ++nnzCtr;\
        col_idx =  (((ghost_bit) (mat->compress->block[j]>>(51) ) & 7)<<16) + mat->compress->compressedCol[nnzCtr];\
        bval[row] = bval[row] + (MT)mval[nnzCtr] * (VT)xval[col_idx];\
        ++nnzCtr;\
        col_idx =  (((ghost_bit) (mat->compress->block[j]>>(54) ) & 7)<<16) + mat->compress->compressedCol[nnzCtr];\
        bval[row] = bval[row] + (MT)mval[nnzCtr] * (VT)xval[col_idx];\
        ++nnzCtr;\
        col_idx =  (((ghost_bit) (mat->compress->block[j]>>(57) ) & 7)<<16) + mat->compress->compressedCol[nnzCtr];\
        bval[row] = bval[row] + (MT)mval[nnzCtr] * (VT)xval[col_idx];\
        ++nnzCtr;\
        col_idx =  (((ghost_bit) (mat->compress->block[j]>>(60) ) & 7)<<16) + mat->compress->compressedCol[nnzCtr];\
        bval[row] = bval[row] + (MT)mval[nnzCtr] * (VT)xval[col_idx];\
        ++nnzCtr;\
    }\
    /*last reminder*/\
    j = mat->compress->blockPtr[row+1]-1;\
    ghost_lidx k_end = mat->chunkStart[row+1]-nnzCtr; \
    for(ghost_lidx k=0; k<k_end; ++k) {\
        col_idx =  (((ghost_bit) (mat->compress->block[j]>>(3*k) ) & 7)<<16) + mat->compress->compressedCol[nnzCtr];\
        bval[row] = bval[row] + (MT)mval[nnzCtr] * (VT)xval[col_idx];\
        ++nnzCtr;\
    }\
}\

//with ghost_unroll
/*#define LOOP(start,end,MT,VT) \
  _Pragma("omp parallel for schedule(static)") \
  for (ghost_lidx row=start; row<end; ++row){ \
  bval[row] = 0; \
  ghost_lidx col_idx, j, nnzCtr;\
  for (j=mat->compress->blockPtr[row], nnzCtr=mat->chunkStart[row]; j<mat->compress->blockPtr[row+1]-1; ++j) { \
#GHOST_UNROLL#col_idx =  (((ghost_bit) (mat->compress->block[j]>>(~(3*@~) ) & 7)<<16) + mat->compress->compressedCol[nnzCtr];bval[row] = bval[row] + (MT)mval[nnzCtr] * (VT)xval[col_idx]; ++nnzCtr;\#(21)\
}\*/
/*last reminder*/\
        /*    j = mat->compress->blockPtr[row+1]-1;\
              ghost_lidx k_end = mat->chunkStart[row+1]-nnzCtr; \
              for(ghost_lidx k=0; k<k_end; ++k) {\
              col_idx =  (((ghost_bit) (mat->compress->block[j]>>(3*k) ) & 7)<<16) + mat->compress->compressedCol[nnzCtr];\
              bval[row] = bval[row] + (MT)mval[nnzCtr] * (VT)xval[col_idx];\
              ++nnzCtr;\
              }\
              }\*/



    //TODO interchange loop j&k depending on avg. rowLen and numBlockPerEl; to make good
    //SIMD.
    /*
#define LOOP(start,end,MT,VT) \
_Pragma("omp parallel for schedule(static)") \
for (ghost_lidx row=start; row<end; ++row){ \
bval[row] = 0; \
ghost_lidx col_idx, j, nnzCtr;\
for (j=mat->compress->blockPtr[row], nnzCtr=mat->chunkStart[row]; j<mat->compress->blockPtr[row+1]-1; ++j) { \
for(ghost_lidx k=0; k<mat->compress->numBlockPerEl; ++k) {\
col_idx =  (((ghost_bit) (mat->compress->block[j]>>(mat->compress->blockBitSize*k) ) & mat->compress->mask)<<16) + mat->compress->compressedCol[nnzCtr];\
bval[row] = bval[row] + (MT)mval[nnzCtr] * (VT)xval[col_idx];\
++nnzCtr;\
}\
}\*/
    /*last reminder \*/
    /*    j = mat->compress->blockPtr[row+1]-1;\
          ghost_lidx k_end = mat->chunkStart[row+1]-nnzCtr; \
          for(ghost_lidx k=0; k<k_end; ++k) {\
          col_idx =  (((ghost_bit) (mat->compress->block[j]>>(mat->compress->blockBitSize*k) ) & mat->compress->mask)<<16) + mat->compress->compressedCol[nnzCtr];\
          bval[row] = bval[row] + (MT)mval[nnzCtr] * (VT)xval[col_idx];\
          ++nnzCtr;\
          }\
          }\


#define LOOP(start,end,MT,VT) \
_Pragma("omp parallel for schedule(static)") \
for (ghost_lidx row=start; row<end; ++row){ \
bval[row] = 0; \
ghost_lidx idx = mat->chunkStart[row];\
decodeColByRow(mat, row, col);\
for (ghost_lidx j=0; j<mat->rowLen[row]; j++) { \
bval[row] = bval[row] + (MT)mval[idx+j] * (VT)xval[col[j]];\
} \
} \
*/

#else

#if defined GHOST_BUILD_AVX512
#define VECLEN 8
#elif defined (GHOST_BUILD_AVX2) || defined (GHOST_BUILD_AVX)
#define VECLEN 4
#elif defined (GHOST_BUILD_SSE)
#define VECLEN 2
#else
#define VECLEN 1
#endif


#define LOOP(start,end,MT,VT) \
        _Pragma("omp parallel for schedule(static)") \
    for (ghost_lidx row=start; row<end; ++row){ \
        double temp = 0; \
        ghost_lidx idx = mat->chunkStart[row]; \
        _Pragma("unroll")\
        _Pragma("simd vectorlength(VECLEN) reduction(+:temp)")\
        for (ghost_lidx j=0; j<mat->rowLen[row]; j++) { \
            temp = temp + (MT)mval[idx+j] * (VT)xval[mat->col[idx+j]];\
        } \
        bval[row]=temp;\
    } \

#endif

#elif CHUNKHEIGHT == 1

#define LOOP(start,end,MT,VT) \
        /*    _Pragma("omp parallel for") */\
    for (ghost_lidx row=start; row<end; ++row){ \
        ghost_lidx idx = mat->chunkStart[row]; \
        for (ghost_lidx j=0; j<mat->rowLen[row]; j++) { \
            MT mval_idx = (MT)mval[idx]; \
            ghost_lidx col_idx = mat->col[idx]; \
            _Pragma("simd") \
            for(int block=0; block<NVECS; ++block) { \
                VT x_row = xval[row]; \
                bval[NVECS*col_idx+block] = bval[NVECS*col_idx+block] + mval_idx * xval[NVECS*row + block];\
            } \
            idx += 1; \
        } \
    } \

#else

#define LOOP(start,end,MT,VT) \
        start_rem = start%CHUNKHEIGHT; \
    start_chunk = start/CHUNKHEIGHT; \
    end_chunk = end/CHUNKHEIGHT; \
    end_rem = end%CHUNKHEIGHT; \
    chunk = 0; \
    rowinchunk = 0; \
    idx=0, row=0; \
    for(rowinchunk=start_rem; rowinchunk<MIN(CHUNKHEIGHT,(end_chunk-start_chunk)*CHUNKHEIGHT+end_rem); ++rowinchunk) { \
        MT rownorm = 0.; \
        MT scal[NVECS] = {0}; \
        idx = mat->chunkStart[start_chunk] + rowinchunk; \
        row = rowinchunk + (start_chunk)*CHUNKHEIGHT; \
        if(bval != NULL) { \
            for(int block=0; block<NVECS; ++block) { \
                scal[block] = -bval[NVECS*row+block]; \
            } \
        } \
        \
        for (ghost_lidx j=0; j<mat->rowLen[row]; ++j) { \
            MT mval_idx = mval[idx]; \
            for(int block=0; block<NVECS; ++block) { \
                scal[block] += mval_idx * xval[NVECS*mat->col[idx]+block]; \
            } \
            rownorm += std::norm(mval_idx); \
            idx+=CHUNKHEIGHT; \
        } \
        for(int block=0; block<NVECS; ++block) { \
            scal[block] /= (MT)rownorm; \
            scal[block] *= omega; \
        } \
        \
        idx -= CHUNKHEIGHT*mat->rowLen[row]; \
        \
        _Pragma("simd vectorlength(4)") \
        for (ghost_lidx j=0; j<mat->rowLen[row]; j++) { \
            for(int block=0; block<NVECS; ++block) { \
                xval[NVECS*mat->col[idx]+block] = xval[NVECS*mat->col[idx]+block] - scal[block] * std::conj(mval[idx]);\
            } \
            idx += CHUNKHEIGHT; \
        } \
    } \
    /*_Pragma("omp parallel for private(chunk, rowinchunk, idx, row)") */\
    for (chunk=start_chunk+1; chunk<end_chunk; ++chunk){ \
        for(rowinchunk=0; rowinchunk<CHUNKHEIGHT; ++rowinchunk) { \
            MT rownorm = 0.; \
            MT scal[NVECS] = {0}; \
            idx = mat->chunkStart[chunk] + rowinchunk; \
            row = rowinchunk + chunk*CHUNKHEIGHT; \
            if(bval != NULL) { \
                for(int block=0; block<NVECS; ++block) { \
                    scal[block] = -bval[NVECS*row+block]; \
                } \
            } \
            for (ghost_lidx j=0; j<mat->rowLen[row]; ++j) { \
                MT mval_idx = (MT)mval[idx]; \
                for(int block=0; block<NVECS; ++block) { \
                    scal[block] += (MT)mval_idx * xval[NVECS*mat->col[idx]+block];\
                } \
                rownorm += std::norm(mval_idx); \
                idx+=CHUNKHEIGHT; \
            } \
            for(int block=0; block<NVECS; ++block){ \
                scal[block] /= (MT)rownorm; \
                scal[block] *= omega; \
            } \
            idx -= CHUNKHEIGHT*mat->rowLen[row]; \
            \
            _Pragma("simd vectorlength(4)") \
            for (ghost_lidx j=0; j<mat->rowLen[row]; j++) { \
                for(int block=0; block<NVECS; ++block) { \
                    xval[NVECS*mat->col[idx]+block] = xval[NVECS*mat->col[idx]+block] - scal[block] * std::conj(mval[idx]);\
                } \
                idx += CHUNKHEIGHT; \
            } \
        } \
    } \
    if(start_chunk<end_chunk) { \
        for(rowinchunk=0; rowinchunk<end_rem; ++rowinchunk) { \
            MT rownorm = 0.; \
            MT scal[NVECS] = {0}; \
            idx = mat->chunkStart[end_chunk] + rowinchunk; \
            row = rowinchunk + (end_chunk)*CHUNKHEIGHT; \
            if(bval != NULL) { \
                for(int block=0; block<NVECS; ++block) { \
                    scal[block] = -bval[NVECS*row+block]; \
                } \
            } \
            for (ghost_lidx j=0; j<mat->rowLen[row]; ++j) { \
                MT mval_idx = (MT)mval[idx]; \
                for(int block=0; block<NVECS; ++block) { \
                    scal[block] += (MT)mval_idx * xval[NVECS*mat->col[idx]+block];\
                } \
                rownorm += std::norm(mval_idx); \
                idx+=CHUNKHEIGHT; \
            } \
            for(int block=0; block<NVECS; ++block){ \
                scal[block] /= (MT)rownorm; \
                scal[block] *= omega; \
            } \
            idx -= CHUNKHEIGHT*mat->rowLen[row]; \
            \
            _Pragma("simd vectorlength(4)") \
            for (ghost_lidx j=0; j<mat->rowLen[row]; j++) { \
                for(int block=0; block<NVECS; ++block) { \
                    xval[NVECS*mat->col[idx]+block] = xval[NVECS*mat->col[idx]+block] - scal[block] * std::conj(mval[idx]);\
                } \
                idx += CHUNKHEIGHT; \
            } \
        } \
    }
#endif


void ghost_spmv_RACE(ghost_densemat *b, ghost_sparsemat *mat, ghost_densemat *x)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);

    typedef double MT;
    typedef double VT;

    MT *bval = (MT *)(b->val);
    MT *xval = (MT *)(x->val);
    MT *mval = (MT *)(mat->val);

    //allocate col
    //    ghost_lidx *col =  (ghost_lidx*) malloc(sizeof(ghost_lidx)*mat->context->row_map->dim);
    LOOP(0, mat->context->row_map->dim, MT, VT);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
}
