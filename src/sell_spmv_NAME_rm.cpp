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

//#GHOST_SUBST NVECS ${BLOCKDIM1}
//#GHOST_SUBST CHUNKHEIGHT ${CHUNKHEIGHT}
#define CHUNKHEIGHT 1
#define NVECS 1

#if (NVECS==1 && CHUNKHEIGHT==1)
//this is necessary since #pragma omp for doesn't understand !=
#define LOOP(start,end,MT,VT) \
_Pragma("omp parallel for schedule(static)") \
for (ghost_lidx row=start; row<end; ++row){ \
    bval[row] = 0; \
    ghost_lidx idx = mat->chunkStart[row]; \
    for (ghost_lidx j=0; j<mat->rowLen[row]; j++) { \
        bval[row] = bval[row] + (MT)mval[idx+j] * xval[mat->col[idx+j]];\
    } \
} \

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


void ghost_spmv_NAME(ghost_densemat *b, ghost_sparsemat *mat, ghost_densemat *x)
{
	GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);

	typedef double MT;
	typedef double VT;

	MT *bval = (MT *)(b->val);
	MT *xval = (MT *)(x->val);
	MT *mval = (MT *)(mat->val);
	
	LOOP(0, mat->context->row_map->dim, MT, VT);
	GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
}
