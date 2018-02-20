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
#include "ghost/sparsemat.h"
#ifdef GHOST_HAVE_RACE
#include <RACE/interface.h>
#endif
#include <vector>

//#GHOST_SUBST NVECS ${BLOCKDIM1}
//#GHOST_SUBST CHUNKHEIGHT ${CHUNKHEIGHT}
#define CHUNKHEIGHT 1
#define NVECS 1

#if 0
#define INNER_LOOP_AVX512\
    __m256i vindex = _mm256_stream_load_si256( &((__m256i*)mat->col)[idx]); \
__m512d bvec = _mm512_i32gather_pd(vindex, (void*) bval, 1);\
__m512d mvec = _mm512_load_pd( &( mval[idx]));\
__m512d xvec = _mm512_set1_pd(x_row);\
bvec = _mm512_fmadd_pd(mvec, xvec, bvec);\
_mm512_i32scatter_pd((void*) bval,  vindex, bvec, 1);\
idx+=8;\




#define INNER_LOOP_REM\
    bval[mat->col[idx]] = bval[mat->col[idx]] + (MT)mval[idx] * x_row;\
idx+=1;\

#endif



#if (NVECS==1 && CHUNKHEIGHT==1)
//this is necessary since #pragma omp for doesn't understand !=
/*
#define LOOP(start,end,MT,VT) \
    for (ghost_lidx row=start; row<end; ++row){ \
        VT x_row = xval[row]; \
        ghost_lidx idx = mat->chunkStart[row]; \
        bval[row] += mval[idx]*x_row;\
        for(ghost_lidx j=1; j<mat->rowLen[row]; j++) { \
            bval[row] = bval[row] + (MT)mval[idx+j] * xval[mat->col[idx+j]];\
        }\
        _Pragma("nounroll")\
        _Pragma("simd")\
        for(ghost_lidx j=1; j<mat->rowLen[row]; j++) { \
            bval[mat->col[idx+j]] = bval[mat->col[idx+j]] + (MT)mval[idx+j] * x_row;\
        }\
    }\
*/

#define LOOP(start,end,MT,VT) \
        for (ghost_lidx row=start; row<end; ++row){ \
            VT x_row = xval[row]; \
            ghost_lidx idx = mat->chunkStart[row]; \
            bval[row] += mval[idx]*x_row;\
            double temp = 0;\
            _Pragma("simd reduction(+:temp) vectorlength(4)")\
            for(ghost_lidx j=1; j<mat->rowLen[row]; j++) { \
                temp = temp + (MT)mval[idx+j] * xval[mat->col[idx+j]];\
                bval[mat->col[idx+j]] = bval[mat->col[idx+j]] + (MT)mval[idx+j] * x_row;\
            }\
            bval[row]+=temp;\
        }\


#else 

#define LOOP(start,end,MT,VT) \
        GHOST_ERROR_LOG("Not defined");

#endif

    struct SYMM_SPMV_ARG {
        ghost_densemat *b;
        ghost_sparsemat *mat;
        ghost_densemat *x;
    };

inline void SYMM_SPMV_Kernel(int start, int end, void *args)
{
    SYMM_SPMV_ARG* symm_spmvArg = (SYMM_SPMV_ARG *) args;
    typedef double MT;
    typedef double VT;

    MT *bval = (MT *)(symm_spmvArg->b->val);
    MT *xval = (MT *)(symm_spmvArg->x->val);
    MT *mval = (MT *)(symm_spmvArg->mat->val);

    ghost_sparsemat *mat = symm_spmvArg->mat;

    LOOP(start, end, MT, VT);
}

void ghost_symm_spmv_RACE(ghost_densemat *b, ghost_sparsemat *mat, ghost_densemat *x, int iterations)
{
#ifdef GHOST_HAVE_RACE
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);

    RACE::Interface *ce = (RACE::Interface*) (mat->context->coloringEngine);

    SYMM_SPMV_ARG *symm_spmvArg = new SYMM_SPMV_ARG;
    symm_spmvArg->mat = mat;
    symm_spmvArg->b = b;
    symm_spmvArg->x = x;

    void* argPtr = (void*) (symm_spmvArg);
    int symm_spmvId = ce->registerFunction(&SYMM_SPMV_Kernel, argPtr);
    std::vector<double> time; 
    //    std::vector<double> barrierTime;
    double start_spmv_inner, end_spmv_inner;
    for(int i=0; i<iterations; ++i)
    {
        ghost_barrier();
        ghost_timing_wcmilli(&start_spmv_inner);

        ce->executeFunction(symm_spmvId);

        ghost_barrier();
        ghost_timing_wcmilli(&end_spmv_inner);

        time.push_back(end_spmv_inner-start_spmv_inner);
        //	    barrierTime.push_back(ce->barrierTime()*1e3);
    }

    /*    for(int i=0; i<iterations; ++i)
          {
          printf("%d \t%f \t%f \t%f\n", i, time[i], barrierTime[i], time[i]-(barrierTime[i]));
          }
          */
    delete symm_spmvArg;
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
#else
    UNUSED(b);
    UNUSED(mat);
    UNUSED(x);
    UNUSED(iterations);
    GHOST_ERROR_LOG("Enable RACE library");
#endif
}

