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
#include "ghost/cpp11_fixes.h"

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
//only Forward LOOP

#define LOOP_preMtx(start,end,MT,VT) \
for (ghost_lidx row=start; row<end; ++row){ \
    MT rownorm = 0.; \
    MT scal = 0; \
    ghost_lidx idx = mat->chunkStart[row]; \
    ghost_lidx rowLen = mat->chunkStart[row+1] - mat->chunkStart[row];\
    if(bval != NULL) { \
        scal = -bval[row]; \
    } \
    _Pragma("nounroll")\
    for (ghost_lidx j=0; j<rowLen; ++j) { \
        scal += (MT)mval[idx+j] * xval[mat->col[idx+j]]; \
    } \
    /*scal *= -1;;*/ \
    /*scal *= omega;*/ \
    \
    _Pragma("simd")\
    for (ghost_lidx j=0; j<rowLen; j++) { \
        xval[mat->col[idx+j]] = xval[mat->col[idx+j]] - scal*(ghost::conj_or_nop(mval[idx+j]));\
    } \
} \




#else
#define LOOP_preMtx(start,end,MT,VT) \
    GHOST_ERROR_LOG("Not Implemented")

#endif

struct KACZ_ARG {
    ghost_densemat *b;
    ghost_sparsemat *mat;
    ghost_densemat *x;
    double* omega;
};

inline void KACZ_preMtx_Kernel(int start, int end, void *args)
{
    KACZ_ARG* kaczArg = (KACZ_ARG *) args;
    typedef double MT;
    //typedef double VT;

    MT *bval = (MT *)(kaczArg->b->val);
    MT *xval = (MT *)(kaczArg->x->val);
    MT *mval = (MT *)(kaczArg->mat->val);
    MT omega = *((MT*)(kaczArg->omega));
    ghost_sparsemat *mat = kaczArg->mat;

    LOOP_preMtx(start, end, MT, VT);
}

//static ghost_error ghost_kacz_u_plain_rm_CHUNKHEIGHT_NVECS_tmpl(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts)
void ghost_kacz_preMtx_RACE(ghost_densemat *b, ghost_sparsemat *mat, ghost_densemat *x, int iterations)
{
#ifdef GHOST_HAVE_RACE
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);

    RACE::Interface *ce = (RACE::Interface*) (mat->context->coloringEngine);

    double omega_ = 1;
    KACZ_ARG *kaczArg = new KACZ_ARG;
    kaczArg->mat = mat;
    kaczArg->b = b;
    kaczArg->x = x;
    kaczArg->omega = &omega_;

    void* argPtr = (void*) (kaczArg);
    int kaczId = ce->registerFunction(&KACZ_preMtx_Kernel, argPtr);

    //  std::vector<double> barrierTime;
    for(int i=0; i<iterations; ++i)
    {
        ce->executeFunction(kaczId);
    }

    /*    for(int i=0; i<iterations; ++i)
          {
          printf("%d \t%f \t%f \t%f\n", i, time[i], barrierTime[i], time[i]-(barrierTime[i]));
          }
          */
    delete kaczArg;
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
#else
    UNUSED(b);
    UNUSED(mat);
    UNUSED(x);
    UNUSED(iterations);

    GHOST_ERROR_LOG("Enable RACE library");
#endif
}
