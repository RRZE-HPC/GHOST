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

#define CHUNKHEIGHT 1

#if defined GHOST_BUILD_AVX512
    #define VECLEN 8
#elif defined (GHOST_BUILD_AVX2) || defined (GHOST_BUILD_AVX)
    #define VECLEN 4
#elif defined (GHOST_BUILD_SSE)
    #define VECLEN 2
#else
    #define VECLEN 1
#endif


#if (CHUNKHEIGHT==1)

#define LOOP(start,end,MT,VT) \
    for (ghost_lidx row=start; row<end; ++row){ \
        double temp = bval[row]; \
        ghost_lidx idx = mat->chunkStart[row]; \
        _Pragma("simd reduction(+:temp)") \
        for (ghost_lidx j=0; j<mat->rowLen[row]; j++) { \
            temp -= (MT)mval[idx+j] * xval[mat->col[idx+j]];\
        }\
        xval[row] = temp/dval[row];\
    }\


#else


#define LOOP(start,end,MT,VT) \
    GHOST_ERROR_LOG("TO Do")\

#endif

struct TRSV_ARG {
    ghost_densemat *b;
    ghost_sparsemat *mat;
    ghost_densemat *diag;
    ghost_densemat *x;
};

inline void TRSV_Kernel(int start, int end, void *args)
{
    TRSV_ARG* trsvArg = (TRSV_ARG *) args;
    typedef double MT;
    //typedef double VT;

    MT *bval = (MT *)(trsvArg->b->val);
    MT *xval = (MT *)(trsvArg->x->val);
    MT *mval = (MT *)(trsvArg->mat->val);
    MT *dval = (MT *)(trsvArg->diag->val);

    ghost_sparsemat *mat = trsvArg->mat;

    LOOP(start, end, MT, VT);
}


void ghost_trsv_RACE_fallback(ghost_densemat *b, ghost_sparsemat *mat, ghost_densemat* diag, ghost_densemat *x, int iterations)
{
#ifdef GHOST_HAVE_RACE
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    RACE::Interface *ce = (RACE::Interface*) (mat->context->coloringEngine);

    TRSV_ARG *trsvArg = new TRSV_ARG;
    trsvArg->mat = mat;
    trsvArg->diag = diag;
    trsvArg->b = b;
    trsvArg->x = x;

    void* argPtr = (void*) (trsvArg);

    int trsvId;
    trsvId = ce->registerFunction(&TRSV_Kernel, argPtr);

    for(int i=0; i<iterations; ++i)
    {
        ce->executeFunction(trsvId);
    }

    delete trsvArg;
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
#else
    UNUSED(b);
    UNUSED(mat);
    UNUSED(x);
    UNUSED(iterations);

    GHOST_ERROR_LOG("Enable RACE library");
#endif
}
