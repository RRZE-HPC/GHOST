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

//this is necessary since #pragma omp for doesn't understand !=
#define LOOP(start,end,MT,VT) \
    for (ghost_lidx row=start; row<end; ++row){ \
        double temp = bval[row]; \
        ghost_lidx idx = mat->chunkStart[row]; \
        _Pragma("simd reduction(+:temp)") \
        for (ghost_lidx j=1; j<mat->rowLen[row]; j++) { \
            temp -= (MT)mval[idx+j] * xval[mat->col[idx+j]];\
        }\
        xval[row] = temp/mval[idx];\
    }\


#define LOOP_rev(start,end,MT,VT) \
    for (ghost_lidx row=end-1; row>=start; --row){ \
        double temp = bval[row]; \
        ghost_lidx idx = mat->chunkStart[row]; \
        _Pragma("simd reduction(+:temp)") \
        for (ghost_lidx j=1; j<mat->rowLen[row]; j++) { \
            temp -= (MT)mval[idx+j] * xval[mat->col[idx+j]];\
        }\
        xval[row] = temp/mval[idx];\
    }\


#else


#define LOOP(start,end,MT,VT) \
    GHOST_ERROR_LOG("TO Do")\

#define LOOP_rev(start,end,MT,VT) \
    GHOST_ERROR_LOG("TO Do")\


#endif

struct GS_ARG {
    ghost_densemat *b;
    ghost_sparsemat *mat;
    ghost_densemat *x;
};

inline void GS_Kernel(int start, int end, void *args)
{
    GS_ARG* gsArg = (GS_ARG *) args;
    typedef double MT;
    //typedef double VT;

    MT *bval = (MT *)(gsArg->b->val);
    MT *xval = (MT *)(gsArg->x->val);
    MT *mval = (MT *)(gsArg->mat->val);

    ghost_sparsemat *mat = gsArg->mat;

    LOOP(start, end, MT, VT);
}

inline void GS_Kernel_rev(int start, int end, void *args)
{
    GS_ARG* gsArg = (GS_ARG *) args;
    typedef double MT;
    //typedef double VT;

    MT *bval = (MT *)(gsArg->b->val);
    MT *xval = (MT *)(gsArg->x->val);
    MT *mval = (MT *)(gsArg->mat->val);

    ghost_sparsemat *mat = gsArg->mat;

    LOOP_rev(start, end, MT, VT);
}


//static ghost_error ghost_gs_BMC_u_plain_rm_CHUNKHEIGHT_NVECS_tmpl(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts)
ghost_error ghost_symm_gs_RACE(ghost_densemat *b, ghost_sparsemat *mat, ghost_densemat *x, int iterations)
{
#ifdef GHOST_HAVE_RACE
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    if(!(mat->traits.flags & GHOST_SPARSEMAT_DIAG_FIRST))
    {
        GHOST_ERROR_LOG("Enable 'GHOST_SPARSEMAT_DIAG_FIRST' flag to execute Gauss-Seidel sweep");
    }
    else
    {
        RACE::Interface *ce = (RACE::Interface*) (mat->context->coloringEngine);

        GS_ARG *gsArg = new GS_ARG;
        gsArg->mat = mat;
        gsArg->b = b;
        gsArg->x = x;

        void* argPtr = (void*) (gsArg);

        int gsId = ce->registerFunction(&GS_Kernel, argPtr);
        int gsId_rev  = ce->registerFunction(&GS_Kernel_rev, argPtr);

        std::vector<double> time;
        //  std::vector<double> barrierTime;
        double start_gs_inner, end_gs_inner;
        for(int i=0; i<iterations; ++i)
        {
            ghost_barrier();
            ghost_timing_wcmilli(&start_gs_inner);

            ce->executeFunction(gsId);
            ce->executeFunction(gsId_rev, true);

            ghost_barrier();
            ghost_timing_wcmilli(&end_gs_inner);

            time.push_back(end_gs_inner-start_gs_inner);
        }

        /*    for(int i=0; i<iterations; ++i)
              {
              printf("%d \t%f \t%f \t%f\n", i, time[i], barrierTime[i], time[i]-(barrierTime[i]));
              }
              */
        delete gsArg;
    }
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);

    return GHOST_SUCCESS;
#else
    UNUSED(b);
    UNUSED(mat);
    UNUSED(x);
    UNUSED(iterations);

    GHOST_ERROR_LOG("Enable RACE library");
    return GHOST_ERR_UNKNOWN;
#endif
}

