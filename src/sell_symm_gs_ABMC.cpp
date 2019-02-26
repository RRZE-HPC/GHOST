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
_Pragma("omp parallel for schedule(runtime)")\
for (ghost_lidx part=start; part<end; ++part){ \
    for (ghost_lidx row=ctx->part_ptr[part]; row<ctx->part_ptr[part+1]; ++row){ \
        double temp = bval[row]; \
        ghost_lidx idx = mat->chunkStart[row]; \
        _Pragma("simd reduction(+:temp)") \
        for (ghost_lidx j=1; j<mat->rowLen[row]; j++) { \
            temp -= (MT)mval[idx+j] * xval[mat->col[idx+j]];\
        }\
        xval[row] = temp/mval[idx];\
    }\
}\

#define LOOP_rev(start,end,MT,VT) \
_Pragma("omp parallel for schedule(runtime)")\
for (ghost_lidx part=end-1; part>=start; --part){ \
    for (ghost_lidx row=ctx->part_ptr[part+1]-1; row>=ctx->part_ptr[part]; --row){ \
        double temp = bval[row]; \
        ghost_lidx idx = mat->chunkStart[row]; \
        _Pragma("simd reduction(+:temp)") \
        for (ghost_lidx j=1; j<mat->rowLen[row]; j++) { \
            temp -= (MT)mval[idx+j] * xval[mat->col[idx+j]];\
        }\
        xval[row] = temp/mval[idx];\
    }\
}\


#else


#define LOOP(start,end,MT,VT) \
    GHOST_ERROR_LOG("TO Do")\

#endif

void ghost_symm_gs_ABMC(ghost_densemat *b, ghost_sparsemat *mat, ghost_densemat *x, int iterations)
{
#ifdef GHOST_HAVE_METIS
#ifdef GHOST_HAVE_COLPACK
    typedef double MT;
    typedef double VT;

    MT *bval = (MT *)(b->val);
    MT *xval = (MT *)(x->val);
    MT *mval = (MT *)(mat->val);
    ghost_context* ctx = (ghost_context*) mat->context;
    if( (mat->context->color_ptr==NULL) || (mat->context->ncolors==0) )
    {
        GHOST_ERROR_LOG("Matrix not colored")
    }
    else
    {
        GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);

        for(int iter=0; iter<iterations; ++iter)
        {
            for(int color=0; color<mat->context->ncolors; ++color)
            {
                LOOP(mat->context->color_ptr[color], mat->context->color_ptr[color+1], MT, VT);
            }
            for(int color=mat->context->ncolors-1; color>=0; --color)
            {
                LOOP_rev(mat->context->color_ptr[color], mat->context->color_ptr[color+1], MT, VT);
            }
        }

        GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    }
#else
    UNUSED(b);
    UNUSED(mat);
    UNUSED(x);
    UNUSED(iterations);

    GHOST_ERROR_LOG("Enable COLPACK library");
#endif
#else
    UNUSED(b);
    UNUSED(mat);
    UNUSED(x);
    UNUSED(iterations);

    GHOST_ERROR_LOG("Enable METIS library");
#endif
}
