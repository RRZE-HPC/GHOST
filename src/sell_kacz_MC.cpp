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
_Pragma("omp parallel for schedule(runtime)")\
for (ghost_lidx row=start; row<end; ++row){ \
    MT rownorm = 0.; \
    MT scal = 0; \
    ghost_lidx idx = mat->chunkStart[row]; \
    if(bval != NULL) { \
        scal = -bval[row]; \
    } \
    for (ghost_lidx j=0; j<mat->rowLen[row]; ++j) { \
        MT mval_idx = (MT)mval[idx+j]; \
        scal += mval_idx * xval[mat->col[idx+j]]; \
        rownorm += ghost::norm(mval_idx); \
    } \
    scal /= (MT)rownorm; \
    scal *= omega; \
    \
    _Pragma("simd vectorlength(VECLEN)") \
    for (ghost_lidx j=0; j<mat->rowLen[row]; j++) { \
        xval[mat->col[idx+j]] = xval[mat->col[idx+j]] - scal*(ghost::conj_or_nop(mval[idx+j]));\
    } \
} \


#else


#define LOOP(start,end,MT,VT) \
    GHOST_ERROR_LOG("TO Do")\

#endif

void ghost_kacz_MC(ghost_densemat *b, ghost_sparsemat *mat, ghost_densemat *x, int iterations)
{
#ifdef GHOST_HAVE_COLPACK
    typedef double MT;
    typedef double VT;

    MT *bval = (MT *)(b->val);
    MT *xval = (MT *)(x->val);
    MT *mval = (MT *)(mat->val);
    ghost_context* ctx = (ghost_context*) mat->context;
    MT omega=1;

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
                LOOP(ctx->color_ptr[color], ctx->color_ptr[color+1], MT, VT);
            }
        }

        GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    }
#else
    UNUSED(b);
    UNUSED(mat);
    UNUSED(x);
    UNUSED(iterations);

    GHOST_ERROR_LOG("Enable RACE library");
#endif
}


