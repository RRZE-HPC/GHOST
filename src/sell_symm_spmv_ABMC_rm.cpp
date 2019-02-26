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


#if defined GHOST_BUILD_AVX512
    #define VECLEN 8
#elif defined (GHOST_BUILD_AVX2) || defined (GHOST_BUILD_AVX)
    #define VECLEN 4
#elif defined (GHOST_BUILD_SSE)
    #define VECLEN 2
#else
    #define VECLEN 1
#endif



#if (NVECS==1 && CHUNKHEIGHT==1)

//this is necessary since #pragma omp for doesn't understand !=
#define LOOP(start,end,MT,VT) \
_Pragma("omp parallel for schedule(runtime)")\
for (ghost_lidx part=start; part<end; ++part){ \
     for (ghost_lidx row=ctx->part_ptr[part]; row<ctx->part_ptr[part+1]; ++row){ \
        VT x_row = xval[row]; \
        ghost_lidx idx = mat->chunkStart[row]; \
        bval[row] += mval[idx]*x_row;\
        double temp=0;\
        _Pragma("simd reduction(+:temp) vectorlength(VECLEN)") \
        for (ghost_lidx j=1; j<mat->rowLen[row]; j++) { \
            temp = temp + (MT)mval[idx+j] * xval[mat->col[idx+j]];\
            bval[mat->col[idx+j]] = bval[mat->col[idx+j]] + (MT)mval[idx+j] * x_row;\
        } \
        bval[row]+=temp;\
    } \
}\


#elif CHUNKHEIGHT == 1

#define LOOP(start,end,MT,VT) \
    /*    _Pragma("omp parallel for") */\
    GHOST_ERROR_LOG("Not Implemented")

#else

#define LOOP(start,end,MT,VT) \
     GHOST_ERROR_LOG("Not Implemented")
#endif

void ghost_symm_spmv_ABMC(ghost_densemat *b, ghost_sparsemat *mat, ghost_densemat *x, int iterations)
{
#ifdef GHOST_HAVE_COLPACK
    typedef double MT;
    typedef double VT;

    MT *bval = (MT *)(b->val);
    MT *xval = (MT *)(x->val);
    MT *mval = (MT *)(mat->val);
    ghost_context* ctx = mat->context;

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

