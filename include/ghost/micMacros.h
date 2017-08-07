#ifndef GHOST_MIC_MACROS_H
#define GHOST_MIC_MACROS_H

#include <immintrin.h>
#include "avxMacros.h"

#if defined(GHOST_BUILD_MIC)


#define MIC_LOAD(src)\
    _mm512_load_pd(src)

#define MIC_STORE(dest, val)\
    _mm512_store_pd(dest, val)


#define MIC_GATHER(src, index)\
    _mm512_i32gather_pd(index, src, 8)\


#define MIC_SCATTER(dest, index, val)\
{\
    _mm512_i32scatter_pd(dest, index, val, 8);\
}\

//a*b+c
#define MIC_FMA(a,b,c)\
   _mm512_fmadd_pd(a, b,c)


inline bool testMICInstructions()
{
    int vecWidth = 8;
    double *a = (double*) malloc(vecWidth*sizeof(double));
    double *b = (double*) malloc(vecWidth*sizeof(double));
    int *mask = (int*) malloc(vecWidth*sizeof(int));
    bool testPass = true;
    for(int i=0; i<vecWidth; ++i)
    {
        a[i] = rand();
        mask[i] = (vecWidth-i-1) ;
    }

    __m512d a_vec = MIC_LOAD(a);
    MIC_STORE(b, a_vec);
    if(!(testEquality(a,b,vecWidth)))
    {
        ERROR_LOG("MIC LD/ST broken");
        testPass = false;
    }


    __m256i index = _mm256_load_si256((__m256i*)mask);
    a_vec = MIC_GATHER(a, index);
    MIC_SCATTER(b, index, a_vec);
    if(!(testEquality(a,b,vecWidth)))
    {
        ERROR_LOG("MIC GATHER/SCATTER broken");
        testPass = false;
    }

    free(a);
    free(b);
    free(mask);

    return testPass;
}


#else


#define MIC_LOAD(src)\

#define MIC_STORE(dest, val)\

#define MIC_GATHER(src, index)\

#define MIC_SCATTER(dest, index, val)\

#define MIC_FMA(a,b,c)\

#endif

/*
inline bool testEquality(double *a, double* b, int len)
{
    bool pass = true;
    for(int i=0; i<len; ++i)
    {
        if(a[i] != b[i])
        {
            pass = false;
            break;
        }
    }
    return pass;
}
*/

#endif
