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

#define MIC_GATHER_manual_with_var(src, idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8)\
    _mm512_setr_pd(src[idx1], src[idx2], src[idx3], src[idx4], src[idx5], src[idx6], src[idx7], src[idx8])\


#define MIC_SCATTER(dest, index, val)\
{\
    _mm512_i32scatter_pd(dest, index, val, 8);\
}\

#define MIC_SCATTER_manual_with_var(dest, idx1, idx2, idx3, idx4, idx5, idx6, idx7,idx8, val)\
{\
    __m256d lo = _mm512_extractf64x4_pd (val, 0);\
    __m256d hi = _mm512_extractf64x4_pd (val, 1);\
    __m128d part1 = _mm256_extractf128_pd(lo, 0);\
    __m128d part2 = _mm256_extractf128_pd(lo, 1);\
    __m128d part3 = _mm256_extractf128_pd(hi, 0);\
    __m128d part4 = _mm256_extractf128_pd(hi, 1);\
    _mm_store_sd(&(dest[idx1]), part1);\
    _mm_storeh_pd(&(dest[idx2]), part1);\
    _mm_store_sd(&(dest[idx3]), part2);\
    _mm_storeh_pd(&(dest[idx4]), part2);\
    _mm_store_sd(&(dest[idx5]), part3);\
    _mm_storeh_pd(&(dest[idx6]), part3);\
    _mm_store_sd(&(dest[idx7]), part4);\
    _mm_storeh_pd(&(dest[idx8]), part4);\
}\


#define SPLIT_M256i(m256i_reg, idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8)\
{\
    idx1 =  _mm256_extract_epi32 (m256i_reg, 0);\
    idx2 =  _mm256_extract_epi32 (m256i_reg, 1);\
    idx3 =  _mm256_extract_epi32 (m256i_reg, 2);\
    idx4 =  _mm256_extract_epi32 (m256i_reg, 3);\
    idx5 =  _mm256_extract_epi32 (m256i_reg, 4);\
    idx6 =  _mm256_extract_epi32 (m256i_reg, 5);\
    idx7 =  _mm256_extract_epi32 (m256i_reg, 6);\
    idx8 =  _mm256_extract_epi32 (m256i_reg, 7);\
}\


//a*b+c
#define MIC_FMA(a,b,c)\
  _mm512_fmadd_pd(a, b,c)

#define MIC_FMS(a,b,c)\
  _mm512_sub_pd(c,_mm512_mul_pd(a, b))

#define MIC_DIV(a, b)\
    _mm512_div_pd(a,b)


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
        GHOST_ERROR_LOG("MIC LD/ST broken");
        testPass = false;
    }


    __m256i index = _mm256_load_si256((__m256i*)mask);
    a_vec = MIC_GATHER(a, index);
    MIC_SCATTER(b, index, a_vec);
    if(!(testEquality(a,b,vecWidth)))
    {
        GHOST_ERROR_LOG("MIC GATHER/SCATTER broken");
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

#define MIC_FMS(a,b,c)\

#define MIC_DIV(a,b)\

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
