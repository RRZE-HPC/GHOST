#ifndef GHOST_AVX2_MACROS_H
#define GHOST_AVX2_MACROS_H

#include <immintrin.h>
#include "avxMacros.h"

#if defined(GHOST_BUILD_AVX2)
//Alignd load
#define AVX2_LOAD(src)\
    _mm256_load_pd(src)

#define AVX2_STORE(dest, val)\
    _mm256_store_pd(dest, val)


#define AVX2_GATHER(src, mask)\
    _mm256_i32gather_pd(src, _mm_load_si128((__m128i*) mask), 8)\


#define AVX2_GATHER_with_128index(src, mask_reg)\
    _mm256_setr_pd(src[ _mm_cvtsi128_si32(mask_reg)], src[_mm_extract_epi32(mask_reg,1)], src[_mm_extract_epi32(mask_reg,2)], src[_mm_extract_epi32(mask_reg,3)])\

/*
_mm256_i32gather_pd(src,  mask_reg, 8)\
*/

#define spl_AVX2_GATHER_with_addr(mask1, mask2, mask3, mask4)\
    _mm256_setr_pd((*mask1), (*mask2), (*mask3), (*mask4))\



#define AVX2_SCATTER(dest, mask, val)\
{\
   __m128d lp128 = _mm256_extractf128_pd(val, 0);\
   _mm_store_sd(&(dest[*(mask)]), lp128);\
   _mm_storeh_pd(&(dest[*((mask)+1)]), lp128);\
   __m128d hp128 = _mm256_extractf128_pd(val, 1);\
   _mm_store_sd(&(dest[*((mask)+2)]), hp128);\
   _mm_storeh_pd(&(dest[*((mask)+3)]), hp128);\
}\

#define AVX2_SCATTER_with_128index(dest, mask_reg, val)\
{\
    __m128d lp128 = _mm256_extractf128_pd(val, 0);\
    _mm_store_sd(&(dest[ _mm_cvtsi128_si32(mask_reg)]), lp128);\
    _mm_storeh_pd(&(dest[_mm_extract_epi32(mask_reg,1)]), lp128);\
    __m128d hp128 = _mm256_extractf128_pd(val, 1);\
    _mm_store_sd(&(dest[_mm_extract_epi32(mask_reg,2)]), hp128);\
    _mm_storeh_pd(&(dest[_mm_extract_epi32(mask_reg,3)]), hp128);\
}

/*{\
    int mask1 = _mm_cvtsi128_si32(mask_reg);\
    int mask2 = _mm_extract_epi32(mask_reg,1);\
    int mask3 = _mm_extract_epi32(mask_reg,2);\
    int mask4 = _mm_extract_epi32(mask_reg,3);\
    __m128d lp128 = _mm256_extractf128_pd(val, 0);\
    _mm_store_sd(&(dest[mask1]), lp128);\
    _mm_storeh_pd(&(dest[mask2]), lp128);\
    __m128d hp128 = _mm256_extractf128_pd(val, 1);\
    _mm_store_sd(&(dest[mask3]), hp128);\
    _mm_storeh_pd(&(dest[mask4]), hp128);\
}*/

#define spl_AVX2_SCATTER_with_addr(mask1, mask2, mask3, mask4, val)\
{\
    __m128d lp128 = _mm256_extractf128_pd(val, 0);\
    _mm_store_sd(mask1, lp128);\
    _mm_storeh_pd(mask2, lp128);\
    __m128d hp128 = _mm256_extractf128_pd(val, 1);\
    _mm_store_sd(mask3, hp128);\
    _mm_storeh_pd(mask4, hp128);\
}


#define SPLIT_M128i(m128i_reg, val1, val2, val3, val4)\
    val1 = _mm_cvtsi128_si32(m128i_reg);\
    val2 = _mm_extract_epi32(m128i_reg,1);\
    val3 = _mm_extract_epi32(m128i_reg,2);\
    val4 = _mm_extract_epi32(m128i_reg,3);\

#define SPLIT_M256i(m256i_reg, val1, val2, val3, val4)\
{\
    __m128i val_lo =  _mm256_extracti128_si256(m256i_reg, 0);\
    __m128i val_hi =  _mm256_extracti128_si256(m256i_reg, 0);\
    val1 = _mm_cvtsi128_si64(val_lo);\
    val3 = _mm_cvtsi128_si64(val_hi);\
    val2 = _mm_extract_epi64(val_lo,1);\
    val4 = _mm_extract_epi32(val_hi,1);\
}

#define spl_SPLIT_M128i(val, m128i_reg, val1, val2, val3, val4)\
{\
    val1 = &(val[_mm_cvtsi128_si32(m128i_reg)]);\
    val2 = &(val[_mm_extract_epi32(m128i_reg,1)]);\
    val3 = &(val[_mm_extract_epi32(m128i_reg,2)]);\
    val4 = &(val[_mm_extract_epi32(m128i_reg,3)]);\
}\

//a*b+c
#define AVX2_FMA(a,b,c)\
    _mm256_fmadd_pd(a, b,c)

#define AVX2_ADD(a,b)\
    _mm256_add_pd(a, b)

#define AVX2_MUL(a,b)\
    _mm256_mul_pd(a, b)



inline bool testAVX2Instructions()
{
    double *a = (double*) malloc(4*sizeof(double));
    double *b = (double*) malloc(4*sizeof(double));
    int *mask = (int*) malloc(4*sizeof(int));
    bool testPass = true;
    for(int i=0; i<4; ++i)
    {
        a[i] = rand();
        mask[i] = (4-i-1) ;
    }

    __m256d a_vec = AVX2_LOAD(a);
    AVX2_STORE(b, a_vec);
    if(!(testEquality(a,b,4)))
    {
        GHOST_ERROR_LOG("AVX2 LD/ST broken");
        testPass = false;
    }


    a_vec = AVX2_GATHER(a, mask);
    AVX2_SCATTER(b, mask, a_vec);
    if(!(testEquality(a,b,4)))
    {
        GHOST_ERROR_LOG("AVX2 GATHER/SCATTER broken");
        testPass = false;
    }

    __m128i mask128 = _mm_load_si128( (__m128i*) mask);
    a_vec = AVX2_GATHER_with_128index(a, mask128);
    AVX2_SCATTER_with_128index(b, mask128, a_vec);
    if(!(testEquality(a,b,4)))
    {
        GHOST_ERROR_LOG("AVX2 GATHER_with_addr/SCATTER_with_addr broken");
        testPass = false;
    }


    free(a);
    free(b);
    free(mask);

    return testPass;
}


#else

#define AVX2_LOAD(src)\

#define AVX2_STORE(dest, val)\

#define AVX2_GATHER(src, mask)\

#define AVX2_GATHER_with_addr(src, mask_reg)\

#define spl_AVX2_GATHER_with_addr(mask1, mask2, mask3, mask4)\


#define AVX2_SCATTER(dest, mask, val)\

#define AVX2_SCATTER_with_addr(dest, mask_reg, val)\

#define spl_AVX2_SCATTER_with_addr(mask1, mask2, mask3, mask4, val)\

#define SPLIT_M128i(m128i_reg, val1, val2, val3, val4)\

#define SPLIT_M256i(m256i_reg, val1, val2, val3, val4)\

#define spl_SPLIT_M128i(val, m128i_reg, val1, val2, val3, val4)\

//a*b+c
#define AVX2_FMA(a,b,c)\


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
