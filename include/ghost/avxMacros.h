#ifndef GHOST_AVX_MACROS_H
#define GHOST_AVX_MACROS_H

#include <immintrin.h>
//Alignd load
#define AVX_256_LOAD(src)\
    _mm256_load_pd(src)

#define AVX_256_STORE(dest, val)\
    _mm256_store_pd(dest, val)

#define AVX_256_GATHER(src, mask)\
    _mm256_set_pd(src[*((mask)+3)], src[*((mask)+2)], src[*((mask)+1)], src[*(mask)])


#define AVX_256_GATHER_with_addr(src, mask1, mask2, mask3, mask4)\
       _mm256_setr_pd(src[mask1], src[mask2], src[mask3], src[mask4])\



#define AVX_256_SCATTER(dest, mask, val)\
{\
   __m128d lp128 = _mm256_extractf128_pd(val, 0);\
   _mm_store_sd(&(dest[*(mask)]), lp128);\
   _mm_storeh_pd(&(dest[*((mask)+1)]), lp128);\
   __m128d hp128 = _mm256_extractf128_pd(val, 1);\
   _mm_store_sd(&(dest[*((mask)+2)]), hp128);\
   _mm_storeh_pd(&(dest[*((mask)+3)]), hp128);\
}\

#define AVX_256_SCATTER_with_addr(dest, mask1, mask2, mask3, mask4, val)\
{\
   __m128d lp128 = _mm256_extractf128_pd(val, 0);\
   _mm_store_sd(&(dest[mask1]), lp128);\
   _mm_storeh_pd(&(dest[mask2]), lp128);\
   __m128d hp128 = _mm256_extractf128_pd(val, 1);\
   _mm_store_sd(&(dest[mask3]), hp128);\
   _mm_storeh_pd(&(dest[mask4]), hp128);\
}\

//a*b+c
#define AVX_256_FMA(a,b,c)\
   _mm256_add_pd(_mm256_mul_pd (a, b),c)


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

#define _mm256_print(val)\
{\
    double temp[4];\
    __m128d lp128 = _mm256_extractf128_pd(val, 0);\
    _mm_store_sd(&(temp[0]), lp128);\
    _mm_storeh_pd(&(temp[1]), lp128);\
    __m128d hp128 = _mm256_extractf128_pd(val, 1);\
    _mm_store_sd(&(temp[2]), hp128);\
    _mm_storeh_pd(&(temp[3]), hp128);\
    printf("%f \t %f \t %f \t %f\n", temp[0], temp[1], temp[2], temp[3]);\
}\

inline bool testInstructions()
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

    __m256d a_vec = AVX_256_LOAD(a);
    AVX_256_STORE(b, a_vec);
    if(!(testEquality(a,b,4)))
    {
        ERROR_LOG("AVX256 LD/ST broken");
        testPass = false;
    }


    a_vec = AVX_256_GATHER(a, mask);
    AVX_256_SCATTER(b, mask, a_vec);
    if(!(testEquality(a,b,4)))
    {
        ERROR_LOG("AVX256 GATHER/SCATTER broken");
        testPass = false;
    }

    free(a);
    free(b);
    free(mask);

    return testPass;
}

#endif
