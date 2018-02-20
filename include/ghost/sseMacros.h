#ifndef GHOST_SSE_MACROS_H
#define GHOST_SSE_MACROS_H

#include <immintrin.h>
//Alignd load
#define SSE_LOAD(src)\
    _mm_load_pd(src)

#define SSE_STORE(dest, val)\
    _mm_store_pd(dest, val)

#define SSE_GATHER(src, mask)\
    _mm_loadh_pd(_mm_load_sd(&(src[*mask])), &(src[*(mask+1)]))

#define SSE_GATHER_with_addr(src, mask_lo, mask_hi)\
    _mm_loadh_pd(_mm_load_sd(&(src[mask_lo])), (&(src[mask_hi])))


#define SSE_SCATTER(dest, mask, val)\
    _mm_store_sd(&(dest[*mask]), val);\
    _mm_storeh_pd(&(dest[*(mask+1)]), val);\

#define SSE_SCATTER_with_addr(dest, mask_lo, mask_hi, val)\
     _mm_store_sd(&(dest[mask_lo]), val);\
    _mm_storeh_pd(&(dest[mask_hi]), val);\

#define SPLIT_M128i(m128i_reg, idx1, idx2)\
{\
    idx1 = _mm_cvtsi128_si32 (m128i_reg);\
    idx2 = _mm_extract_epi32 (m128i_reg, 1);\
}


//a*b+c
#define SSE_FMA(a,b,c)\
    _mm_add_pd(_mm_mul_pd(a, b),c)

#define SSE_MUL(a,b)\
    _mm_mul_pd(a, b)

#define SSE_ADD(a,b)\
    _mm_add_pd(a, b)


#endif
