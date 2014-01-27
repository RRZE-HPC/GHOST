#include "ghost/config.h"
#undef GHOST_HAVE_MPI
#include "ghost/types.h"
#include "ghost/sell.h"
#include "ghost/constants.h"
#include "ghost/util.h"
#include <immintrin.h>

void dd_SELL_kernel_SSE (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * invec, int options)
{
#if GHOST_HAVE_SSE
    ghost_midx_t c,j;
    ghost_mnnz_t offs;
    double *mval = (double *)SELL(mat)->val;
    double *lval = lhs->val[0];
    double *rval = invec->val[0];
    __m128d tmp;
    __m128d val;
    __m128d rhs;

#pragma omp parallel for schedule(runtime) private(j,tmp,val,rhs,offs)
    for (c=0; c<SELL(mat)->nrowsPadded>>1; c++) 
    { // loop over chunks
        tmp = _mm_setzero_pd(); // tmp = 0
        offs = SELL(mat)->chunkStart[c];

        for (j=0; j<(SELL(mat)->chunkStart[c+1]-SELL(mat)->chunkStart[c])>>1; j++) 
        { // loop inside chunk
            val    = _mm_load_pd(&mval[offs]);                      // load values
            rhs    = _mm_loadl_pd(rhs,&rval[(SELL(mat)->col[offs++])]); // load first 128 bits of RHS
            rhs    = _mm_loadh_pd(rhs,&rval[(SELL(mat)->col[offs++])]);
            tmp    = _mm_add_pd(tmp,_mm_mul_pd(val,rhs));           // accumulate
        }
        if (options & GHOST_SPMVM_AXPY) {
            _mm_store_pd(&lval[c*2],_mm_add_pd(tmp,_mm_load_pd(&lval[c*2])));
        } else {
            _mm_stream_pd(&lval[c*2],tmp);
        }
    }
#else
    UNUSED(mat);
    UNUSED(lhs);
    UNUSED(invec);
    UNUSED(options);
#endif
}

void dd_SELL_kernel_AVX(ghost_mat_t *mat, ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions)
{
#if GHOST_HAVE_AVX
    ghost_midx_t c,j;
    ghost_mnnz_t offs;
    double *mval = (double *)SELL(mat)->val;
    double *lval = (double *)res->val[0];
    double *rval = (double *)invec->val[0];
    __m256d tmp;
    __m256d val;
    __m256d rhs;
    __m128d rhstmp;

#pragma omp parallel for schedule(runtime) private(j,tmp,val,offs,rhs,rhstmp)
    for (c=0; c<SELL(mat)->nrowsPadded>>2; c++) 
    { // loop over chunks
        tmp = _mm256_setzero_pd(); // tmp = 0
        offs = SELL(mat)->chunkStart[c];

        for (j=0; j<(SELL(mat)->chunkStart[c+1]-SELL(mat)->chunkStart[c])>>2; j++) 
        { // loop inside chunk

            val    = _mm256_load_pd(&mval[offs]);                      // load values
            rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load first 128 bits of RHS
            rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
            rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);                  // insert to RHS
            rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load second 128 bits of RHS
            rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
            rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);                  // insert to RHS
            tmp    = _mm256_add_pd(tmp,_mm256_mul_pd(val,rhs));           // accumulate
        }
        if (spmvmOptions & GHOST_SPMVM_AXPY) {
            _mm256_store_pd(&lval[c*SELL(mat)->chunkHeight],_mm256_add_pd(tmp,_mm256_load_pd(&lval[c*SELL(mat)->chunkHeight])));
        } else {
            _mm256_stream_pd(&lval[c*SELL(mat)->chunkHeight],tmp);
        }
    }
#else
    UNUSED(mat);
    UNUSED(res);
    UNUSED(invec);
    UNUSED(spmvmOptions);
#endif
}

void dd_SELL_kernel_AVX_32_rich(ghost_mat_t *mat, ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions)
{
#if GHOST_HAVE_AVX
    ghost_midx_t j,c;
    int nthreads = 1;
    ghost_mnnz_t offs;
    double *mval = (double *)SELL(mat)->val;
    double *lval = (double *)res->val[0];
    double *rval = (double *)invec->val[0];
    __m256d tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8;
    __m256d dot1,dot2,dot3;
    double dots1 = 0, dots2 = 0, dots3 = 0;
    __m256d val;
    __m256d rhs;
    __m128d rhstmp;
    __m256d shift, scale, beta;
        
    
    if (spmvmOptions & GHOST_SPMVM_APPLY_SHIFT)
        shift = _mm256_broadcast_sd(mat->traits->shift);
    if (spmvmOptions & GHOST_SPMVM_APPLY_SCALE)
        scale = _mm256_broadcast_sd(mat->traits->scale);
    if (spmvmOptions & GHOST_SPMVM_AXPBY)
        beta = _mm256_broadcast_sd(mat->traits->beta);


#pragma omp parallel private(c,j,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,val,offs,rhs,rhstmp,dot1,dot2,dot3) reduction (+:dots1,dots2,dots3)
    {
        dot1 = _mm256_setzero_pd();
        dot2 = _mm256_setzero_pd();
        dot3 = _mm256_setzero_pd();
#pragma omp for schedule(runtime)
        for (c=0; c<SELL(mat)->nrowsPadded>>5; c++) 
        { // loop over chunks
            tmp1 = _mm256_setzero_pd(); // tmp = 0
            tmp2 = _mm256_setzero_pd(); // tmp = 0
            tmp3 = _mm256_setzero_pd(); // tmp = 0
            tmp4 = _mm256_setzero_pd(); // tmp = 0
            tmp5 = _mm256_setzero_pd(); // tmp = 0
            tmp6 = _mm256_setzero_pd(); // tmp = 0
            tmp7 = _mm256_setzero_pd(); // tmp = 0
            tmp8 = _mm256_setzero_pd(); // tmp = 0
            offs = SELL(mat)->chunkStart[c];

            for (j=0; j<(SELL(mat)->chunkStart[c+1]-SELL(mat)->chunkStart[c])>>5; j++) 
            { // loop inside chunk

                val    = _mm256_load_pd(&mval[offs]);                      // load values
                rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load first 128 bits of RHS
                rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
                rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);                  // insert to RHS
                rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load second 128 bits of RHS
                rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
                rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);                  // insert to RHS
                tmp1    = _mm256_add_pd(tmp1,_mm256_mul_pd(val,rhs));           // accumulate
                
                val    = _mm256_load_pd(&mval[offs]);                      // load values
                rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load first 128 bits of RHS
                rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
                rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);                  // insert to RHS
                rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load second 128 bits of RHS
                rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
                rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);                  // insert to RHS
                tmp2    = _mm256_add_pd(tmp2,_mm256_mul_pd(val,rhs));           // accumulate
                
                val    = _mm256_load_pd(&mval[offs]);                      // load values
                rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load first 128 bits of RHS
                rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
                rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);                  // insert to RHS
                rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load second 128 bits of RHS
                rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
                rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);                  // insert to RHS
                tmp3    = _mm256_add_pd(tmp3,_mm256_mul_pd(val,rhs));           // accumulate
                
                val    = _mm256_load_pd(&mval[offs]);                      // load values
                rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load first 128 bits of RHS
                rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
                rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);                  // insert to RHS
                rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load second 128 bits of RHS
                rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
                rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);                  // insert to RHS
                tmp4    = _mm256_add_pd(tmp4,_mm256_mul_pd(val,rhs));           // accumulate

                val    = _mm256_load_pd(&mval[offs]);                      // load values
                rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load first 128 bits of RHS
                rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
                rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);                  // insert to RHS
                rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load second 128 bits of RHS
                rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
                rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);                  // insert to RHS
                tmp5    = _mm256_add_pd(tmp5,_mm256_mul_pd(val,rhs));           // accumulate
                
                val    = _mm256_load_pd(&mval[offs]);                      // load values
                rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load first 128 bits of RHS
                rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
                rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);                  // insert to RHS
                rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load second 128 bits of RHS
                rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
                rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);                  // insert to RHS
                tmp6    = _mm256_add_pd(tmp6,_mm256_mul_pd(val,rhs));           // accumulate
                
                val    = _mm256_load_pd(&mval[offs]);                      // load values
                rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load first 128 bits of RHS
                rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
                rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);                  // insert to RHS
                rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load second 128 bits of RHS
                rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
                rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);                  // insert to RHS
                tmp7    = _mm256_add_pd(tmp7,_mm256_mul_pd(val,rhs));           // accumulate
                
                val    = _mm256_load_pd(&mval[offs]);                      // load values
                rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load first 128 bits of RHS
                rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
                rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);                  // insert to RHS
                rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load second 128 bits of RHS
                rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
                rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);                  // insert to RHS
                tmp8    = _mm256_add_pd(tmp8,_mm256_mul_pd(val,rhs));           // accumulate
            }

            if (spmvmOptions & GHOST_SPMVM_APPLY_SHIFT) {
                tmp1 = _mm256_sub_pd(tmp1,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32])));
                tmp2 = _mm256_sub_pd(tmp2,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+4])));
                tmp3 = _mm256_sub_pd(tmp3,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+8])));
                tmp4 = _mm256_sub_pd(tmp4,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+12])));
                tmp5 = _mm256_sub_pd(tmp5,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+16])));
                tmp6 = _mm256_sub_pd(tmp6,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+20])));
                tmp7 = _mm256_sub_pd(tmp7,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+24])));
                tmp8 = _mm256_sub_pd(tmp8,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+28])));
            }
            if (spmvmOptions & GHOST_SPMVM_APPLY_SCALE) {
                tmp1 = _mm256_mul_pd(scale,tmp1);
                tmp2 = _mm256_mul_pd(scale,tmp2);
                tmp3 = _mm256_mul_pd(scale,tmp3);
                tmp4 = _mm256_mul_pd(scale,tmp4);
                tmp5 = _mm256_mul_pd(scale,tmp5);
                tmp6 = _mm256_mul_pd(scale,tmp6);
                tmp7 = _mm256_mul_pd(scale,tmp7);
                tmp8 = _mm256_mul_pd(scale,tmp8);
            }
            if (spmvmOptions & GHOST_SPMVM_AXPY) {
                _mm256_store_pd(&lval[c*32],_mm256_add_pd(tmp1,_mm256_load_pd(&lval[c*32])));
                _mm256_store_pd(&lval[c*32+4],_mm256_add_pd(tmp2,_mm256_load_pd(&lval[c*32+4])));
                _mm256_store_pd(&lval[c*32+8],_mm256_add_pd(tmp3,_mm256_load_pd(&lval[c*32+8])));
                _mm256_store_pd(&lval[c*32+12],_mm256_add_pd(tmp4,_mm256_load_pd(&lval[c*32+12])));
                _mm256_store_pd(&lval[c*32+16],_mm256_add_pd(tmp5,_mm256_load_pd(&lval[c*32+16])));
                _mm256_store_pd(&lval[c*32+20],_mm256_add_pd(tmp6,_mm256_load_pd(&lval[c*32+20])));
                _mm256_store_pd(&lval[c*32+24],_mm256_add_pd(tmp7,_mm256_load_pd(&lval[c*32+24])));
                _mm256_store_pd(&lval[c*32+28],_mm256_add_pd(tmp8,_mm256_load_pd(&lval[c*32+28])));
            } else if (spmvmOptions & GHOST_SPMVM_AXPBY) {
                _mm256_store_pd(&lval[c*32],_mm256_add_pd(tmp1,_mm256_mul_pd(beta,_mm256_load_pd(&lval[c*32]))));
                _mm256_store_pd(&lval[c*32+4],_mm256_add_pd(tmp2,_mm256_mul_pd(beta,_mm256_load_pd(&lval[c*32+4]))));
                _mm256_store_pd(&lval[c*32+8],_mm256_add_pd(tmp3,_mm256_mul_pd(beta,_mm256_load_pd(&lval[c*32+8]))));
                _mm256_store_pd(&lval[c*32+12],_mm256_add_pd(tmp4,_mm256_mul_pd(beta,_mm256_load_pd(&lval[c*32+12]))));
                _mm256_store_pd(&lval[c*32+16],_mm256_add_pd(tmp5,_mm256_mul_pd(beta,_mm256_load_pd(&lval[c*32+16]))));
                _mm256_store_pd(&lval[c*32+20],_mm256_add_pd(tmp6,_mm256_mul_pd(beta,_mm256_load_pd(&lval[c*32+20]))));
                _mm256_store_pd(&lval[c*32+24],_mm256_add_pd(tmp7,_mm256_mul_pd(beta,_mm256_load_pd(&lval[c*32+24]))));
                _mm256_store_pd(&lval[c*32+28],_mm256_add_pd(tmp8,_mm256_mul_pd(beta,_mm256_load_pd(&lval[c*32+28]))));
            } else {
                _mm256_stream_pd(&lval[c*32],tmp1);
                _mm256_stream_pd(&lval[c*32+4],tmp2);
                _mm256_stream_pd(&lval[c*32+8],tmp3);
                _mm256_stream_pd(&lval[c*32+12],tmp4);
                _mm256_stream_pd(&lval[c*32+16],tmp5);
                _mm256_stream_pd(&lval[c*32+20],tmp6);
                _mm256_stream_pd(&lval[c*32+24],tmp7);
                _mm256_stream_pd(&lval[c*32+28],tmp8);
            }
            if (spmvmOptions & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {
                dot1 = _mm256_add_pd(dot1,_mm256_mul_pd(_mm256_load_pd(&lval[c*32]),_mm256_load_pd(&lval[c*32])));
                dot1 = _mm256_add_pd(dot1,_mm256_mul_pd(_mm256_load_pd(&lval[c*32+4]),_mm256_load_pd(&lval[c*32+4])));
                dot1 = _mm256_add_pd(dot1,_mm256_mul_pd(_mm256_load_pd(&lval[c*32+8]),_mm256_load_pd(&lval[c*32+8])));
                dot1 = _mm256_add_pd(dot1,_mm256_mul_pd(_mm256_load_pd(&lval[c*32+12]),_mm256_load_pd(&lval[c*32+12])));
                dot1 = _mm256_add_pd(dot1,_mm256_mul_pd(_mm256_load_pd(&lval[c*32+16]),_mm256_load_pd(&lval[c*32+16])));
                dot1 = _mm256_add_pd(dot1,_mm256_mul_pd(_mm256_load_pd(&lval[c*32+20]),_mm256_load_pd(&lval[c*32+20])));
                dot1 = _mm256_add_pd(dot1,_mm256_mul_pd(_mm256_load_pd(&lval[c*32+24]),_mm256_load_pd(&lval[c*32+24])));
                dot1 = _mm256_add_pd(dot1,_mm256_mul_pd(_mm256_load_pd(&lval[c*32+28]),_mm256_load_pd(&lval[c*32+28])));
                
                dot2 = _mm256_add_pd(dot2,_mm256_mul_pd(_mm256_load_pd(&rval[c*32]),_mm256_load_pd(&lval[c*32])));
                dot2 = _mm256_add_pd(dot2,_mm256_mul_pd(_mm256_load_pd(&rval[c*32+4]),_mm256_load_pd(&lval[c*32+4])));
                dot2 = _mm256_add_pd(dot2,_mm256_mul_pd(_mm256_load_pd(&rval[c*32+8]),_mm256_load_pd(&lval[c*32+8])));
                dot2 = _mm256_add_pd(dot2,_mm256_mul_pd(_mm256_load_pd(&rval[c*32+12]),_mm256_load_pd(&lval[c*32+12])));
                dot2 = _mm256_add_pd(dot2,_mm256_mul_pd(_mm256_load_pd(&rval[c*32+16]),_mm256_load_pd(&lval[c*32+16])));
                dot2 = _mm256_add_pd(dot2,_mm256_mul_pd(_mm256_load_pd(&rval[c*32+20]),_mm256_load_pd(&lval[c*32+20])));
                dot2 = _mm256_add_pd(dot2,_mm256_mul_pd(_mm256_load_pd(&rval[c*32+24]),_mm256_load_pd(&lval[c*32+24])));
                dot2 = _mm256_add_pd(dot2,_mm256_mul_pd(_mm256_load_pd(&rval[c*32+28]),_mm256_load_pd(&lval[c*32+28])));
                
                dot3 = _mm256_add_pd(dot3,_mm256_mul_pd(_mm256_load_pd(&rval[c*32]),_mm256_load_pd(&rval[c*32])));
                dot3 = _mm256_add_pd(dot3,_mm256_mul_pd(_mm256_load_pd(&rval[c*32+4]),_mm256_load_pd(&rval[c*32+4])));
                dot3 = _mm256_add_pd(dot3,_mm256_mul_pd(_mm256_load_pd(&rval[c*32+8]),_mm256_load_pd(&rval[c*32+8])));
                dot3 = _mm256_add_pd(dot3,_mm256_mul_pd(_mm256_load_pd(&rval[c*32+12]),_mm256_load_pd(&rval[c*32+12])));
                dot3 = _mm256_add_pd(dot3,_mm256_mul_pd(_mm256_load_pd(&rval[c*32+16]),_mm256_load_pd(&rval[c*32+16])));
                dot3 = _mm256_add_pd(dot3,_mm256_mul_pd(_mm256_load_pd(&rval[c*32+20]),_mm256_load_pd(&rval[c*32+20])));
                dot3 = _mm256_add_pd(dot3,_mm256_mul_pd(_mm256_load_pd(&rval[c*32+24]),_mm256_load_pd(&rval[c*32+24])));
                dot3 = _mm256_add_pd(dot3,_mm256_mul_pd(_mm256_load_pd(&rval[c*32+28]),_mm256_load_pd(&rval[c*32+28])));
                
            }
        }
   
        if (spmvmOptions & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {
            __m256d sum12 = _mm256_hadd_pd(dot1,dot2);
            __m128d sum12high = _mm256_extractf128_pd(sum12,1);
            __m128d res12 = _mm_add_pd(sum12high, _mm256_castpd256_pd128(sum12));

            dots1 = ((double *)&res12)[0];
            dots2 = ((double *)&res12)[1];
            
            sum12 = _mm256_hadd_pd(dot3,dot3);
            sum12high = _mm256_extractf128_pd(sum12,1);
            res12 = _mm_add_pd(sum12high, _mm256_castpd256_pd128(sum12));
            dots3 = ((double *)&res12)[0];
        }
    }
    if (spmvmOptions & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {
        ((double *)(invec->traits->localdot))[0] = dots1;
        ((double *)(invec->traits->localdot))[1] = dots2;
        ((double *)(invec->traits->localdot))[2] = dots3;
    }
    
#else
    UNUSED(mat);
    UNUSED(res);
    UNUSED(invec);
    UNUSED(spmvmOptions);
#endif
}

void dd_SELL_kernel_AVX_32(ghost_mat_t *mat, ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions)
{
#if GHOST_HAVE_AVX
    ghost_midx_t c,j;
    ghost_mnnz_t offs;
    double *mval = (double *)SELL(mat)->val;
    double *lval = (double *)res->val[0];
    double *rval = (double *)invec->val[0];
    __m256d tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8;
    __m256d val;
    __m256d rhs;
    __m128d rhstmp;

#pragma omp parallel for schedule(runtime) private(j,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,val,offs,rhs,rhstmp)
    for (c=0; c<SELL(mat)->nrowsPadded>>5; c++) 
    { // loop over chunks
        tmp1 = _mm256_setzero_pd(); // tmp = 0
        tmp2 = _mm256_setzero_pd(); // tmp = 0
        tmp3 = _mm256_setzero_pd(); // tmp = 0
        tmp4 = _mm256_setzero_pd(); // tmp = 0
        tmp5 = _mm256_setzero_pd(); // tmp = 0
        tmp6 = _mm256_setzero_pd(); // tmp = 0
        tmp7 = _mm256_setzero_pd(); // tmp = 0
        tmp8 = _mm256_setzero_pd(); // tmp = 0
        offs = SELL(mat)->chunkStart[c];

        for (j=0; j<(SELL(mat)->chunkStart[c+1]-SELL(mat)->chunkStart[c])>>5; j++) 
        { // loop inside chunk

            val    = _mm256_load_pd(&mval[offs]);                      // load values
            rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load first 128 bits of RHS
            rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
            rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);                  // insert to RHS
            rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load second 128 bits of RHS
            rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
            rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);                  // insert to RHS
            tmp1    = _mm256_add_pd(tmp1,_mm256_mul_pd(val,rhs));           // accumulate
            
            val    = _mm256_load_pd(&mval[offs]);                      // load values
            rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load first 128 bits of RHS
            rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
            rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);                  // insert to RHS
            rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load second 128 bits of RHS
            rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
            rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);                  // insert to RHS
            tmp2    = _mm256_add_pd(tmp2,_mm256_mul_pd(val,rhs));           // accumulate
            
            val    = _mm256_load_pd(&mval[offs]);                      // load values
            rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load first 128 bits of RHS
            rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
            rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);                  // insert to RHS
            rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load second 128 bits of RHS
            rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
            rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);                  // insert to RHS
            tmp3    = _mm256_add_pd(tmp3,_mm256_mul_pd(val,rhs));           // accumulate
            
            val    = _mm256_load_pd(&mval[offs]);                      // load values
            rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load first 128 bits of RHS
            rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
            rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);                  // insert to RHS
            rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load second 128 bits of RHS
            rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
            rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);                  // insert to RHS
            tmp4    = _mm256_add_pd(tmp4,_mm256_mul_pd(val,rhs));           // accumulate

            val    = _mm256_load_pd(&mval[offs]);                      // load values
            rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load first 128 bits of RHS
            rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
            rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);                  // insert to RHS
            rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load second 128 bits of RHS
            rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
            rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);                  // insert to RHS
            tmp5    = _mm256_add_pd(tmp5,_mm256_mul_pd(val,rhs));           // accumulate
            
            val    = _mm256_load_pd(&mval[offs]);                      // load values
            rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load first 128 bits of RHS
            rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
            rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);                  // insert to RHS
            rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load second 128 bits of RHS
            rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
            rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);                  // insert to RHS
            tmp6    = _mm256_add_pd(tmp6,_mm256_mul_pd(val,rhs));           // accumulate
            
            val    = _mm256_load_pd(&mval[offs]);                      // load values
            rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load first 128 bits of RHS
            rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
            rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);                  // insert to RHS
            rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load second 128 bits of RHS
            rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
            rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);                  // insert to RHS
            tmp7    = _mm256_add_pd(tmp7,_mm256_mul_pd(val,rhs));           // accumulate
            
            val    = _mm256_load_pd(&mval[offs]);                      // load values
            rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load first 128 bits of RHS
            rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
            rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);                  // insert to RHS
            rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]); // load second 128 bits of RHS
            rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);
            rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);                  // insert to RHS
            tmp8    = _mm256_add_pd(tmp8,_mm256_mul_pd(val,rhs));           // accumulate
        }
        if (spmvmOptions & GHOST_SPMVM_AXPY) {
            _mm256_store_pd(&lval[c*SELL(mat)->chunkHeight],_mm256_add_pd(tmp1,_mm256_load_pd(&lval[c*SELL(mat)->chunkHeight])));
            _mm256_store_pd(&lval[c*SELL(mat)->chunkHeight+4],_mm256_add_pd(tmp2,_mm256_load_pd(&lval[c*SELL(mat)->chunkHeight+4])));
            _mm256_store_pd(&lval[c*SELL(mat)->chunkHeight+8],_mm256_add_pd(tmp3,_mm256_load_pd(&lval[c*SELL(mat)->chunkHeight+8])));
            _mm256_store_pd(&lval[c*SELL(mat)->chunkHeight+12],_mm256_add_pd(tmp4,_mm256_load_pd(&lval[c*SELL(mat)->chunkHeight+12])));
            _mm256_store_pd(&lval[c*SELL(mat)->chunkHeight+16],_mm256_add_pd(tmp5,_mm256_load_pd(&lval[c*SELL(mat)->chunkHeight+16])));
            _mm256_store_pd(&lval[c*SELL(mat)->chunkHeight+20],_mm256_add_pd(tmp6,_mm256_load_pd(&lval[c*SELL(mat)->chunkHeight+20])));
            _mm256_store_pd(&lval[c*SELL(mat)->chunkHeight+24],_mm256_add_pd(tmp7,_mm256_load_pd(&lval[c*SELL(mat)->chunkHeight+24])));
            _mm256_store_pd(&lval[c*SELL(mat)->chunkHeight+28],_mm256_add_pd(tmp8,_mm256_load_pd(&lval[c*SELL(mat)->chunkHeight+28])));
        } else {
            _mm256_stream_pd(&lval[c*SELL(mat)->chunkHeight],tmp1);
            _mm256_stream_pd(&lval[c*SELL(mat)->chunkHeight+4],tmp2);
            _mm256_stream_pd(&lval[c*SELL(mat)->chunkHeight+8],tmp3);
            _mm256_stream_pd(&lval[c*SELL(mat)->chunkHeight+12],tmp4);
            _mm256_stream_pd(&lval[c*SELL(mat)->chunkHeight+16],tmp5);
            _mm256_stream_pd(&lval[c*SELL(mat)->chunkHeight+20],tmp6);
            _mm256_stream_pd(&lval[c*SELL(mat)->chunkHeight+24],tmp7);
            _mm256_stream_pd(&lval[c*SELL(mat)->chunkHeight+28],tmp8);
        }
    }
#else
    UNUSED(mat);
    UNUSED(res);
    UNUSED(invec);
    UNUSED(spmvmOptions);
#endif
}

void dd_SELL_kernel_MIC_16(ghost_mat_t *mat, ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions)
{
#if GHOST_HAVE_MIC
    ghost_midx_t c,j;
    ghost_mnnz_t offs;
    double *mval = (double *)SELL(mat)->val;
    double *lval = (double *)res->val[0];
    double *rval = (double *)invec->val[0];
    __m512d tmp1;
    __m512d tmp2;
    __m512d val;
    __m512d rhs;
    __m512i idx;

#pragma omp parallel for schedule(runtime) private(j,tmp1,tmp2,idx,val,rhs,offs)
    for (c=0; c<SELL(mat)->nrowsPadded>>4; c++) 
    { // loop over chunks
        tmp1 = _mm512_setzero_pd(); // tmp1 = 0
        tmp2 = _mm512_setzero_pd(); // tmp2 = 0
        offs = SELL(mat)->chunkStart[c];

        for (j=0; j<(SELL(mat)->chunkStart[c+1]-SELL(mat)->chunkStart[c])>>4; j++) 
        { // loop inside chunk
            val = _mm512_load_pd(&mval[offs]);
            idx = _mm512_load_epi32(&SELL(mat)->col[offs]);
            rhs = _mm512_i32logather_pd(idx,rval,8);
//            rhs = _mm512_extload_pd(&rval[SELL(mat)->col[offs]],_MM_UPCONV_PD_NONE,_MM_BROADCAST_1X8,_MM_HINT_NONE);
            tmp1 = _mm512_add_pd(tmp1,_mm512_mul_pd(val,rhs));

            offs += 8;

            val = _mm512_load_pd(&mval[offs]);
            idx = _mm512_permute4f128_epi32(idx,_MM_PERM_BADC);
            rhs = _mm512_i32logather_pd(idx,rval,8);
//            rhs = _mm512_extload_pd(&rval[SELL(mat)->col[offs]],_MM_UPCONV_PD_NONE,_MM_BROADCAST_1X8,_MM_HINT_NONE);
            tmp2 = _mm512_add_pd(tmp2,_mm512_mul_pd(val,rhs));

            offs += 8;
        }
        if (spmvmOptions & GHOST_SPMVM_AXPY) {
            _mm512_store_pd(&lval[c*SELL(mat)->chunkHeight],_mm512_add_pd(tmp1,_mm512_load_pd(&lval[c*SELL(mat)->chunkHeight])));
            _mm512_store_pd(&lval[c*SELL(mat)->chunkHeight+8],_mm512_add_pd(tmp2,_mm512_load_pd(&lval[c*SELL(mat)->chunkHeight+8])));
        } else {
            _mm512_storenrngo_pd(&lval[c*SELL(mat)->chunkHeight],tmp1);
            _mm512_storenrngo_pd(&lval[c*SELL(mat)->chunkHeight+8],tmp2);
        }
    }
#else 
    UNUSED(mat);
    UNUSED(res);
    UNUSED(invec);
    UNUSED(spmvmOptions);
#endif
}

void dd_SELL_kernel_MIC_32(ghost_mat_t *mat, ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions)
{
#if GHOST_HAVE_MIC
    ghost_midx_t c,j;
    ghost_mnnz_t offs;
    double *mval = (double *)SELL(mat)->val;
    double *lval = (double *)res->val[0];
    double *rval = (double *)invec->val[0];
    __m512d tmp1;
    __m512d tmp2;
    __m512d tmp3;
    __m512d tmp4;
    __m512d val;
    __m512d rhs;
    __m512i idx;

#pragma omp parallel for schedule(runtime) private(j,tmp1,tmp2,tmp3,tmp4,idx,val,rhs,offs)
    for (c=0; c<SELL(mat)->nrowsPadded>>5; c++) 
    { // loop over chunks
        tmp1 = _mm512_setzero_pd(); // tmp1 = 0
        tmp2 = _mm512_setzero_pd(); // tmp2 = 0
        tmp3 = _mm512_setzero_pd(); // tmp3 = 0
        tmp4 = _mm512_setzero_pd(); // tmp4 = 0
        offs = SELL(mat)->chunkStart[c];

        for (j=0; j<(SELL(mat)->chunkStart[c+1]-SELL(mat)->chunkStart[c])>>5; j++) 
        { // loop inside chunk
//        _mm_prefetch(&((const char*)mval)[offs+512], _MM_HINT_T1);
//        _mm_prefetch(&((const char*)SELL(mat)->col)[offs+512], _MM_HINT_T1);
//        _mm_prefetch(&((const char*)mval)[offs+100000], _MM_HINT_NTA);
//        _mm_prefetch(&((const char *)SELL(mat)->col)[offs+500000], _MM_HINT_T0);
 
            val = _mm512_load_pd(&mval[offs]);
            idx = _mm512_load_epi32(&SELL(mat)->col[offs]);
            rhs = _mm512_i32logather_pd(idx,rval,8);
            tmp1 = _mm512_add_pd(tmp1,_mm512_mul_pd(val,rhs));

            offs += 8;

            val = _mm512_load_pd(&mval[offs]);
            idx = _mm512_permute4f128_epi32(idx,_MM_PERM_BADC);
            rhs = _mm512_i32logather_pd(idx,rval,8);
            tmp2 = _mm512_add_pd(tmp2,_mm512_mul_pd(val,rhs));

            offs += 8;
            
            val = _mm512_load_pd(&mval[offs]);
            idx = _mm512_load_epi32(&SELL(mat)->col[offs]);
            rhs = _mm512_i32logather_pd(idx,rval,8);
            tmp3 = _mm512_add_pd(tmp3,_mm512_mul_pd(val,rhs));

            offs += 8;

            val = _mm512_load_pd(&mval[offs]);
            idx = _mm512_permute4f128_epi32(idx,_MM_PERM_BADC);
            rhs = _mm512_i32logather_pd(idx,rval,8);
            tmp4 = _mm512_add_pd(tmp4,_mm512_mul_pd(val,rhs));

            offs += 8;
        }
        if (spmvmOptions & GHOST_SPMVM_AXPY) {
            _mm512_store_pd(&lval[c*SELL(mat)->chunkHeight],_mm512_add_pd(tmp1,_mm512_load_pd(&lval[c*SELL(mat)->chunkHeight])));
            _mm512_store_pd(&lval[c*SELL(mat)->chunkHeight+8],_mm512_add_pd(tmp2,_mm512_load_pd(&lval[c*SELL(mat)->chunkHeight+8])));
            _mm512_store_pd(&lval[c*SELL(mat)->chunkHeight+16],_mm512_add_pd(tmp3,_mm512_load_pd(&lval[c*SELL(mat)->chunkHeight+16])));
            _mm512_store_pd(&lval[c*SELL(mat)->chunkHeight+24],_mm512_add_pd(tmp4,_mm512_load_pd(&lval[c*SELL(mat)->chunkHeight+24])));
        } else {
            _mm512_storenrngo_pd(&lval[c*SELL(mat)->chunkHeight],tmp1);
            _mm512_storenrngo_pd(&lval[c*SELL(mat)->chunkHeight+8],tmp2);
            _mm512_storenrngo_pd(&lval[c*SELL(mat)->chunkHeight+16],tmp3);
            _mm512_storenrngo_pd(&lval[c*SELL(mat)->chunkHeight+24],tmp4);
        }
    }
#else 
    UNUSED(mat);
    UNUSED(res);
    UNUSED(invec);
    UNUSED(spmvmOptions);
#endif
}
