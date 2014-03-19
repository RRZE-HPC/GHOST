#include "ghost/config.h"
#undef GHOST_HAVE_MPI
#include "ghost/types.h"
#include "ghost/sell.h"
#include "ghost/util.h"
#include "ghost/instr.h"
#include <immintrin.h>

ghost_error_t dd_SELL_kernel_SSE (ghost_sparsemat_t *mat, ghost_densemat_t * lhs, ghost_densemat_t * invec, ghost_spmv_flags_t options,va_list argp)
{
    UNUSED(argp);
#ifdef GHOST_HAVE_SSE
    ghost_idx_t c,j;
    ghost_nnz_t offs;
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
        if (options & GHOST_SPMV_AXPY) {
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
    return GHOST_SUCCESS;
}

ghost_error_t dd_SELL_kernel_AVX(ghost_sparsemat_t *mat, ghost_densemat_t* res, ghost_densemat_t* invec, ghost_spmv_flags_t spmvmOptions,va_list argp)
{
    UNUSED(argp);
#ifdef GHOST_HAVE_AVX
    ghost_idx_t c,j;
    ghost_nnz_t offs;
    double *mval = (double *)SELL(mat)->val;
    double *lval = (double *)res->val[0];
    double *rval = (double *)invec->val[0];
    __m256d tmp;
    __m256d val;
    __m256d rhs;
    __m128d rhstmp;

#pragma omp parallel for schedule(runtime) private(j,tmp,val,offs,rhs,rhstmp)
    for (c=0; c<mat->nrowsPadded>>2; c++) 
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
        if (spmvmOptions & GHOST_SPMV_AXPY) {
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
    return GHOST_SUCCESS;
}

ghost_error_t dd_SELL_kernel_AVX_32_rich(ghost_sparsemat_t *mat, ghost_densemat_t* res, ghost_densemat_t* invec, ghost_spmv_flags_t spmvmOptions,va_list argp)
{
#ifdef GHOST_HAVE_AVX
    ghost_idx_t j,c,v;
    ghost_nnz_t offs;
    double *lval = NULL, *rval = NULL;
    double *mval = (double *)SELL(mat)->val;
    double *local_dot_product = NULL;
    __m256d tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8;
    __m256d dot1,dot2,dot3;
    double dots1 = 0, dots2 = 0, dots3 = 0;
    __m256d val;
    __m256d rhs;
    __m128d rhstmp;
    __m256d shift, scale, beta;

    if (spmvmOptions & GHOST_SPMV_SCALE) {
        scale = _mm256_broadcast_sd(va_arg(argp,double *));
    }
    if (spmvmOptions & GHOST_SPMV_AXPBY) {
        beta = _mm256_broadcast_sd(va_arg(argp,double *));
    }
    if (spmvmOptions & GHOST_SPMV_SHIFT) {
        shift = _mm256_broadcast_sd(va_arg(argp,double *));
    }
    if (spmvmOptions & GHOST_SPMV_DOT) {
        local_dot_product = va_arg(argp,double *);
    }


#pragma omp parallel private(v,c,j,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,val,offs,rhs,rhstmp,dot1,dot2,dot3) reduction (+:dots1,dots2,dots3)
    {
        dot1 = _mm256_setzero_pd();
        dot2 = _mm256_setzero_pd();
        dot3 = _mm256_setzero_pd();
#pragma omp for schedule(runtime)
        for (c=0; c<mat->nrowsPadded>>5; c++) 
        { // loop over chunks

            for (v=0; v<invec->traits.ncols; v++)
            {
                tmp1 = _mm256_setzero_pd(); // tmp = 0
                tmp2 = _mm256_setzero_pd(); // tmp = 0
                tmp3 = _mm256_setzero_pd(); // tmp = 0
                tmp4 = _mm256_setzero_pd(); // tmp = 0
                tmp5 = _mm256_setzero_pd(); // tmp = 0
                tmp6 = _mm256_setzero_pd(); // tmp = 0
                tmp7 = _mm256_setzero_pd(); // tmp = 0
                tmp8 = _mm256_setzero_pd(); // tmp = 0
                lval = (double *)res->val[v];
                rval = (double *)invec->val[v];
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

                if (spmvmOptions & GHOST_SPMV_SHIFT) {
                    tmp1 = _mm256_sub_pd(tmp1,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32])));
                    tmp2 = _mm256_sub_pd(tmp2,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+4])));
                    tmp3 = _mm256_sub_pd(tmp3,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+8])));
                    tmp4 = _mm256_sub_pd(tmp4,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+12])));
                    tmp5 = _mm256_sub_pd(tmp5,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+16])));
                    tmp6 = _mm256_sub_pd(tmp6,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+20])));
                    tmp7 = _mm256_sub_pd(tmp7,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+24])));
                    tmp8 = _mm256_sub_pd(tmp8,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+28])));
                }
                if (spmvmOptions & GHOST_SPMV_SCALE) {
                    tmp1 = _mm256_mul_pd(scale,tmp1);
                    tmp2 = _mm256_mul_pd(scale,tmp2);
                    tmp3 = _mm256_mul_pd(scale,tmp3);
                    tmp4 = _mm256_mul_pd(scale,tmp4);
                    tmp5 = _mm256_mul_pd(scale,tmp5);
                    tmp6 = _mm256_mul_pd(scale,tmp6);
                    tmp7 = _mm256_mul_pd(scale,tmp7);
                    tmp8 = _mm256_mul_pd(scale,tmp8);
                }
                if (spmvmOptions & GHOST_SPMV_AXPY) {
                    _mm256_store_pd(&lval[c*32],_mm256_add_pd(tmp1,_mm256_load_pd(&lval[c*32])));
                    _mm256_store_pd(&lval[c*32+4],_mm256_add_pd(tmp2,_mm256_load_pd(&lval[c*32+4])));
                    _mm256_store_pd(&lval[c*32+8],_mm256_add_pd(tmp3,_mm256_load_pd(&lval[c*32+8])));
                    _mm256_store_pd(&lval[c*32+12],_mm256_add_pd(tmp4,_mm256_load_pd(&lval[c*32+12])));
                    _mm256_store_pd(&lval[c*32+16],_mm256_add_pd(tmp5,_mm256_load_pd(&lval[c*32+16])));
                    _mm256_store_pd(&lval[c*32+20],_mm256_add_pd(tmp6,_mm256_load_pd(&lval[c*32+20])));
                    _mm256_store_pd(&lval[c*32+24],_mm256_add_pd(tmp7,_mm256_load_pd(&lval[c*32+24])));
                    _mm256_store_pd(&lval[c*32+28],_mm256_add_pd(tmp8,_mm256_load_pd(&lval[c*32+28])));
                } else if (spmvmOptions & GHOST_SPMV_AXPBY) {
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
                if (spmvmOptions & GHOST_SPMV_DOT) {
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
        }

        if (spmvmOptions & GHOST_SPMV_DOT) {
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
    if (spmvmOptions & GHOST_SPMV_DOT) {
        local_dot_product[0] = dots1;
        local_dot_product[1] = dots2;
        local_dot_product[2] = dots3;
    }

#else
    UNUSED(mat);
    UNUSED(res);
    UNUSED(invec);
    UNUSED(spmvmOptions);
    UNUSED(argp);
#endif
    return GHOST_SUCCESS;
}

ghost_error_t dd_SELL_kernel_AVX_32_rich_multivecx_rm(ghost_sparsemat_t *mat, ghost_densemat_t* res, ghost_densemat_t* invec, ghost_spmv_flags_t spmvmOptions,va_list argp)
{
#ifdef GHOST_HAVE_AVX
    ghost_idx_t j,c;
    ghost_nnz_t offs;
    double *mval = (double *)SELL(mat)->val;
    double *local_dot_product = NULL;
    __m256d dot1,dot2,dot3;
    double dots1 = 0, dots2 = 0, dots3 = 0;
    __m256d rhs;

    const int64_t mask1int[4] = {-1,0,0,0};
    __m256i mask1 = _mm256_loadu_si256((__m256i *)mask1int);
    const int64_t mask2int[4] = {-1,-1,0,0};
    __m256i mask2 = _mm256_loadu_si256((__m256i *)mask2int);
    const int64_t mask3int[4] = {-1,-1,-1,0};
    __m256i mask3 = _mm256_loadu_si256((__m256i *)mask3int);
    UNUSED(argp);
    
    double shift = 0., scale = 1., beta = 1.;

    GHOST_SPMV_PARSE_ARGS(spmvmOptions,argp,scale,beta,shift,local_dot_product,double);
    
    //__m256d shift, scale, beta;

    /*if (spmvmOptions & GHOST_SPMV_SCALE) {
      scale = _mm256_broadcast_sd(va_arg(argp,double *));
      }
      if (spmvmOptions & GHOST_SPMV_AXPBY) {
      beta = _mm256_broadcast_sd(va_arg(argp,double *));
      }
      if (spmvmOptions & GHOST_SPMV_SHIFT) {
      shift = _mm256_broadcast_sd(va_arg(argp,double *));
      }
      if (spmvmOptions & GHOST_SPMV_DOT) {
      local_dot_product = va_arg(argp,double *);
      }*/

#pragma omp parallel private(c,j,offs,rhs,dot1,dot2,dot3) reduction (+:dots1,dots2,dots3)
    {
        __m256d tmp11,tmp21,tmp31,tmp41,tmp51,tmp61,tmp71,tmp81;
        __m256d tmp12,tmp22,tmp32,tmp42,tmp52,tmp62,tmp72,tmp82;
        __m256d tmp13,tmp23,tmp33,tmp43,tmp53,tmp63,tmp73,tmp83;
        __m256d tmp14,tmp24,tmp34,tmp44,tmp54,tmp64,tmp74,tmp84;
        dot1 = _mm256_setzero_pd();
        dot2 = _mm256_setzero_pd();
        dot3 = _mm256_setzero_pd();
        ghost_idx_t remainder;
        ghost_idx_t donecols;

#pragma omp for schedule(runtime)
        for (c=0; c<mat->nrowsPadded>>5; c++) 
        { // loop over chunks
            remainder = invec->traits.ncols;
            donecols = 0;
            double *lval = (double *)res->val[c*32];

            while(remainder >= 4) { // this is done multiple times
                tmp11 = _mm256_setzero_pd(); // tmp = 0
                tmp21 = _mm256_setzero_pd(); // tmp = 0
                tmp31 = _mm256_setzero_pd(); // tmp = 0
                tmp41 = _mm256_setzero_pd(); // tmp = 0
                tmp51 = _mm256_setzero_pd(); // tmp = 0
                tmp61 = _mm256_setzero_pd(); // tmp = 0
                tmp71 = _mm256_setzero_pd(); // tmp = 0
                tmp81 = _mm256_setzero_pd(); // tmp = 0
                tmp12 = _mm256_setzero_pd(); // tmp = 0
                tmp22 = _mm256_setzero_pd(); // tmp = 0
                tmp32 = _mm256_setzero_pd(); // tmp = 0
                tmp42 = _mm256_setzero_pd(); // tmp = 0
                tmp52 = _mm256_setzero_pd(); // tmp = 0
                tmp62 = _mm256_setzero_pd(); // tmp = 0
                tmp72 = _mm256_setzero_pd(); // tmp = 0
                tmp82 = _mm256_setzero_pd(); // tmp = 0
                tmp13 = _mm256_setzero_pd(); // tmp = 0
                tmp23 = _mm256_setzero_pd(); // tmp = 0
                tmp33 = _mm256_setzero_pd(); // tmp = 0
                tmp43 = _mm256_setzero_pd(); // tmp = 0
                tmp53 = _mm256_setzero_pd(); // tmp = 0
                tmp63 = _mm256_setzero_pd(); // tmp = 0
                tmp73 = _mm256_setzero_pd(); // tmp = 0
                tmp83 = _mm256_setzero_pd(); // tmp = 0
                tmp14 = _mm256_setzero_pd(); // tmp = 0
                tmp24 = _mm256_setzero_pd(); // tmp = 0
                tmp34 = _mm256_setzero_pd(); // tmp = 0
                tmp44 = _mm256_setzero_pd(); // tmp = 0
                tmp54 = _mm256_setzero_pd(); // tmp = 0
                tmp64 = _mm256_setzero_pd(); // tmp = 0
                tmp74 = _mm256_setzero_pd(); // tmp = 0
                tmp84 = _mm256_setzero_pd(); // tmp = 0
                offs = SELL(mat)->chunkStart[c];

                //GHOST_INSTR_START(chunkloop)
                for (j=0; j<SELL(mat)->chunkLen[c]; j++) 
                { // loop inside chunk


                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp11 = _mm256_add_pd(tmp11,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp12 = _mm256_add_pd(tmp12,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp13 = _mm256_add_pd(tmp13,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp14 = _mm256_add_pd(tmp14,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp21 = _mm256_add_pd(tmp21,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp22 = _mm256_add_pd(tmp22,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp23 = _mm256_add_pd(tmp23,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp24 = _mm256_add_pd(tmp24,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp31 = _mm256_add_pd(tmp31,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp32 = _mm256_add_pd(tmp32,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp33 = _mm256_add_pd(tmp33,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp34 = _mm256_add_pd(tmp34,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp41 = _mm256_add_pd(tmp41,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp42 = _mm256_add_pd(tmp42,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp43 = _mm256_add_pd(tmp43,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp44 = _mm256_add_pd(tmp44,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp51 = _mm256_add_pd(tmp51,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp52 = _mm256_add_pd(tmp52,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp53 = _mm256_add_pd(tmp53,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp54 = _mm256_add_pd(tmp54,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp61 = _mm256_add_pd(tmp61,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp62 = _mm256_add_pd(tmp62,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp63 = _mm256_add_pd(tmp63,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp64 = _mm256_add_pd(tmp64,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp71 = _mm256_add_pd(tmp71,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp72 = _mm256_add_pd(tmp72,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp73 = _mm256_add_pd(tmp73,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp74 = _mm256_add_pd(tmp74,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp81 = _mm256_add_pd(tmp81,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp82 = _mm256_add_pd(tmp82,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp83 = _mm256_add_pd(tmp83,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                    tmp84 = _mm256_add_pd(tmp84,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                }

                if (spmvmOptions & GHOST_SPMV_AXPY) {
                    _mm256_store_pd(&lval[invec->traits.ncols*0+donecols],_mm256_add_pd(tmp11,_mm256_load_pd(&lval[invec->traits.ncols*0+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*1+donecols],_mm256_add_pd(tmp12,_mm256_load_pd(&lval[invec->traits.ncols*1+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*2+donecols],_mm256_add_pd(tmp13,_mm256_load_pd(&lval[invec->traits.ncols*2+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*3+donecols],_mm256_add_pd(tmp14,_mm256_load_pd(&lval[invec->traits.ncols*3+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*4+donecols],_mm256_add_pd(tmp21,_mm256_load_pd(&lval[invec->traits.ncols*4+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*5+donecols],_mm256_add_pd(tmp22,_mm256_load_pd(&lval[invec->traits.ncols*5+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*6+donecols],_mm256_add_pd(tmp23,_mm256_load_pd(&lval[invec->traits.ncols*6+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*7+donecols],_mm256_add_pd(tmp24,_mm256_load_pd(&lval[invec->traits.ncols*7+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*8+donecols],_mm256_add_pd(tmp31,_mm256_load_pd(&lval[invec->traits.ncols*8+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*9+donecols],_mm256_add_pd(tmp32,_mm256_load_pd(&lval[invec->traits.ncols*9+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*10+donecols],_mm256_add_pd(tmp33,_mm256_load_pd(&lval[invec->traits.ncols*10+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*11+donecols],_mm256_add_pd(tmp34,_mm256_load_pd(&lval[invec->traits.ncols*11+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*12+donecols],_mm256_add_pd(tmp41,_mm256_load_pd(&lval[invec->traits.ncols*12+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*13+donecols],_mm256_add_pd(tmp42,_mm256_load_pd(&lval[invec->traits.ncols*13+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*14+donecols],_mm256_add_pd(tmp43,_mm256_load_pd(&lval[invec->traits.ncols*14+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*15+donecols],_mm256_add_pd(tmp44,_mm256_load_pd(&lval[invec->traits.ncols*15+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*16+donecols],_mm256_add_pd(tmp51,_mm256_load_pd(&lval[invec->traits.ncols*16+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*17+donecols],_mm256_add_pd(tmp52,_mm256_load_pd(&lval[invec->traits.ncols*17+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*18+donecols],_mm256_add_pd(tmp53,_mm256_load_pd(&lval[invec->traits.ncols*18+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*19+donecols],_mm256_add_pd(tmp54,_mm256_load_pd(&lval[invec->traits.ncols*19+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*20+donecols],_mm256_add_pd(tmp61,_mm256_load_pd(&lval[invec->traits.ncols*20+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*21+donecols],_mm256_add_pd(tmp62,_mm256_load_pd(&lval[invec->traits.ncols*21+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*22+donecols],_mm256_add_pd(tmp63,_mm256_load_pd(&lval[invec->traits.ncols*22+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*23+donecols],_mm256_add_pd(tmp64,_mm256_load_pd(&lval[invec->traits.ncols*23+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*24+donecols],_mm256_add_pd(tmp71,_mm256_load_pd(&lval[invec->traits.ncols*24+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*25+donecols],_mm256_add_pd(tmp72,_mm256_load_pd(&lval[invec->traits.ncols*25+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*26+donecols],_mm256_add_pd(tmp73,_mm256_load_pd(&lval[invec->traits.ncols*26+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*27+donecols],_mm256_add_pd(tmp74,_mm256_load_pd(&lval[invec->traits.ncols*27+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*28+donecols],_mm256_add_pd(tmp81,_mm256_load_pd(&lval[invec->traits.ncols*28+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*29+donecols],_mm256_add_pd(tmp82,_mm256_load_pd(&lval[invec->traits.ncols*29+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*30+donecols],_mm256_add_pd(tmp83,_mm256_load_pd(&lval[invec->traits.ncols*30+donecols])));
                    _mm256_store_pd(&lval[invec->traits.ncols*31+donecols],_mm256_add_pd(tmp84,_mm256_load_pd(&lval[invec->traits.ncols*31+donecols])));
                } else if (spmvmOptions & GHOST_SPMV_AXPBY) {

                } else {
                    _mm256_store_pd(&lval[invec->traits.ncols*0+donecols],tmp11);
                    _mm256_store_pd(&lval[invec->traits.ncols*1+donecols],tmp12);
                    _mm256_store_pd(&lval[invec->traits.ncols*2+donecols],tmp13);
                    _mm256_store_pd(&lval[invec->traits.ncols*3+donecols],tmp14);
                    _mm256_store_pd(&lval[invec->traits.ncols*4+donecols],tmp21);
                    _mm256_store_pd(&lval[invec->traits.ncols*5+donecols],tmp22);
                    _mm256_store_pd(&lval[invec->traits.ncols*6+donecols],tmp23);
                    _mm256_store_pd(&lval[invec->traits.ncols*7+donecols],tmp24);
                    _mm256_store_pd(&lval[invec->traits.ncols*8+donecols],tmp31);
                    _mm256_store_pd(&lval[invec->traits.ncols*9+donecols],tmp32);
                    _mm256_store_pd(&lval[invec->traits.ncols*10+donecols],tmp33);
                    _mm256_store_pd(&lval[invec->traits.ncols*11+donecols],tmp34);
                    _mm256_store_pd(&lval[invec->traits.ncols*12+donecols],tmp41);
                    _mm256_store_pd(&lval[invec->traits.ncols*13+donecols],tmp42);
                    _mm256_store_pd(&lval[invec->traits.ncols*14+donecols],tmp43);
                    _mm256_store_pd(&lval[invec->traits.ncols*15+donecols],tmp44);
                    _mm256_store_pd(&lval[invec->traits.ncols*16+donecols],tmp51);
                    _mm256_store_pd(&lval[invec->traits.ncols*17+donecols],tmp52);
                    _mm256_store_pd(&lval[invec->traits.ncols*18+donecols],tmp53);
                    _mm256_store_pd(&lval[invec->traits.ncols*19+donecols],tmp54);
                    _mm256_store_pd(&lval[invec->traits.ncols*20+donecols],tmp61);
                    _mm256_store_pd(&lval[invec->traits.ncols*21+donecols],tmp62);
                    _mm256_store_pd(&lval[invec->traits.ncols*22+donecols],tmp63);
                    _mm256_store_pd(&lval[invec->traits.ncols*23+donecols],tmp64);
                    _mm256_store_pd(&lval[invec->traits.ncols*24+donecols],tmp71);
                    _mm256_store_pd(&lval[invec->traits.ncols*25+donecols],tmp72);
                    _mm256_store_pd(&lval[invec->traits.ncols*26+donecols],tmp73);
                    _mm256_store_pd(&lval[invec->traits.ncols*27+donecols],tmp74);
                    _mm256_store_pd(&lval[invec->traits.ncols*28+donecols],tmp81);
                    _mm256_store_pd(&lval[invec->traits.ncols*29+donecols],tmp82);
                    _mm256_store_pd(&lval[invec->traits.ncols*30+donecols],tmp83);
                    _mm256_store_pd(&lval[invec->traits.ncols*31+donecols],tmp84);



                }
                donecols += 4; 
                remainder -= 4;
            }
            while (remainder>=3) { // this should be done only once
                offs = SELL(mat)->chunkStart[c];
                tmp11 = _mm256_setzero_pd(); // tmp = 0
                tmp21 = _mm256_setzero_pd(); // tmp = 0
                tmp31 = _mm256_setzero_pd(); // tmp = 0
                tmp41 = _mm256_setzero_pd(); // tmp = 0
                tmp51 = _mm256_setzero_pd(); // tmp = 0
                tmp61 = _mm256_setzero_pd(); // tmp = 0
                tmp71 = _mm256_setzero_pd(); // tmp = 0
                tmp81 = _mm256_setzero_pd(); // tmp = 0
                tmp12 = _mm256_setzero_pd(); // tmp = 0
                tmp22 = _mm256_setzero_pd(); // tmp = 0
                tmp32 = _mm256_setzero_pd(); // tmp = 0
                tmp42 = _mm256_setzero_pd(); // tmp = 0
                tmp52 = _mm256_setzero_pd(); // tmp = 0
                tmp62 = _mm256_setzero_pd(); // tmp = 0
                tmp72 = _mm256_setzero_pd(); // tmp = 0
                tmp82 = _mm256_setzero_pd(); // tmp = 0
                tmp13 = _mm256_setzero_pd(); // tmp = 0
                tmp23 = _mm256_setzero_pd(); // tmp = 0
                tmp33 = _mm256_setzero_pd(); // tmp = 0
                tmp43 = _mm256_setzero_pd(); // tmp = 0
                tmp53 = _mm256_setzero_pd(); // tmp = 0
                tmp63 = _mm256_setzero_pd(); // tmp = 0
                tmp73 = _mm256_setzero_pd(); // tmp = 0
                tmp83 = _mm256_setzero_pd(); // tmp = 0
                tmp14 = _mm256_setzero_pd(); // tmp = 0
                tmp24 = _mm256_setzero_pd(); // tmp = 0
                tmp34 = _mm256_setzero_pd(); // tmp = 0
                tmp44 = _mm256_setzero_pd(); // tmp = 0
                tmp54 = _mm256_setzero_pd(); // tmp = 0
                tmp64 = _mm256_setzero_pd(); // tmp = 0
                tmp74 = _mm256_setzero_pd(); // tmp = 0
                tmp84 = _mm256_setzero_pd(); // tmp = 0
                for (j=0; j<SELL(mat)->chunkLen[c]; j++) 
                { // loop inside chunk
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp11 = _mm256_add_pd(tmp11,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp12 = _mm256_add_pd(tmp12,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp13 = _mm256_add_pd(tmp13,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp14 = _mm256_add_pd(tmp14,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp21 = _mm256_add_pd(tmp21,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp22 = _mm256_add_pd(tmp22,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp23 = _mm256_add_pd(tmp23,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp24 = _mm256_add_pd(tmp24,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp31 = _mm256_add_pd(tmp31,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp32 = _mm256_add_pd(tmp32,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp33 = _mm256_add_pd(tmp33,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp34 = _mm256_add_pd(tmp34,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp41 = _mm256_add_pd(tmp41,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp42 = _mm256_add_pd(tmp42,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp43 = _mm256_add_pd(tmp43,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp44 = _mm256_add_pd(tmp44,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp51 = _mm256_add_pd(tmp51,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp52 = _mm256_add_pd(tmp52,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp53 = _mm256_add_pd(tmp53,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp54 = _mm256_add_pd(tmp54,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp61 = _mm256_add_pd(tmp61,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp62 = _mm256_add_pd(tmp62,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp63 = _mm256_add_pd(tmp63,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp64 = _mm256_add_pd(tmp64,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp71 = _mm256_add_pd(tmp71,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp72 = _mm256_add_pd(tmp72,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp73 = _mm256_add_pd(tmp73,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp74 = _mm256_add_pd(tmp74,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp81 = _mm256_add_pd(tmp81,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp82 = _mm256_add_pd(tmp82,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp83 = _mm256_add_pd(tmp83,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask3); // maskload rhs
                    tmp84 = _mm256_add_pd(tmp84,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                }

                if (spmvmOptions & GHOST_SPMV_AXPY) {
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*0+donecols],mask3,_mm256_add_pd(tmp11,_mm256_maskload_pd(&lval[invec->traits.ncols*0+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*1+donecols],mask3,_mm256_add_pd(tmp12,_mm256_maskload_pd(&lval[invec->traits.ncols*1+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*2+donecols],mask3,_mm256_add_pd(tmp13,_mm256_maskload_pd(&lval[invec->traits.ncols*2+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*3+donecols],mask3,_mm256_add_pd(tmp14,_mm256_maskload_pd(&lval[invec->traits.ncols*3+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*4+donecols],mask3,_mm256_add_pd(tmp21,_mm256_maskload_pd(&lval[invec->traits.ncols*4+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*5+donecols],mask3,_mm256_add_pd(tmp22,_mm256_maskload_pd(&lval[invec->traits.ncols*5+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*6+donecols],mask3,_mm256_add_pd(tmp23,_mm256_maskload_pd(&lval[invec->traits.ncols*6+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*7+donecols],mask3,_mm256_add_pd(tmp24,_mm256_maskload_pd(&lval[invec->traits.ncols*7+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*8+donecols],mask3,_mm256_add_pd(tmp31,_mm256_maskload_pd(&lval[invec->traits.ncols*8+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*9+donecols],mask3,_mm256_add_pd(tmp32,_mm256_maskload_pd(&lval[invec->traits.ncols*9+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*10+donecols],mask3,_mm256_add_pd(tmp33,_mm256_maskload_pd(&lval[invec->traits.ncols*10+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*11+donecols],mask3,_mm256_add_pd(tmp34,_mm256_maskload_pd(&lval[invec->traits.ncols*11+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*12+donecols],mask3,_mm256_add_pd(tmp41,_mm256_maskload_pd(&lval[invec->traits.ncols*12+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*13+donecols],mask3,_mm256_add_pd(tmp42,_mm256_maskload_pd(&lval[invec->traits.ncols*13+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*14+donecols],mask3,_mm256_add_pd(tmp43,_mm256_maskload_pd(&lval[invec->traits.ncols*14+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*15+donecols],mask3,_mm256_add_pd(tmp44,_mm256_maskload_pd(&lval[invec->traits.ncols*15+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*16+donecols],mask3,_mm256_add_pd(tmp51,_mm256_maskload_pd(&lval[invec->traits.ncols*16+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*17+donecols],mask3,_mm256_add_pd(tmp52,_mm256_maskload_pd(&lval[invec->traits.ncols*17+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*18+donecols],mask3,_mm256_add_pd(tmp53,_mm256_maskload_pd(&lval[invec->traits.ncols*18+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*19+donecols],mask3,_mm256_add_pd(tmp54,_mm256_maskload_pd(&lval[invec->traits.ncols*19+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*20+donecols],mask3,_mm256_add_pd(tmp61,_mm256_maskload_pd(&lval[invec->traits.ncols*20+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*21+donecols],mask3,_mm256_add_pd(tmp62,_mm256_maskload_pd(&lval[invec->traits.ncols*21+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*22+donecols],mask3,_mm256_add_pd(tmp63,_mm256_maskload_pd(&lval[invec->traits.ncols*22+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*23+donecols],mask3,_mm256_add_pd(tmp64,_mm256_maskload_pd(&lval[invec->traits.ncols*23+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*24+donecols],mask3,_mm256_add_pd(tmp71,_mm256_maskload_pd(&lval[invec->traits.ncols*24+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*25+donecols],mask3,_mm256_add_pd(tmp72,_mm256_maskload_pd(&lval[invec->traits.ncols*25+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*26+donecols],mask3,_mm256_add_pd(tmp73,_mm256_maskload_pd(&lval[invec->traits.ncols*26+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*27+donecols],mask3,_mm256_add_pd(tmp74,_mm256_maskload_pd(&lval[invec->traits.ncols*27+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*28+donecols],mask3,_mm256_add_pd(tmp81,_mm256_maskload_pd(&lval[invec->traits.ncols*28+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*29+donecols],mask3,_mm256_add_pd(tmp82,_mm256_maskload_pd(&lval[invec->traits.ncols*29+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*30+donecols],mask3,_mm256_add_pd(tmp83,_mm256_maskload_pd(&lval[invec->traits.ncols*30+donecols],mask3)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*31+donecols],mask3,_mm256_add_pd(tmp84,_mm256_maskload_pd(&lval[invec->traits.ncols*31+donecols],mask3)));
                }
                remainder-=3;
                donecols+=3;
            }
            
            while (remainder>=2) { // this should be at most once
                offs = SELL(mat)->chunkStart[c];
                tmp11 = _mm256_setzero_pd(); // tmp = 0
                tmp21 = _mm256_setzero_pd(); // tmp = 0
                tmp31 = _mm256_setzero_pd(); // tmp = 0
                tmp41 = _mm256_setzero_pd(); // tmp = 0
                tmp51 = _mm256_setzero_pd(); // tmp = 0
                tmp61 = _mm256_setzero_pd(); // tmp = 0
                tmp71 = _mm256_setzero_pd(); // tmp = 0
                tmp81 = _mm256_setzero_pd(); // tmp = 0
                tmp12 = _mm256_setzero_pd(); // tmp = 0
                tmp22 = _mm256_setzero_pd(); // tmp = 0
                tmp32 = _mm256_setzero_pd(); // tmp = 0
                tmp42 = _mm256_setzero_pd(); // tmp = 0
                tmp52 = _mm256_setzero_pd(); // tmp = 0
                tmp62 = _mm256_setzero_pd(); // tmp = 0
                tmp72 = _mm256_setzero_pd(); // tmp = 0
                tmp82 = _mm256_setzero_pd(); // tmp = 0
                tmp13 = _mm256_setzero_pd(); // tmp = 0
                tmp23 = _mm256_setzero_pd(); // tmp = 0
                tmp33 = _mm256_setzero_pd(); // tmp = 0
                tmp43 = _mm256_setzero_pd(); // tmp = 0
                tmp53 = _mm256_setzero_pd(); // tmp = 0
                tmp63 = _mm256_setzero_pd(); // tmp = 0
                tmp73 = _mm256_setzero_pd(); // tmp = 0
                tmp83 = _mm256_setzero_pd(); // tmp = 0
                tmp14 = _mm256_setzero_pd(); // tmp = 0
                tmp24 = _mm256_setzero_pd(); // tmp = 0
                tmp34 = _mm256_setzero_pd(); // tmp = 0
                tmp44 = _mm256_setzero_pd(); // tmp = 0
                tmp54 = _mm256_setzero_pd(); // tmp = 0
                tmp64 = _mm256_setzero_pd(); // tmp = 0
                tmp74 = _mm256_setzero_pd(); // tmp = 0
                tmp84 = _mm256_setzero_pd(); // tmp = 0
                for (j=0; j<SELL(mat)->chunkLen[c]; j++) 
                { // loop inside chunk
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp11 = _mm256_add_pd(tmp11,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp12 = _mm256_add_pd(tmp12,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp13 = _mm256_add_pd(tmp13,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp14 = _mm256_add_pd(tmp14,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp21 = _mm256_add_pd(tmp21,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp22 = _mm256_add_pd(tmp22,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp23 = _mm256_add_pd(tmp23,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp24 = _mm256_add_pd(tmp24,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp31 = _mm256_add_pd(tmp31,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp32 = _mm256_add_pd(tmp32,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp33 = _mm256_add_pd(tmp33,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp34 = _mm256_add_pd(tmp34,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp41 = _mm256_add_pd(tmp41,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp42 = _mm256_add_pd(tmp42,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp43 = _mm256_add_pd(tmp43,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp44 = _mm256_add_pd(tmp44,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp51 = _mm256_add_pd(tmp51,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp52 = _mm256_add_pd(tmp52,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp53 = _mm256_add_pd(tmp53,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp54 = _mm256_add_pd(tmp54,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp61 = _mm256_add_pd(tmp61,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp62 = _mm256_add_pd(tmp62,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp63 = _mm256_add_pd(tmp63,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp64 = _mm256_add_pd(tmp64,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp71 = _mm256_add_pd(tmp71,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp72 = _mm256_add_pd(tmp72,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp73 = _mm256_add_pd(tmp73,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp74 = _mm256_add_pd(tmp74,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp81 = _mm256_add_pd(tmp81,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp82 = _mm256_add_pd(tmp82,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp83 = _mm256_add_pd(tmp83,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask2); // maskload rhs
                    tmp84 = _mm256_add_pd(tmp84,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                }

                if (spmvmOptions & GHOST_SPMV_AXPY) {
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*0+donecols],mask2,_mm256_add_pd(tmp11,_mm256_maskload_pd(&lval[invec->traits.ncols*0+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*1+donecols],mask2,_mm256_add_pd(tmp12,_mm256_maskload_pd(&lval[invec->traits.ncols*1+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*2+donecols],mask2,_mm256_add_pd(tmp13,_mm256_maskload_pd(&lval[invec->traits.ncols*2+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*3+donecols],mask2,_mm256_add_pd(tmp14,_mm256_maskload_pd(&lval[invec->traits.ncols*3+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*4+donecols],mask2,_mm256_add_pd(tmp21,_mm256_maskload_pd(&lval[invec->traits.ncols*4+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*5+donecols],mask2,_mm256_add_pd(tmp22,_mm256_maskload_pd(&lval[invec->traits.ncols*5+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*6+donecols],mask2,_mm256_add_pd(tmp23,_mm256_maskload_pd(&lval[invec->traits.ncols*6+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*7+donecols],mask2,_mm256_add_pd(tmp24,_mm256_maskload_pd(&lval[invec->traits.ncols*7+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*8+donecols],mask2,_mm256_add_pd(tmp31,_mm256_maskload_pd(&lval[invec->traits.ncols*8+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*9+donecols],mask2,_mm256_add_pd(tmp32,_mm256_maskload_pd(&lval[invec->traits.ncols*9+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*10+donecols],mask2,_mm256_add_pd(tmp33,_mm256_maskload_pd(&lval[invec->traits.ncols*10+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*11+donecols],mask2,_mm256_add_pd(tmp34,_mm256_maskload_pd(&lval[invec->traits.ncols*11+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*12+donecols],mask2,_mm256_add_pd(tmp41,_mm256_maskload_pd(&lval[invec->traits.ncols*12+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*13+donecols],mask2,_mm256_add_pd(tmp42,_mm256_maskload_pd(&lval[invec->traits.ncols*13+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*14+donecols],mask2,_mm256_add_pd(tmp43,_mm256_maskload_pd(&lval[invec->traits.ncols*14+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*15+donecols],mask2,_mm256_add_pd(tmp44,_mm256_maskload_pd(&lval[invec->traits.ncols*15+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*16+donecols],mask2,_mm256_add_pd(tmp51,_mm256_maskload_pd(&lval[invec->traits.ncols*16+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*17+donecols],mask2,_mm256_add_pd(tmp52,_mm256_maskload_pd(&lval[invec->traits.ncols*17+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*18+donecols],mask2,_mm256_add_pd(tmp53,_mm256_maskload_pd(&lval[invec->traits.ncols*18+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*19+donecols],mask2,_mm256_add_pd(tmp54,_mm256_maskload_pd(&lval[invec->traits.ncols*19+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*20+donecols],mask2,_mm256_add_pd(tmp61,_mm256_maskload_pd(&lval[invec->traits.ncols*20+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*21+donecols],mask2,_mm256_add_pd(tmp62,_mm256_maskload_pd(&lval[invec->traits.ncols*21+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*22+donecols],mask2,_mm256_add_pd(tmp63,_mm256_maskload_pd(&lval[invec->traits.ncols*22+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*23+donecols],mask2,_mm256_add_pd(tmp64,_mm256_maskload_pd(&lval[invec->traits.ncols*23+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*24+donecols],mask2,_mm256_add_pd(tmp71,_mm256_maskload_pd(&lval[invec->traits.ncols*24+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*25+donecols],mask2,_mm256_add_pd(tmp72,_mm256_maskload_pd(&lval[invec->traits.ncols*25+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*26+donecols],mask2,_mm256_add_pd(tmp73,_mm256_maskload_pd(&lval[invec->traits.ncols*26+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*27+donecols],mask2,_mm256_add_pd(tmp74,_mm256_maskload_pd(&lval[invec->traits.ncols*27+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*28+donecols],mask2,_mm256_add_pd(tmp81,_mm256_maskload_pd(&lval[invec->traits.ncols*28+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*29+donecols],mask2,_mm256_add_pd(tmp82,_mm256_maskload_pd(&lval[invec->traits.ncols*29+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*30+donecols],mask2,_mm256_add_pd(tmp83,_mm256_maskload_pd(&lval[invec->traits.ncols*30+donecols],mask2)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*31+donecols],mask2,_mm256_add_pd(tmp84,_mm256_maskload_pd(&lval[invec->traits.ncols*31+donecols],mask2)));
                }
                remainder-=2;
                donecols+=2;
            }


            while (remainder) { // this should be done only once
                offs = SELL(mat)->chunkStart[c];
                tmp11 = _mm256_setzero_pd(); // tmp = 0
                tmp21 = _mm256_setzero_pd(); // tmp = 0
                tmp31 = _mm256_setzero_pd(); // tmp = 0
                tmp41 = _mm256_setzero_pd(); // tmp = 0
                tmp51 = _mm256_setzero_pd(); // tmp = 0
                tmp61 = _mm256_setzero_pd(); // tmp = 0
                tmp71 = _mm256_setzero_pd(); // tmp = 0
                tmp81 = _mm256_setzero_pd(); // tmp = 0
                tmp12 = _mm256_setzero_pd(); // tmp = 0
                tmp22 = _mm256_setzero_pd(); // tmp = 0
                tmp32 = _mm256_setzero_pd(); // tmp = 0
                tmp42 = _mm256_setzero_pd(); // tmp = 0
                tmp52 = _mm256_setzero_pd(); // tmp = 0
                tmp62 = _mm256_setzero_pd(); // tmp = 0
                tmp72 = _mm256_setzero_pd(); // tmp = 0
                tmp82 = _mm256_setzero_pd(); // tmp = 0
                tmp13 = _mm256_setzero_pd(); // tmp = 0
                tmp23 = _mm256_setzero_pd(); // tmp = 0
                tmp33 = _mm256_setzero_pd(); // tmp = 0
                tmp43 = _mm256_setzero_pd(); // tmp = 0
                tmp53 = _mm256_setzero_pd(); // tmp = 0
                tmp63 = _mm256_setzero_pd(); // tmp = 0
                tmp73 = _mm256_setzero_pd(); // tmp = 0
                tmp83 = _mm256_setzero_pd(); // tmp = 0
                tmp14 = _mm256_setzero_pd(); // tmp = 0
                tmp24 = _mm256_setzero_pd(); // tmp = 0
                tmp34 = _mm256_setzero_pd(); // tmp = 0
                tmp44 = _mm256_setzero_pd(); // tmp = 0
                tmp54 = _mm256_setzero_pd(); // tmp = 0
                tmp64 = _mm256_setzero_pd(); // tmp = 0
                tmp74 = _mm256_setzero_pd(); // tmp = 0
                tmp84 = _mm256_setzero_pd(); // tmp = 0
                for (j=0; j<SELL(mat)->chunkLen[c]; j++) 
                { // loop inside chunk
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp11 = _mm256_add_pd(tmp11,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp12 = _mm256_add_pd(tmp12,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp13 = _mm256_add_pd(tmp13,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp14 = _mm256_add_pd(tmp14,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp21 = _mm256_add_pd(tmp21,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp22 = _mm256_add_pd(tmp22,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp23 = _mm256_add_pd(tmp23,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp24 = _mm256_add_pd(tmp24,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp31 = _mm256_add_pd(tmp31,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp32 = _mm256_add_pd(tmp32,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp33 = _mm256_add_pd(tmp33,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp34 = _mm256_add_pd(tmp34,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp41 = _mm256_add_pd(tmp41,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp42 = _mm256_add_pd(tmp42,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp43 = _mm256_add_pd(tmp43,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp44 = _mm256_add_pd(tmp44,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp51 = _mm256_add_pd(tmp51,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp52 = _mm256_add_pd(tmp52,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp53 = _mm256_add_pd(tmp53,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp54 = _mm256_add_pd(tmp54,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp61 = _mm256_add_pd(tmp61,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp62 = _mm256_add_pd(tmp62,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp63 = _mm256_add_pd(tmp63,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp64 = _mm256_add_pd(tmp64,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp71 = _mm256_add_pd(tmp71,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp72 = _mm256_add_pd(tmp72,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp73 = _mm256_add_pd(tmp73,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp74 = _mm256_add_pd(tmp74,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate

                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp81 = _mm256_add_pd(tmp81,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp82 = _mm256_add_pd(tmp82,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp83 = _mm256_add_pd(tmp83,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                    rhs  = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]],mask1); // maskload rhs
                    tmp84 = _mm256_add_pd(tmp84,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));           // accumulate
                }

                if (spmvmOptions & GHOST_SPMV_AXPY) {
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*0+donecols],mask1,_mm256_add_pd(tmp11,_mm256_maskload_pd(&lval[invec->traits.ncols*0+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*1+donecols],mask1,_mm256_add_pd(tmp12,_mm256_maskload_pd(&lval[invec->traits.ncols*1+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*2+donecols],mask1,_mm256_add_pd(tmp13,_mm256_maskload_pd(&lval[invec->traits.ncols*2+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*3+donecols],mask1,_mm256_add_pd(tmp14,_mm256_maskload_pd(&lval[invec->traits.ncols*3+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*4+donecols],mask1,_mm256_add_pd(tmp21,_mm256_maskload_pd(&lval[invec->traits.ncols*4+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*5+donecols],mask1,_mm256_add_pd(tmp22,_mm256_maskload_pd(&lval[invec->traits.ncols*5+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*6+donecols],mask1,_mm256_add_pd(tmp23,_mm256_maskload_pd(&lval[invec->traits.ncols*6+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*7+donecols],mask1,_mm256_add_pd(tmp24,_mm256_maskload_pd(&lval[invec->traits.ncols*7+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*8+donecols],mask1,_mm256_add_pd(tmp31,_mm256_maskload_pd(&lval[invec->traits.ncols*8+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*9+donecols],mask1,_mm256_add_pd(tmp32,_mm256_maskload_pd(&lval[invec->traits.ncols*9+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*10+donecols],mask1,_mm256_add_pd(tmp33,_mm256_maskload_pd(&lval[invec->traits.ncols*10+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*11+donecols],mask1,_mm256_add_pd(tmp34,_mm256_maskload_pd(&lval[invec->traits.ncols*11+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*12+donecols],mask1,_mm256_add_pd(tmp41,_mm256_maskload_pd(&lval[invec->traits.ncols*12+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*13+donecols],mask1,_mm256_add_pd(tmp42,_mm256_maskload_pd(&lval[invec->traits.ncols*13+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*14+donecols],mask1,_mm256_add_pd(tmp43,_mm256_maskload_pd(&lval[invec->traits.ncols*14+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*15+donecols],mask1,_mm256_add_pd(tmp44,_mm256_maskload_pd(&lval[invec->traits.ncols*15+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*16+donecols],mask1,_mm256_add_pd(tmp51,_mm256_maskload_pd(&lval[invec->traits.ncols*16+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*17+donecols],mask1,_mm256_add_pd(tmp52,_mm256_maskload_pd(&lval[invec->traits.ncols*17+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*18+donecols],mask1,_mm256_add_pd(tmp53,_mm256_maskload_pd(&lval[invec->traits.ncols*18+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*19+donecols],mask1,_mm256_add_pd(tmp54,_mm256_maskload_pd(&lval[invec->traits.ncols*19+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*20+donecols],mask1,_mm256_add_pd(tmp61,_mm256_maskload_pd(&lval[invec->traits.ncols*20+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*21+donecols],mask1,_mm256_add_pd(tmp62,_mm256_maskload_pd(&lval[invec->traits.ncols*21+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*22+donecols],mask1,_mm256_add_pd(tmp63,_mm256_maskload_pd(&lval[invec->traits.ncols*22+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*23+donecols],mask1,_mm256_add_pd(tmp64,_mm256_maskload_pd(&lval[invec->traits.ncols*23+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*24+donecols],mask1,_mm256_add_pd(tmp71,_mm256_maskload_pd(&lval[invec->traits.ncols*24+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*25+donecols],mask1,_mm256_add_pd(tmp72,_mm256_maskload_pd(&lval[invec->traits.ncols*25+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*26+donecols],mask1,_mm256_add_pd(tmp73,_mm256_maskload_pd(&lval[invec->traits.ncols*26+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*27+donecols],mask1,_mm256_add_pd(tmp74,_mm256_maskload_pd(&lval[invec->traits.ncols*27+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*28+donecols],mask1,_mm256_add_pd(tmp81,_mm256_maskload_pd(&lval[invec->traits.ncols*28+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*29+donecols],mask1,_mm256_add_pd(tmp82,_mm256_maskload_pd(&lval[invec->traits.ncols*29+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*30+donecols],mask1,_mm256_add_pd(tmp83,_mm256_maskload_pd(&lval[invec->traits.ncols*30+donecols],mask1)));
                    _mm256_maskstore_pd(&lval[invec->traits.ncols*31+donecols],mask1,_mm256_add_pd(tmp84,_mm256_maskload_pd(&lval[invec->traits.ncols*31+donecols],mask1)));
                }
                remainder--;
                donecols++;
            }
            //GHOST_INSTR_STOP(chunkloop)

            /*if (spmvmOptions & GHOST_SPMV_SHIFT) {
              tmp1 = _mm256_sub_pd(tmp1,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32])));
              tmp2 = _mm256_sub_pd(tmp2,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+4])));
              tmp3 = _mm256_sub_pd(tmp3,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+8])));
              tmp4 = _mm256_sub_pd(tmp4,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+12])));
              tmp5 = _mm256_sub_pd(tmp5,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+16])));
              tmp6 = _mm256_sub_pd(tmp6,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+20])));
              tmp7 = _mm256_sub_pd(tmp7,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+24])));
              tmp8 = _mm256_sub_pd(tmp8,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+28])));
              }
              if (spmvmOptions & GHOST_SPMV_SCALE) {
              tmp1 = _mm256_mul_pd(scale,tmp1);
              tmp2 = _mm256_mul_pd(scale,tmp2);
              tmp3 = _mm256_mul_pd(scale,tmp3);
              tmp4 = _mm256_mul_pd(scale,tmp4);
              tmp5 = _mm256_mul_pd(scale,tmp5);
              tmp6 = _mm256_mul_pd(scale,tmp6);
              tmp7 = _mm256_mul_pd(scale,tmp7);
              tmp8 = _mm256_mul_pd(scale,tmp8);
              }*/
            if (spmvmOptions & GHOST_SPMV_AXPY) {
                /*    _mm256_store_pd(&lval[0],_mm256_add_pd(tmp11,_mm256_load_pd(&lval[0])));
                      _mm256_store_pd(&lval[4*1],_mm256_add_pd(tmp12,_mm256_load_pd(&lval[4*1])));
                      _mm256_store_pd(&lval[4*2],_mm256_add_pd(tmp13,_mm256_load_pd(&lval[4*2])));
                      _mm256_store_pd(&lval[4*3],_mm256_add_pd(tmp14,_mm256_load_pd(&lval[4*3])));
                      _mm256_store_pd(&lval[4*4],_mm256_add_pd(tmp21,_mm256_load_pd(&lval[4*4])));
                      _mm256_store_pd(&lval[4*5],_mm256_add_pd(tmp22,_mm256_load_pd(&lval[4*5])));
                      _mm256_store_pd(&lval[4*6],_mm256_add_pd(tmp23,_mm256_load_pd(&lval[4*6])));
                      _mm256_store_pd(&lval[4*7],_mm256_add_pd(tmp24,_mm256_load_pd(&lval[4*7])));
                      _mm256_store_pd(&lval[4*8],_mm256_add_pd(tmp31,_mm256_load_pd(&lval[4*8])));
                      _mm256_store_pd(&lval[4*9],_mm256_add_pd(tmp32,_mm256_load_pd(&lval[4*9])));
                      _mm256_store_pd(&lval[4*10],_mm256_add_pd(tmp33,_mm256_load_pd(&lval[4*10])));
                      _mm256_store_pd(&lval[4*11],_mm256_add_pd(tmp34,_mm256_load_pd(&lval[4*11])));
                      _mm256_store_pd(&lval[4*12],_mm256_add_pd(tmp41,_mm256_load_pd(&lval[4*12])));
                      _mm256_store_pd(&lval[4*13],_mm256_add_pd(tmp42,_mm256_load_pd(&lval[4*13])));
                      _mm256_store_pd(&lval[4*14],_mm256_add_pd(tmp43,_mm256_load_pd(&lval[4*14])));
                      _mm256_store_pd(&lval[4*15],_mm256_add_pd(tmp44,_mm256_load_pd(&lval[4*15])));
                      _mm256_store_pd(&lval[4*16],_mm256_add_pd(tmp51,_mm256_load_pd(&lval[4*16])));
                      _mm256_store_pd(&lval[4*17],_mm256_add_pd(tmp52,_mm256_load_pd(&lval[4*17])));
                      _mm256_store_pd(&lval[4*18],_mm256_add_pd(tmp53,_mm256_load_pd(&lval[4*18])));
                      _mm256_store_pd(&lval[4*19],_mm256_add_pd(tmp54,_mm256_load_pd(&lval[4*19])));
                      _mm256_store_pd(&lval[4*20],_mm256_add_pd(tmp61,_mm256_load_pd(&lval[4*20])));
                      _mm256_store_pd(&lval[4*21],_mm256_add_pd(tmp62,_mm256_load_pd(&lval[4*21])));
                      _mm256_store_pd(&lval[4*22],_mm256_add_pd(tmp63,_mm256_load_pd(&lval[4*22])));
                      _mm256_store_pd(&lval[4*23],_mm256_add_pd(tmp64,_mm256_load_pd(&lval[4*23])));
                      _mm256_store_pd(&lval[4*24],_mm256_add_pd(tmp71,_mm256_load_pd(&lval[4*24])));
                      _mm256_store_pd(&lval[4*25],_mm256_add_pd(tmp72,_mm256_load_pd(&lval[4*25])));
                      _mm256_store_pd(&lval[4*26],_mm256_add_pd(tmp73,_mm256_load_pd(&lval[4*26])));
                      _mm256_store_pd(&lval[4*27],_mm256_add_pd(tmp74,_mm256_load_pd(&lval[4*27])));
                      _mm256_store_pd(&lval[4*28],_mm256_add_pd(tmp81,_mm256_load_pd(&lval[4*28])));
                      _mm256_store_pd(&lval[4*29],_mm256_add_pd(tmp82,_mm256_load_pd(&lval[4*29])));
                      _mm256_store_pd(&lval[4*30],_mm256_add_pd(tmp83,_mm256_load_pd(&lval[4*30])));
                      _mm256_store_pd(&lval[4*31],_mm256_add_pd(tmp84,_mm256_load_pd(&lval[4*31])));
                 */
                /* _mm256_store_pd(&lval[c*32+4],_mm256_add_pd(tmp2,_mm256_load_pd(&lval[c*32+4])));
                   _mm256_store_pd(&lval[c*32+8],_mm256_add_pd(tmp3,_mm256_load_pd(&lval[c*32+8])));
                   _mm256_store_pd(&lval[c*32+12],_mm256_add_pd(tmp4,_mm256_load_pd(&lval[c*32+12])));
                   _mm256_store_pd(&lval[c*32+16],_mm256_add_pd(tmp5,_mm256_load_pd(&lval[c*32+16])));
                   _mm256_store_pd(&lval[c*32+20],_mm256_add_pd(tmp6,_mm256_load_pd(&lval[c*32+20])));
                   _mm256_store_pd(&lval[c*32+24],_mm256_add_pd(tmp7,_mm256_load_pd(&lval[c*32+24])));
                   _mm256_store_pd(&lval[c*32+28],_mm256_add_pd(tmp8,_mm256_load_pd(&lval[c*32+28])));*/
            } /*else if (spmvmOptions & GHOST_SPMV_AXPBY) {
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
                if (spmvmOptions & GHOST_SPMV_DOT) {
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

                }*/
        }

        if (spmvmOptions & GHOST_SPMV_DOT) {
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
    if (spmvmOptions & GHOST_SPMV_DOT) {
        local_dot_product[0] = dots1;
        local_dot_product[1] = dots2;
        local_dot_product[2] = dots3;
    }

#else
    UNUSED(mat);
    UNUSED(res);
    UNUSED(invec);
    UNUSED(spmvmOptions);
    UNUSED(argp);
#endif
    return GHOST_SUCCESS;
}

ghost_error_t dd_SELL_kernel_AVX_32_rich_multivec4_rm(ghost_sparsemat_t *mat, ghost_densemat_t* res, ghost_densemat_t* invec, ghost_spmv_flags_t spmvmOptions,va_list argp)
{
#ifdef GHOST_HAVE_AVX
    ghost_idx_t j,c;
    ghost_nnz_t offs;
    double *lval = NULL;
    double *mval = (double *)SELL(mat)->val;
    double *local_dot_product = NULL;
    __m256d dot1,dot2,dot3;
    double dots1 = 0, dots2 = 0, dots3 = 0;
    __m256d rhs1;
    __m256d rhs2;
    __m256d rhs3;
    __m256d rhs4;
    UNUSED(argp);
    //__m256d shift, scale, beta;

    /*if (spmvmOptions & GHOST_SPMV_SCALE) {
      scale = _mm256_broadcast_sd(va_arg(argp,double *));
      }
      if (spmvmOptions & GHOST_SPMV_AXPBY) {
      beta = _mm256_broadcast_sd(va_arg(argp,double *));
      }
      if (spmvmOptions & GHOST_SPMV_SHIFT) {
      shift = _mm256_broadcast_sd(va_arg(argp,double *));
      }
      if (spmvmOptions & GHOST_SPMV_DOT) {
      local_dot_product = va_arg(argp,double *);
      }*/

#pragma omp parallel private(c,j,offs,rhs1,rhs2,rhs3,rhs4,dot1,dot2,dot3) reduction (+:dots1,dots2,dots3)
    {
        __m256d tmp11,tmp21,tmp31,tmp41,tmp51,tmp61,tmp71,tmp81;
        __m256d tmp12,tmp22,tmp32,tmp42,tmp52,tmp62,tmp72,tmp82;
        __m256d tmp13,tmp23,tmp33,tmp43,tmp53,tmp63,tmp73,tmp83;
        __m256d tmp14,tmp24,tmp34,tmp44,tmp54,tmp64,tmp74,tmp84;
        dot1 = _mm256_setzero_pd();
        dot2 = _mm256_setzero_pd();
        dot3 = _mm256_setzero_pd();
#pragma omp for schedule(runtime)
        for (c=0; c<mat->nrowsPadded>>5; c++) 
        { // loop over chunks
            lval = (double *)res->val[c*32];

            tmp11 = _mm256_setzero_pd(); // tmp = 0
            tmp21 = _mm256_setzero_pd(); // tmp = 0
            tmp31 = _mm256_setzero_pd(); // tmp = 0
            tmp41 = _mm256_setzero_pd(); // tmp = 0
            tmp51 = _mm256_setzero_pd(); // tmp = 0
            tmp61 = _mm256_setzero_pd(); // tmp = 0
            tmp71 = _mm256_setzero_pd(); // tmp = 0
            tmp81 = _mm256_setzero_pd(); // tmp = 0
            tmp12 = _mm256_setzero_pd(); // tmp = 0
            tmp22 = _mm256_setzero_pd(); // tmp = 0
            tmp32 = _mm256_setzero_pd(); // tmp = 0
            tmp42 = _mm256_setzero_pd(); // tmp = 0
            tmp52 = _mm256_setzero_pd(); // tmp = 0
            tmp62 = _mm256_setzero_pd(); // tmp = 0
            tmp72 = _mm256_setzero_pd(); // tmp = 0
            tmp82 = _mm256_setzero_pd(); // tmp = 0
            tmp13 = _mm256_setzero_pd(); // tmp = 0
            tmp23 = _mm256_setzero_pd(); // tmp = 0
            tmp33 = _mm256_setzero_pd(); // tmp = 0
            tmp43 = _mm256_setzero_pd(); // tmp = 0
            tmp53 = _mm256_setzero_pd(); // tmp = 0
            tmp63 = _mm256_setzero_pd(); // tmp = 0
            tmp73 = _mm256_setzero_pd(); // tmp = 0
            tmp83 = _mm256_setzero_pd(); // tmp = 0
            tmp14 = _mm256_setzero_pd(); // tmp = 0
            tmp24 = _mm256_setzero_pd(); // tmp = 0
            tmp34 = _mm256_setzero_pd(); // tmp = 0
            tmp44 = _mm256_setzero_pd(); // tmp = 0
            tmp54 = _mm256_setzero_pd(); // tmp = 0
            tmp64 = _mm256_setzero_pd(); // tmp = 0
            tmp74 = _mm256_setzero_pd(); // tmp = 0
            tmp84 = _mm256_setzero_pd(); // tmp = 0
            offs = SELL(mat)->chunkStart[c];

            //GHOST_INSTR_START(chunkloop)
            for (j=0; j<SELL(mat)->chunkLen[c]; j++) 
            { // loop inside chunk

                rhs1  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp11 = _mm256_add_pd(tmp11,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs1));           // accumulate
                rhs2  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp12 = _mm256_add_pd(tmp12,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs2));           // accumulate
                rhs3  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp13 = _mm256_add_pd(tmp13,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs3));           // accumulate
                rhs4  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp14 = _mm256_add_pd(tmp14,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs4));           // accumulate

                rhs1  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp21 = _mm256_add_pd(tmp21,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs1));           // accumulate
                rhs2  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp22 = _mm256_add_pd(tmp22,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs2));           // accumulate
                rhs3  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp23 = _mm256_add_pd(tmp23,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs3));           // accumulate
                rhs4  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp24 = _mm256_add_pd(tmp24,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs4));           // accumulate

                rhs1  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp31 = _mm256_add_pd(tmp31,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs1));           // accumulate
                rhs2  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp32 = _mm256_add_pd(tmp32,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs2));           // accumulate
                rhs3  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp33 = _mm256_add_pd(tmp33,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs3));           // accumulate
                rhs4  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp34 = _mm256_add_pd(tmp34,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs4));           // accumulate

                rhs1  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp41 = _mm256_add_pd(tmp41,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs1));           // accumulate
                rhs2  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp42 = _mm256_add_pd(tmp42,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs2));           // accumulate
                rhs3  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp43 = _mm256_add_pd(tmp43,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs3));           // accumulate
                rhs4  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp44 = _mm256_add_pd(tmp44,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs4));           // accumulate

                rhs1  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp51 = _mm256_add_pd(tmp51,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs1));           // accumulate
                rhs2  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp52 = _mm256_add_pd(tmp52,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs2));           // accumulate
                rhs3  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp53 = _mm256_add_pd(tmp53,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs3));           // accumulate
                rhs4  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp54 = _mm256_add_pd(tmp54,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs4));           // accumulate

                rhs1  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp61 = _mm256_add_pd(tmp61,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs1));           // accumulate
                rhs2  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp62 = _mm256_add_pd(tmp62,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs2));           // accumulate
                rhs3  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp63 = _mm256_add_pd(tmp63,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs3));           // accumulate
                rhs4  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp64 = _mm256_add_pd(tmp64,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs4));           // accumulate

                rhs1  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp71 = _mm256_add_pd(tmp71,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs1));           // accumulate
                rhs2  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp72 = _mm256_add_pd(tmp72,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs2));           // accumulate
                rhs3  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp73 = _mm256_add_pd(tmp73,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs3));           // accumulate
                rhs4  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp74 = _mm256_add_pd(tmp74,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs4));           // accumulate

                rhs1  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp81 = _mm256_add_pd(tmp81,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs1));           // accumulate
                rhs2  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp82 = _mm256_add_pd(tmp82,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs2));           // accumulate
                rhs3  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp83 = _mm256_add_pd(tmp83,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs3));           // accumulate
                rhs4  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp84 = _mm256_add_pd(tmp84,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs4));           // accumulate
            }
            //GHOST_INSTR_STOP(chunkloop)

            /*if (spmvmOptions & GHOST_SPMV_SHIFT) {
              tmp1 = _mm256_sub_pd(tmp1,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32])));
              tmp2 = _mm256_sub_pd(tmp2,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+4])));
              tmp3 = _mm256_sub_pd(tmp3,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+8])));
              tmp4 = _mm256_sub_pd(tmp4,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+12])));
              tmp5 = _mm256_sub_pd(tmp5,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+16])));
              tmp6 = _mm256_sub_pd(tmp6,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+20])));
              tmp7 = _mm256_sub_pd(tmp7,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+24])));
              tmp8 = _mm256_sub_pd(tmp8,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+28])));
              }
              if (spmvmOptions & GHOST_SPMV_SCALE) {
              tmp1 = _mm256_mul_pd(scale,tmp1);
              tmp2 = _mm256_mul_pd(scale,tmp2);
              tmp3 = _mm256_mul_pd(scale,tmp3);
              tmp4 = _mm256_mul_pd(scale,tmp4);
              tmp5 = _mm256_mul_pd(scale,tmp5);
              tmp6 = _mm256_mul_pd(scale,tmp6);
              tmp7 = _mm256_mul_pd(scale,tmp7);
              tmp8 = _mm256_mul_pd(scale,tmp8);
              }*/
            if (spmvmOptions & GHOST_SPMV_AXPY) {
                _mm256_store_pd(&lval[0],_mm256_add_pd(tmp11,_mm256_load_pd(&lval[0])));
                _mm256_store_pd(&lval[4*1],_mm256_add_pd(tmp12,_mm256_load_pd(&lval[4*1])));
                _mm256_store_pd(&lval[4*2],_mm256_add_pd(tmp13,_mm256_load_pd(&lval[4*2])));
                _mm256_store_pd(&lval[4*3],_mm256_add_pd(tmp14,_mm256_load_pd(&lval[4*3])));
                _mm256_store_pd(&lval[4*4],_mm256_add_pd(tmp21,_mm256_load_pd(&lval[4*4])));
                _mm256_store_pd(&lval[4*5],_mm256_add_pd(tmp22,_mm256_load_pd(&lval[4*5])));
                _mm256_store_pd(&lval[4*6],_mm256_add_pd(tmp23,_mm256_load_pd(&lval[4*6])));
                _mm256_store_pd(&lval[4*7],_mm256_add_pd(tmp24,_mm256_load_pd(&lval[4*7])));
                _mm256_store_pd(&lval[4*8],_mm256_add_pd(tmp31,_mm256_load_pd(&lval[4*8])));
                _mm256_store_pd(&lval[4*9],_mm256_add_pd(tmp32,_mm256_load_pd(&lval[4*9])));
                _mm256_store_pd(&lval[4*10],_mm256_add_pd(tmp33,_mm256_load_pd(&lval[4*10])));
                _mm256_store_pd(&lval[4*11],_mm256_add_pd(tmp34,_mm256_load_pd(&lval[4*11])));
                _mm256_store_pd(&lval[4*12],_mm256_add_pd(tmp41,_mm256_load_pd(&lval[4*12])));
                _mm256_store_pd(&lval[4*13],_mm256_add_pd(tmp42,_mm256_load_pd(&lval[4*13])));
                _mm256_store_pd(&lval[4*14],_mm256_add_pd(tmp43,_mm256_load_pd(&lval[4*14])));
                _mm256_store_pd(&lval[4*15],_mm256_add_pd(tmp44,_mm256_load_pd(&lval[4*15])));
                _mm256_store_pd(&lval[4*16],_mm256_add_pd(tmp51,_mm256_load_pd(&lval[4*16])));
                _mm256_store_pd(&lval[4*17],_mm256_add_pd(tmp52,_mm256_load_pd(&lval[4*17])));
                _mm256_store_pd(&lval[4*18],_mm256_add_pd(tmp53,_mm256_load_pd(&lval[4*18])));
                _mm256_store_pd(&lval[4*19],_mm256_add_pd(tmp54,_mm256_load_pd(&lval[4*19])));
                _mm256_store_pd(&lval[4*20],_mm256_add_pd(tmp61,_mm256_load_pd(&lval[4*20])));
                _mm256_store_pd(&lval[4*21],_mm256_add_pd(tmp62,_mm256_load_pd(&lval[4*21])));
                _mm256_store_pd(&lval[4*22],_mm256_add_pd(tmp63,_mm256_load_pd(&lval[4*22])));
                _mm256_store_pd(&lval[4*23],_mm256_add_pd(tmp64,_mm256_load_pd(&lval[4*23])));
                _mm256_store_pd(&lval[4*24],_mm256_add_pd(tmp71,_mm256_load_pd(&lval[4*24])));
                _mm256_store_pd(&lval[4*25],_mm256_add_pd(tmp72,_mm256_load_pd(&lval[4*25])));
                _mm256_store_pd(&lval[4*26],_mm256_add_pd(tmp73,_mm256_load_pd(&lval[4*26])));
                _mm256_store_pd(&lval[4*27],_mm256_add_pd(tmp74,_mm256_load_pd(&lval[4*27])));
                _mm256_store_pd(&lval[4*28],_mm256_add_pd(tmp81,_mm256_load_pd(&lval[4*28])));
                _mm256_store_pd(&lval[4*29],_mm256_add_pd(tmp82,_mm256_load_pd(&lval[4*29])));
                _mm256_store_pd(&lval[4*30],_mm256_add_pd(tmp83,_mm256_load_pd(&lval[4*30])));
                _mm256_store_pd(&lval[4*31],_mm256_add_pd(tmp84,_mm256_load_pd(&lval[4*31])));

                /* _mm256_store_pd(&lval[c*32+4],_mm256_add_pd(tmp2,_mm256_load_pd(&lval[c*32+4])));
                   _mm256_store_pd(&lval[c*32+8],_mm256_add_pd(tmp3,_mm256_load_pd(&lval[c*32+8])));
                   _mm256_store_pd(&lval[c*32+12],_mm256_add_pd(tmp4,_mm256_load_pd(&lval[c*32+12])));
                   _mm256_store_pd(&lval[c*32+16],_mm256_add_pd(tmp5,_mm256_load_pd(&lval[c*32+16])));
                   _mm256_store_pd(&lval[c*32+20],_mm256_add_pd(tmp6,_mm256_load_pd(&lval[c*32+20])));
                   _mm256_store_pd(&lval[c*32+24],_mm256_add_pd(tmp7,_mm256_load_pd(&lval[c*32+24])));
                   _mm256_store_pd(&lval[c*32+28],_mm256_add_pd(tmp8,_mm256_load_pd(&lval[c*32+28])));*/
            } /*else if (spmvmOptions & GHOST_SPMV_AXPBY) {
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
                if (spmvmOptions & GHOST_SPMV_DOT) {
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

                }*/
        }

        if (spmvmOptions & GHOST_SPMV_DOT) {
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
    if (spmvmOptions & GHOST_SPMV_DOT) {
        local_dot_product[0] = dots1;
        local_dot_product[1] = dots2;
        local_dot_product[2] = dots3;
    }

#else
    UNUSED(mat);
    UNUSED(res);
    UNUSED(invec);
    UNUSED(spmvmOptions);
    UNUSED(argp);
#endif
    return GHOST_SUCCESS;
}

ghost_error_t dd_SELL_kernel_AVX_32_rich_multivec_rm(ghost_sparsemat_t *mat, ghost_densemat_t* res, ghost_densemat_t* invec, ghost_spmv_flags_t spmvmOptions,va_list argp)
{
#ifdef GHOST_HAVE_AVX
    UNUSED(argp);
    const ghost_idx_t nsimdblocks = (invec->traits.ncols+3)/4;
    const ghost_idx_t nfullblocks = (invec->traits.ncols)/4;
    const int remainder = invec->traits.ncols%4;

    int64_t maskint[4] = {0,0,0,0};
    int slot;
    for (slot = 0; slot<remainder; slot++) {
        maskint[slot] = -1;
    }
    __m256i mask = _mm256_loadu_si256((__m256i *)maskint);

#pragma omp parallel
    {
        __m256d rval;
        const double **rhsval = (const double **)invec->val;
        const double *mval = ((double *)(SELL(mat)->val));
        ghost_idx_t j=0,c=0,k=0,x=0,v=0;
        ghost_nnz_t offs;
        __m256d tmp [32][nsimdblocks];
        __m256d matval;
#pragma omp for schedule(runtime)
        for (c=0; c<mat->nrowsPadded>>5; c++) 
        { // loop over chunks
            for (k=0; k<32; k++) {
                //                    for (i=0; i<4; i++) {
                for (j=0; j<nsimdblocks; j++) {
                    tmp[k][j] = _mm256_setzero_pd();
                }
                //                    }
            }

            offs = 0;//SELL(mat)->chunkStart[c];

            //GHOST_INSTR_START(chunkloop)
            for (j=0; j<SELL(mat)->chunkLen[c]; j++) 
            { // loop inside chunk
#pragma unroll_and_jam(4)
                for (k=0; k<32; k++) {
                    matval = _mm256_broadcast_sd(mval+k+offs+SELL(mat)->chunkStart[c]);
                    for (x=0,v=0; x<nfullblocks; x++,v+=4) {
                        rval = _mm256_load_pd(&rhsval[SELL(mat)->col[k+offs+SELL(mat)->chunkStart[c]]][v]);
                        tmp[k][x] = _mm256_add_pd(tmp[k][x],_mm256_mul_pd(matval,rval));           // accumulate
                    }
                    if (remainder) {
                        tmp[k][x] = _mm256_add_pd(tmp[k][x],_mm256_mul_pd(matval,_mm256_maskload_pd(&rhsval[SELL(mat)->col[k+offs+SELL(mat)->chunkStart[c]]][v],mask)));           // accumulate
                    }
                }
                offs+=32;
            }
            //GHOST_INSTR_STOP(chunkloop)

            if (spmvmOptions & GHOST_SPMV_AXPY) {
                for (k=0; k<32; k++) { 
                    for (x=0,v=0; x<invec->traits.ncols/4; x++,v+=4) {
                    _mm256_store_pd(((double *)(res->val[c*32+k]))+v,_mm256_add_pd(tmp[k][x],_mm256_load_pd(((double *)(res->val[c*32+k]))+v)));
                    }
                    if (remainder) {
                        _mm256_maskstore_pd(((double *)(res->val[c*32+k]))+v,mask,_mm256_add_pd(tmp[k][x],_mm256_maskload_pd(((double *)(res->val[c*32+k]))+v,mask)));
                    }
                }
            }
        }

    }

#else
    UNUSED(mat);
    UNUSED(res);
    UNUSED(invec);
    UNUSED(spmvmOptions);
    UNUSED(argp);
#endif
    return GHOST_SUCCESS;
}

ghost_error_t dd_SELL_kernel_AVX_32(ghost_sparsemat_t *mat, ghost_densemat_t* res, ghost_densemat_t* invec, ghost_spmv_flags_t spmvmOptions,va_list argp)
{
    UNUSED(argp);
#ifdef GHOST_HAVE_AVX
    ghost_idx_t c,j;
    ghost_nnz_t offs;
    double *mval = (double *)SELL(mat)->val;
    double *lval = (double *)res->val[0];
    double *rval = (double *)invec->val[0];
    __m256d tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8;
    __m256d val;
    __m256d rhs;
    __m128d rhstmp;

#pragma omp parallel for schedule(runtime) private(j,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,val,offs,rhs,rhstmp)
    for (c=0; c<mat->nrowsPadded>>5; c++) 
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
        if (spmvmOptions & GHOST_SPMV_AXPY) {
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
    return GHOST_SUCCESS;
}

ghost_error_t dd_SELL_kernel_MIC_16(ghost_sparsemat_t *mat, ghost_densemat_t* res, ghost_densemat_t* invec, ghost_spmv_flags_t spmvmOptions,va_list argp)
{
    UNUSED(argp);
#ifdef GHOST_HAVE_MIC
    ghost_idx_t c,j;
    ghost_nnz_t offs;
    double *mval = (double *)SELL(mat)->val;
    double *lval = (double *)res->val[0];
    double *rval = (double *)invec->val[0];
    __m512d tmp1;
    __m512d tmp2;
    __m512d val;
    __m512d rhs;
    __m512i idx;

#pragma omp parallel for schedule(runtime) private(j,tmp1,tmp2,idx,val,rhs,offs)
    for (c=0; c<mat->nrowsPadded>>4; c++) 
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
        if (spmvmOptions & GHOST_SPMV_AXPY) {
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
    return GHOST_SUCCESS;
}

ghost_error_t dd_SELL_kernel_MIC_32(ghost_sparsemat_t *mat, ghost_densemat_t* res, ghost_densemat_t* invec, ghost_spmv_flags_t spmvmOptions,va_list argp)
{
    UNUSED(argp);
#ifdef GHOST_HAVE_MIC
    ghost_idx_t c,j;
    ghost_nnz_t offs;
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
    for (c=0; c<mat->nrowsPadded>>5; c++) 
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
        if (spmvmOptions & GHOST_SPMV_AXPY) {
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
    return GHOST_SUCCESS;
}
