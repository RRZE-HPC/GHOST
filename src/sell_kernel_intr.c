#include "ghost/config.h"
#undef GHOST_HAVE_MPI
#include "ghost/types.h"
#include "ghost/sell.h"
#include "ghost/util.h"
#include "ghost/instr.h"
#include "ghost/machine.h"
#include "ghost/omp.h"
#include <immintrin.h>

ghost_error_t dd_SELL_kernel_SSE (ghost_sparsemat_t *mat, ghost_densemat_t * lhs, ghost_densemat_t * invec, ghost_spmv_flags_t options,va_list argp)
{
#if 0/*GHOST_HAVE_SSE*/
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
    UNUSED(argp);
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
    ghost_idx_t j,c,v,i;
    ghost_nnz_t offs;
    double *lval = NULL, *rval = NULL;
    double *mval = (double *)SELL(mat)->val;
    double *local_dot_product = NULL;
    double *partsums = NULL;
    __m256d tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8;
    __m256d val;
    __m256d rhs;
    __m128d rhstmp;
    
    double sscale = 1., sbeta = 1.;
    double *sshift = NULL;
    __m256d shift, scale, beta;

    GHOST_SPMV_PARSE_ARGS(spmvmOptions,argp,sscale,sbeta,sshift,local_dot_product,double);
    scale = _mm256_broadcast_sd(&sscale);
    beta = _mm256_broadcast_sd(&sbeta);

    int nthreads = 1;
    unsigned clsize;
    ghost_machine_cacheline_size(&clsize);
    unsigned padding = clsize/sizeof(double);
    if (spmvmOptions & GHOST_SPMV_DOT) {

#pragma omp parallel 
        {
#pragma omp single
            nthreads = ghost_omp_nthread();
        }

        GHOST_CALL_RETURN(ghost_malloc((void **)&partsums,(3*invec->traits.ncols+padding)*nthreads*sizeof(double))); 
        for (i=0; i<(3*invec->traits.ncols+padding)*nthreads; i++) {
            partsums[i] = 0.;
        }
    }

#pragma omp parallel private(v,c,j,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,val,offs,rhs,rhstmp) shared(partsums)
    {
        int tid = ghost_omp_threadnum();
        __m256d dot1[invec->traits.ncols],dot2[invec->traits.ncols],dot3[invec->traits.ncols];
        for (v=0; v<invec->traits.ncols; v++) {
            dot1[v] = _mm256_setzero_pd();
            dot2[v] = _mm256_setzero_pd();
            dot3[v] = _mm256_setzero_pd();
        }
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

                if (spmvmOptions & (GHOST_SPMV_SHIFT | GHOST_SPMV_VSHIFT)) {
                    if (spmvmOptions & GHOST_SPMV_SHIFT) {
                        shift = _mm256_broadcast_sd(&sshift[0]);
                    } else {
                        shift = _mm256_broadcast_sd(&sshift[v]);
                    }
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
                    dot1[v] = _mm256_add_pd(dot1[v],_mm256_mul_pd(_mm256_load_pd(&lval[c*32]),_mm256_load_pd(&lval[c*32])));
                    dot1[v] = _mm256_add_pd(dot1[v],_mm256_mul_pd(_mm256_load_pd(&lval[c*32+4]),_mm256_load_pd(&lval[c*32+4])));
                    dot1[v] = _mm256_add_pd(dot1[v],_mm256_mul_pd(_mm256_load_pd(&lval[c*32+8]),_mm256_load_pd(&lval[c*32+8])));
                    dot1[v] = _mm256_add_pd(dot1[v],_mm256_mul_pd(_mm256_load_pd(&lval[c*32+12]),_mm256_load_pd(&lval[c*32+12])));
                    dot1[v] = _mm256_add_pd(dot1[v],_mm256_mul_pd(_mm256_load_pd(&lval[c*32+16]),_mm256_load_pd(&lval[c*32+16])));
                    dot1[v] = _mm256_add_pd(dot1[v],_mm256_mul_pd(_mm256_load_pd(&lval[c*32+20]),_mm256_load_pd(&lval[c*32+20])));
                    dot1[v] = _mm256_add_pd(dot1[v],_mm256_mul_pd(_mm256_load_pd(&lval[c*32+24]),_mm256_load_pd(&lval[c*32+24])));
                    dot1[v] = _mm256_add_pd(dot1[v],_mm256_mul_pd(_mm256_load_pd(&lval[c*32+28]),_mm256_load_pd(&lval[c*32+28])));

                    dot2[v] = _mm256_add_pd(dot2[v],_mm256_mul_pd(_mm256_load_pd(&rval[c*32]),_mm256_load_pd(&lval[c*32])));
                    dot2[v] = _mm256_add_pd(dot2[v],_mm256_mul_pd(_mm256_load_pd(&rval[c*32+4]),_mm256_load_pd(&lval[c*32+4])));
                    dot2[v] = _mm256_add_pd(dot2[v],_mm256_mul_pd(_mm256_load_pd(&rval[c*32+8]),_mm256_load_pd(&lval[c*32+8])));
                    dot2[v] = _mm256_add_pd(dot2[v],_mm256_mul_pd(_mm256_load_pd(&rval[c*32+12]),_mm256_load_pd(&lval[c*32+12])));
                    dot2[v] = _mm256_add_pd(dot2[v],_mm256_mul_pd(_mm256_load_pd(&rval[c*32+16]),_mm256_load_pd(&lval[c*32+16])));
                    dot2[v] = _mm256_add_pd(dot2[v],_mm256_mul_pd(_mm256_load_pd(&rval[c*32+20]),_mm256_load_pd(&lval[c*32+20])));
                    dot2[v] = _mm256_add_pd(dot2[v],_mm256_mul_pd(_mm256_load_pd(&rval[c*32+24]),_mm256_load_pd(&lval[c*32+24])));
                    dot2[v] = _mm256_add_pd(dot2[v],_mm256_mul_pd(_mm256_load_pd(&rval[c*32+28]),_mm256_load_pd(&lval[c*32+28])));

                    dot3[v] = _mm256_add_pd(dot3[v],_mm256_mul_pd(_mm256_load_pd(&rval[c*32]),_mm256_load_pd(&rval[c*32])));
                    dot3[v] = _mm256_add_pd(dot3[v],_mm256_mul_pd(_mm256_load_pd(&rval[c*32+4]),_mm256_load_pd(&rval[c*32+4])));
                    dot3[v] = _mm256_add_pd(dot3[v],_mm256_mul_pd(_mm256_load_pd(&rval[c*32+8]),_mm256_load_pd(&rval[c*32+8])));
                    dot3[v] = _mm256_add_pd(dot3[v],_mm256_mul_pd(_mm256_load_pd(&rval[c*32+12]),_mm256_load_pd(&rval[c*32+12])));
                    dot3[v] = _mm256_add_pd(dot3[v],_mm256_mul_pd(_mm256_load_pd(&rval[c*32+16]),_mm256_load_pd(&rval[c*32+16])));
                    dot3[v] = _mm256_add_pd(dot3[v],_mm256_mul_pd(_mm256_load_pd(&rval[c*32+20]),_mm256_load_pd(&rval[c*32+20])));
                    dot3[v] = _mm256_add_pd(dot3[v],_mm256_mul_pd(_mm256_load_pd(&rval[c*32+24]),_mm256_load_pd(&rval[c*32+24])));
                    dot3[v] = _mm256_add_pd(dot3[v],_mm256_mul_pd(_mm256_load_pd(&rval[c*32+28]),_mm256_load_pd(&rval[c*32+28])));

                }
            }
        }

        if (spmvmOptions & GHOST_SPMV_DOT) {
            __m256d sum12;
            __m128d sum12high;
            __m128d res12;
            for (v=0; v<invec->traits.ncols; v++) {

                sum12 = _mm256_hadd_pd(dot1[v],dot2[v]);
                sum12high = _mm256_extractf128_pd(sum12,1);
                res12 = _mm_add_pd(sum12high, _mm256_castpd256_pd128(sum12));

                partsums[((padding+3*invec->traits.ncols)*tid)+3*v+0] += ((double *)&res12)[0];
                partsums[((padding+3*invec->traits.ncols)*tid)+3*v+1] += ((double *)&res12)[1];

                sum12 = _mm256_hadd_pd(dot3[v],dot3[v]);
                sum12high = _mm256_extractf128_pd(sum12,1);
                res12 = _mm_add_pd(sum12high, _mm256_castpd256_pd128(sum12));
                partsums[((padding+3*invec->traits.ncols)*tid)+3*v+2] += ((double *)&res12)[0];
            }
        }
    }
    if (spmvmOptions & GHOST_SPMV_DOT) {
        for (v=0; v<invec->traits.ncols; v++) {
            for (i=0; i<nthreads; i++) {
                local_dot_product[v                       ] += partsums[(padding+3*invec->traits.ncols)*i + 3*v + 0];
                local_dot_product[v  +   invec->traits.ncols] += partsums[(padding+3*invec->traits.ncols)*i + 3*v + 1];
                local_dot_product[v  + 2*invec->traits.ncols] += partsums[(padding+3*invec->traits.ncols)*i + 3*v + 2];
            }
        }
        free(partsums);
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
    ghost_idx_t j,c,col;
    ghost_nnz_t offs;
    int maskidx;
    double *mval = (double *)SELL(mat)->val;
    double *local_dot_product = NULL;
    double *partsums = NULL;
    __m256d rhs;
    int nthreads = 1, i;
    
    unsigned clsize;
    ghost_machine_cacheline_size(&clsize);
    unsigned padding = clsize/sizeof(double);

    const int64_t mask1int[4] = {-1,0,0,0};
    const int64_t mask2int[4] = {-1,-1,0,0};
    const int64_t mask3int[4] = {-1,-1,-1,0};

    __m256i mask[3] = {_mm256_loadu_si256((__m256i *)mask3int), _mm256_loadu_si256((__m256i *)mask2int), _mm256_loadu_si256((__m256i *)mask1int)};
    UNUSED(argp);
    
    double sscale = 1., sbeta = 1.;
    double *sshift = NULL;
    __m256d shift, scale, beta;

    GHOST_SPMV_PARSE_ARGS(spmvmOptions,argp,sscale,sbeta,sshift,local_dot_product,double);
    scale = _mm256_broadcast_sd(&sscale);
    beta = _mm256_broadcast_sd(&sbeta);
    
    if (spmvmOptions & GHOST_SPMV_DOT) {

#pragma omp parallel 
        {
#pragma omp single
            nthreads = ghost_omp_nthread();
        }

        GHOST_CALL_RETURN(ghost_malloc((void **)&partsums,(3*invec->traits.ncols+padding)*nthreads*sizeof(double))); 
        for (col=0; col<(3*invec->traits.ncols+padding)*nthreads; col++) {
            partsums[col] = 0.;
        }
    }

#pragma omp parallel private(c,j,offs,rhs,col) shared (partsums)
    {
        int tid = ghost_omp_threadnum();
        #GHOST_UNROLL#__m256d tmp@;#32

        ghost_idx_t remainder;
        ghost_idx_t donecols;

#pragma omp for schedule(runtime)
        for (c=0; c<mat->nrowsPadded>>5; c++) 
        { // loop over chunks
            remainder = invec->traits.ncols;
            donecols = 0;
            double *lval = (double *)res->val[c*32];
            double *rval = (double *)invec->val[c*32];

            while(remainder >= 4) { // this is done multiple times
                #GHOST_UNROLL#tmp@ = _mm256_setzero_pd();#32
                offs = SELL(mat)->chunkStart[c];

                for (j=0; j<SELL(mat)->chunkLen[c]; j++) { // loop inside chunk
                    #GHOST_UNROLL#rhs = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]+donecols);tmp@ = _mm256_add_pd(tmp@,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));#32
                }
              
                if (spmvmOptions & (GHOST_SPMV_SHIFT | GHOST_SPMV_VSHIFT)) {
                    if (spmvmOptions & GHOST_SPMV_SHIFT) {
                        shift = _mm256_broadcast_sd(&sshift[0]);
                    } else {
                        shift = _mm256_load_pd(&sshift[donecols]);
                    }
                    #GHOST_UNROLL#tmp@ = _mm256_sub_pd(tmp@,_mm256_mul_pd(shift,_mm256_load_pd((double *)invec->val[c*32+@]+donecols)));#32
                }
                if (spmvmOptions & GHOST_SPMV_SCALE) {
                    #GHOST_UNROLL#tmp@ = _mm256_mul_pd(scale,tmp@);#32
                }
                if (spmvmOptions & GHOST_SPMV_AXPY) {
                    #GHOST_UNROLL#_mm256_store_pd(&lval[invec->traits.ncols*@+donecols],_mm256_add_pd(tmp@,_mm256_load_pd(&lval[invec->traits.ncols*@+donecols])));#32
                } else if (spmvmOptions & GHOST_SPMV_AXPBY) {
                    #GHOST_UNROLL#_mm256_store_pd(&lval[invec->traits.ncols*@+donecols],_mm256_add_pd(tmp@,_mm256_mul_pd(_mm256_load_pd(&lval[invec->traits.ncols*@+donecols]),beta)));#32
                } else {
                    #GHOST_UNROLL#_mm256_store_pd(&lval[invec->traits.ncols*@+donecols],tmp@);#32
                }
                if (spmvmOptions & GHOST_SPMV_DOT) {
                    for (col = donecols; col<donecols+4; col++) {
                        #GHOST_UNROLL#partsums[((padding+3*invec->traits.ncols)*tid)+3*col+0] += lval[col+@*invec->traits.ncols]*lval[col+@*invec->traits.ncols];#32
                        #GHOST_UNROLL#partsums[((padding+3*invec->traits.ncols)*tid)+3*col+1] += lval[col+@*invec->traits.ncols]*rval[col+@*invec->traits.ncols];#32
                        #GHOST_UNROLL#partsums[((padding+3*invec->traits.ncols)*tid)+3*col+2] += rval[col+@*invec->traits.ncols]*rval[col+@*invec->traits.ncols];#32
                    }
                }
                
                donecols += 4; 
                remainder -= 4;
            }
            for (maskidx = 0; maskidx < 3; maskidx++) {
                int maskwidth = 4-maskidx-1;
                while (remainder>=maskwidth) {
                    #GHOST_UNROLL#tmp@ = _mm256_setzero_pd();#32
                    offs = SELL(mat)->chunkStart[c];
                    
                    for (j=0; j<SELL(mat)->chunkLen[c]; j++) { // loop inside chunk
                        #GHOST_UNROLL#rhs = _mm256_maskload_pd((double *)invec->val[SELL(mat)->col[offs]]+donecols,mask[maskidx]);tmp@ = _mm256_add_pd(tmp@,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));#32
                    }
                    
                    if (spmvmOptions & (GHOST_SPMV_SHIFT | GHOST_SPMV_VSHIFT)) {
                        if (spmvmOptions & GHOST_SPMV_SHIFT) {
                            shift = _mm256_broadcast_sd(&sshift[0]);
                        } else {
                            shift = _mm256_maskload_pd(&sshift[donecols],mask[maskidx]);
                        }
                        #GHOST_UNROLL#tmp@ = _mm256_sub_pd(tmp@,_mm256_mul_pd(shift,_mm256_maskload_pd((double *)invec->val[c*32+@]+donecols,mask[maskidx])));#32
                    }
                    if (spmvmOptions & GHOST_SPMV_SHIFT) {
                    }
                    if (spmvmOptions & GHOST_SPMV_SCALE) {
                        #GHOST_UNROLL#tmp@ = _mm256_mul_pd(scale,tmp@);#32
                    }
                    if (spmvmOptions & GHOST_SPMV_AXPY) {
                        #GHOST_UNROLL#_mm256_maskstore_pd(&lval[invec->traits.ncols*@+donecols],mask[maskidx],_mm256_add_pd(tmp@,_mm256_load_pd(&lval[invec->traits.ncols*@+donecols])));#32
                    } else if (spmvmOptions & GHOST_SPMV_AXPBY) {
                        #GHOST_UNROLL#_mm256_maskstore_pd(&lval[invec->traits.ncols*@+donecols],mask[maskidx],_mm256_add_pd(tmp@,_mm256_mul_pd(_mm256_load_pd(&lval[invec->traits.ncols*@+donecols]),beta)));#32
                    } else {
                        #GHOST_UNROLL#_mm256_maskstore_pd(&lval[invec->traits.ncols*@+donecols],mask[maskidx],tmp@);#32
                    }
                    if (spmvmOptions & GHOST_SPMV_DOT) {
                        for (col = donecols; col<donecols+maskwidth; col++) {
                            #GHOST_UNROLL#partsums[((padding+3*invec->traits.ncols)*tid)+3*col+0] += lval[col+@*invec->traits.ncols]*lval[col+@*invec->traits.ncols];#32
                            #GHOST_UNROLL#partsums[((padding+3*invec->traits.ncols)*tid)+3*col+1] += lval[col+@*invec->traits.ncols]*rval[col+@*invec->traits.ncols];#32
                            #GHOST_UNROLL#partsums[((padding+3*invec->traits.ncols)*tid)+3*col+2] += rval[col+@*invec->traits.ncols]*rval[col+@*invec->traits.ncols];#32
                        }
                    }
                    remainder -= maskwidth;
                    donecols += maskwidth;
                }
            }
        }
    }
    if (spmvmOptions & GHOST_SPMV_DOT) {
        for (col=0; col<invec->traits.ncols; col++) {
            for (i=0; i<nthreads; i++) {
                local_dot_product[col                         ] += partsums[(padding+3*invec->traits.ncols)*i + 3*col + 0];
                local_dot_product[col  +   invec->traits.ncols] += partsums[(padding+3*invec->traits.ncols)*i + 3*col + 1];
                local_dot_product[col  + 2*invec->traits.ncols] += partsums[(padding+3*invec->traits.ncols)*i + 3*col + 2];
            }
        }
        free(partsums);
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
        __m256d tmp0,tmp4,tmp8,tmp12,tmp16,tmp20,tmp24,tmp28;
        __m256d tmp1,tmp5,tmp9,tmp13,tmp17,tmp21,tmp25,tmp29;
        __m256d tmp2,tmp6,tmp10,tmp14,tmp18,tmp22,tmp26,tmp30;
        __m256d tmp3,tmp7,tmp11,tmp15,tmp19,tmp23,tmp27,tmp31;
        dot1 = _mm256_setzero_pd();
        dot2 = _mm256_setzero_pd();
        dot3 = _mm256_setzero_pd();
#pragma omp for schedule(runtime)
        for (c=0; c<mat->nrowsPadded>>5; c++) 
        { // loop over chunks
            lval = (double *)res->val[c*32];

            tmp0 = _mm256_setzero_pd(); // tmp = 0
            tmp4 = _mm256_setzero_pd(); // tmp = 0
            tmp8 = _mm256_setzero_pd(); // tmp = 0
            tmp12 = _mm256_setzero_pd(); // tmp = 0
            tmp16 = _mm256_setzero_pd(); // tmp = 0
            tmp20 = _mm256_setzero_pd(); // tmp = 0
            tmp24 = _mm256_setzero_pd(); // tmp = 0
            tmp28 = _mm256_setzero_pd(); // tmp = 0
            tmp1 = _mm256_setzero_pd(); // tmp = 0
            tmp5 = _mm256_setzero_pd(); // tmp = 0
            tmp9 = _mm256_setzero_pd(); // tmp = 0
            tmp13 = _mm256_setzero_pd(); // tmp = 0
            tmp17 = _mm256_setzero_pd(); // tmp = 0
            tmp21 = _mm256_setzero_pd(); // tmp = 0
            tmp25 = _mm256_setzero_pd(); // tmp = 0
            tmp29 = _mm256_setzero_pd(); // tmp = 0
            tmp2 = _mm256_setzero_pd(); // tmp = 0
            tmp6 = _mm256_setzero_pd(); // tmp = 0
            tmp10 = _mm256_setzero_pd(); // tmp = 0
            tmp14 = _mm256_setzero_pd(); // tmp = 0
            tmp18 = _mm256_setzero_pd(); // tmp = 0
            tmp22 = _mm256_setzero_pd(); // tmp = 0
            tmp26 = _mm256_setzero_pd(); // tmp = 0
            tmp30 = _mm256_setzero_pd(); // tmp = 0
            tmp3 = _mm256_setzero_pd(); // tmp = 0
            tmp7 = _mm256_setzero_pd(); // tmp = 0
            tmp11 = _mm256_setzero_pd(); // tmp = 0
            tmp15 = _mm256_setzero_pd(); // tmp = 0
            tmp19 = _mm256_setzero_pd(); // tmp = 0
            tmp23 = _mm256_setzero_pd(); // tmp = 0
            tmp27 = _mm256_setzero_pd(); // tmp = 0
            tmp31 = _mm256_setzero_pd(); // tmp = 0
            offs = SELL(mat)->chunkStart[c];

            //GHOST_INSTR_START(chunkloop)
            for (j=0; j<SELL(mat)->chunkLen[c]; j++) 
            { // loop inside chunk

                rhs1  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp0 = _mm256_add_pd(tmp0,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs1));           // accumulate
                rhs2  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp1 = _mm256_add_pd(tmp1,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs2));           // accumulate
                rhs3  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp2 = _mm256_add_pd(tmp2,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs3));           // accumulate
                rhs4  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp3 = _mm256_add_pd(tmp3,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs4));           // accumulate

                rhs1  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp4 = _mm256_add_pd(tmp4,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs1));           // accumulate
                rhs2  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp5 = _mm256_add_pd(tmp5,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs2));           // accumulate
                rhs3  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp6 = _mm256_add_pd(tmp6,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs3));           // accumulate
                rhs4  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp7 = _mm256_add_pd(tmp7,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs4));           // accumulate

                rhs1  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp8 = _mm256_add_pd(tmp8,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs1));           // accumulate
                rhs2  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp9 = _mm256_add_pd(tmp9,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs2));           // accumulate
                rhs3  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp10 = _mm256_add_pd(tmp10,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs3));           // accumulate
                rhs4  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp11 = _mm256_add_pd(tmp11,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs4));           // accumulate

                rhs1  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp12 = _mm256_add_pd(tmp12,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs1));           // accumulate
                rhs2  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp13 = _mm256_add_pd(tmp13,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs2));           // accumulate
                rhs3  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp14 = _mm256_add_pd(tmp14,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs3));           // accumulate
                rhs4  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp15 = _mm256_add_pd(tmp15,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs4));           // accumulate

                rhs1  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp16 = _mm256_add_pd(tmp16,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs1));           // accumulate
                rhs2  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp17 = _mm256_add_pd(tmp17,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs2));           // accumulate
                rhs3  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp18 = _mm256_add_pd(tmp18,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs3));           // accumulate
                rhs4  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp19 = _mm256_add_pd(tmp19,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs4));           // accumulate

                rhs1  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp20 = _mm256_add_pd(tmp20,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs1));           // accumulate
                rhs2  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp21 = _mm256_add_pd(tmp21,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs2));           // accumulate
                rhs3  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp22 = _mm256_add_pd(tmp22,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs3));           // accumulate
                rhs4  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp23 = _mm256_add_pd(tmp23,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs4));           // accumulate

                rhs1  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp24 = _mm256_add_pd(tmp24,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs1));           // accumulate
                rhs2  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp25 = _mm256_add_pd(tmp25,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs2));           // accumulate
                rhs3  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp26 = _mm256_add_pd(tmp26,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs3));           // accumulate
                rhs4  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp27 = _mm256_add_pd(tmp27,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs4));           // accumulate

                rhs1  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp28 = _mm256_add_pd(tmp28,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs1));           // accumulate
                rhs2  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp29 = _mm256_add_pd(tmp29,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs2));           // accumulate
                rhs3  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp30 = _mm256_add_pd(tmp30,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs3));           // accumulate
                rhs4  = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]); // load rhs
                tmp31 = _mm256_add_pd(tmp31,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs4));           // accumulate
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
                _mm256_store_pd(&lval[0],_mm256_add_pd(tmp0,_mm256_load_pd(&lval[0])));
                _mm256_store_pd(&lval[4*1],_mm256_add_pd(tmp1,_mm256_load_pd(&lval[4*1])));
                _mm256_store_pd(&lval[4*2],_mm256_add_pd(tmp2,_mm256_load_pd(&lval[4*2])));
                _mm256_store_pd(&lval[4*3],_mm256_add_pd(tmp3,_mm256_load_pd(&lval[4*3])));
                _mm256_store_pd(&lval[4*4],_mm256_add_pd(tmp4,_mm256_load_pd(&lval[4*4])));
                _mm256_store_pd(&lval[4*5],_mm256_add_pd(tmp5,_mm256_load_pd(&lval[4*5])));
                _mm256_store_pd(&lval[4*6],_mm256_add_pd(tmp6,_mm256_load_pd(&lval[4*6])));
                _mm256_store_pd(&lval[4*7],_mm256_add_pd(tmp7,_mm256_load_pd(&lval[4*7])));
                _mm256_store_pd(&lval[4*8],_mm256_add_pd(tmp8,_mm256_load_pd(&lval[4*8])));
                _mm256_store_pd(&lval[4*9],_mm256_add_pd(tmp9,_mm256_load_pd(&lval[4*9])));
                _mm256_store_pd(&lval[4*10],_mm256_add_pd(tmp10,_mm256_load_pd(&lval[4*10])));
                _mm256_store_pd(&lval[4*11],_mm256_add_pd(tmp11,_mm256_load_pd(&lval[4*11])));
                _mm256_store_pd(&lval[4*12],_mm256_add_pd(tmp12,_mm256_load_pd(&lval[4*12])));
                _mm256_store_pd(&lval[4*13],_mm256_add_pd(tmp13,_mm256_load_pd(&lval[4*13])));
                _mm256_store_pd(&lval[4*14],_mm256_add_pd(tmp14,_mm256_load_pd(&lval[4*14])));
                _mm256_store_pd(&lval[4*15],_mm256_add_pd(tmp15,_mm256_load_pd(&lval[4*15])));
                _mm256_store_pd(&lval[4*16],_mm256_add_pd(tmp16,_mm256_load_pd(&lval[4*16])));
                _mm256_store_pd(&lval[4*17],_mm256_add_pd(tmp17,_mm256_load_pd(&lval[4*17])));
                _mm256_store_pd(&lval[4*18],_mm256_add_pd(tmp18,_mm256_load_pd(&lval[4*18])));
                _mm256_store_pd(&lval[4*19],_mm256_add_pd(tmp19,_mm256_load_pd(&lval[4*19])));
                _mm256_store_pd(&lval[4*20],_mm256_add_pd(tmp20,_mm256_load_pd(&lval[4*20])));
                _mm256_store_pd(&lval[4*21],_mm256_add_pd(tmp21,_mm256_load_pd(&lval[4*21])));
                _mm256_store_pd(&lval[4*22],_mm256_add_pd(tmp22,_mm256_load_pd(&lval[4*22])));
                _mm256_store_pd(&lval[4*23],_mm256_add_pd(tmp23,_mm256_load_pd(&lval[4*23])));
                _mm256_store_pd(&lval[4*24],_mm256_add_pd(tmp24,_mm256_load_pd(&lval[4*24])));
                _mm256_store_pd(&lval[4*25],_mm256_add_pd(tmp25,_mm256_load_pd(&lval[4*25])));
                _mm256_store_pd(&lval[4*26],_mm256_add_pd(tmp26,_mm256_load_pd(&lval[4*26])));
                _mm256_store_pd(&lval[4*27],_mm256_add_pd(tmp27,_mm256_load_pd(&lval[4*27])));
                _mm256_store_pd(&lval[4*28],_mm256_add_pd(tmp28,_mm256_load_pd(&lval[4*28])));
                _mm256_store_pd(&lval[4*29],_mm256_add_pd(tmp29,_mm256_load_pd(&lval[4*29])));
                _mm256_store_pd(&lval[4*30],_mm256_add_pd(tmp30,_mm256_load_pd(&lval[4*30])));
                _mm256_store_pd(&lval[4*31],_mm256_add_pd(tmp31,_mm256_load_pd(&lval[4*31])));

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
