#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/sell.h"
#include "ghost/util.h"
#include "ghost/instr.h"
#include "ghost/machine.h"
#include "ghost/omp.h"
#include <immintrin.h>

ghost_error_t dd_SELL_kernel_SSE_32_multivec_cm(ghost_sparsemat_t *mat, ghost_densemat_t* res, ghost_densemat_t* invec, ghost_spmv_flags_t spmvmOptions,va_list argp)
{
#ifdef GHOST_HAVE_SSE
    ghost_idx_t j,c,v,i;
    ghost_nnz_t offs;
    double *lval = NULL, *rval = NULL;
    double *mval = (double *)SELL(mat)->val;
    double *local_dot_product = NULL;
    double *partsums = NULL;
    __m128d val;
    __m128d rhs;
    
    double sscale = 1., sbeta = 1.;
    double *sshift = NULL;
    __m128d shift, scale, beta;

    GHOST_SPMV_PARSE_ARGS(spmvmOptions,argp,sscale,sbeta,sshift,local_dot_product,double);
    scale = _mm_load1_pd(&sscale);
    beta = _mm_load1_pd(&sbeta);

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

#pragma omp parallel private(v,c,j,val,offs,rhs) shared(partsums)
    {
        __m128d tmp0,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,tmp9,tmp10,tmp11,tmp12,tmp13,tmp14,tmp15;
        int tid = ghost_omp_threadnum();
        __m128d dot1[invec->traits.ncols],dot2[invec->traits.ncols],dot3[invec->traits.ncols];
        for (v=0; v<invec->traits.ncols; v++) {
            dot1[v] = _mm_setzero_pd();
            dot2[v] = _mm_setzero_pd();
            dot3[v] = _mm_setzero_pd();
        }
#pragma omp for schedule(runtime)
        for (c=0; c<mat->nrowsPadded>>5; c++) 
        { // loop over chunks

            for (v=0; v<invec->traits.ncols; v++)
            {
                #GHOST_UNROLL#tmp@ = _mm_setzero_pd();#16
                lval = (double *)res->val[v];
                rval = (double *)invec->val[v];
                offs = SELL(mat)->chunkStart[c];

                for (j=0; j<(SELL(mat)->chunkStart[c+1]-SELL(mat)->chunkStart[c])>>5; j++) 
                { // loop inside chunk

                    #GHOST_UNROLL#val = _mm_load_pd(&mval[offs]);rhs = _mm_loadl_pd(rhs,&rval[(SELL(mat)->col[offs++])]);rhs = _mm_loadh_pd(rhs,&rval[(SELL(mat)->col[offs++])]);tmp@ = _mm_add_pd(tmp@,_mm_mul_pd(val,rhs));#16
                }

                if (spmvmOptions & (GHOST_SPMV_SHIFT | GHOST_SPMV_VSHIFT)) {
                    if (spmvmOptions & GHOST_SPMV_SHIFT) {
                        shift = _mm_load1_pd(&sshift[0]);
                    } else {
                        shift = _mm_load1_pd(&sshift[v]);
                    }
                    #GHOST_UNROLL#tmp@ = _mm_sub_pd(tmp@,_mm_mul_pd(shift,_mm_load_pd(&rval[c*32+2*@])));#16
                }
                if (spmvmOptions & GHOST_SPMV_SCALE) {
                    #GHOST_UNROLL#tmp@ = _mm_mul_pd(scale,tmp@);#16
                }
                if (spmvmOptions & GHOST_SPMV_AXPY) {
                    #GHOST_UNROLL#_mm_store_pd(&lval[c*32+2*@],_mm_add_pd(tmp@,_mm_load_pd(&lval[c*32+2*@])));#16
                } else if (spmvmOptions & GHOST_SPMV_AXPBY) {
                    #GHOST_UNROLL#_mm_store_pd(&lval[c*32+2*@],_mm_add_pd(tmp@,_mm_mul_pd(beta,_mm_load_pd(&lval[c*32+2*@]))));#16
                } else {
                    #GHOST_UNROLL#_mm_stream_pd(&lval[c*32+2*@],tmp@);#16
                }
                if (spmvmOptions & GHOST_SPMV_DOT) {
                    if ((c+1)*32 <= mat->nrows) {
                        #GHOST_UNROLL#dot1[v] = _mm_add_pd(dot1[v],_mm_mul_pd(_mm_load_pd(&lval[c*32+2*@]),_mm_load_pd(&lval[c*32+2*@])));#16
                        #GHOST_UNROLL#dot2[v] = _mm_add_pd(dot2[v],_mm_mul_pd(_mm_load_pd(&rval[c*32+2*@]),_mm_load_pd(&lval[c*32+2*@])));#16
                        #GHOST_UNROLL#dot3[v] = _mm_add_pd(dot3[v],_mm_mul_pd(_mm_load_pd(&rval[c*32+2*@]),_mm_load_pd(&rval[c*32+2*@])));#16
                    } else {
                        ghost_idx_t rem;
                        for (rem=0; rem<mat->nrows-c*32; rem++) {
                            partsums[((padding+3*invec->traits.ncols)*tid)+3*v+0] += lval[c*32+rem]*lval[c*32+rem];
                            partsums[((padding+3*invec->traits.ncols)*tid)+3*v+1] += lval[c*32+rem]*rval[c*32+rem];
                            partsums[((padding+3*invec->traits.ncols)*tid)+3*v+2] += rval[c*32+rem]*rval[c*32+rem];
                        }
                    }
                }
            }
        }

        if (spmvmOptions & GHOST_SPMV_DOT) {
            __m128d hsum;
            for (v=0; v<invec->traits.ncols; v++) {

                hsum = _mm_hadd_pd(dot1[v],dot2[v]);
                partsums[((padding+3*invec->traits.ncols)*tid)+3*v+0] += ((double *)&hsum)[0];
                partsums[((padding+3*invec->traits.ncols)*tid)+3*v+1] += ((double *)&hsum)[1];

                hsum = _mm_hadd_pd(dot3[v],dot3[v]);
                partsums[((padding+3*invec->traits.ncols)*tid)+3*v+2] += ((double *)&hsum)[0];
            }
        }
    }
    if (spmvmOptions & GHOST_SPMV_DOT) {
        for (v=0; v<invec->traits.ncols; v++) {
            local_dot_product[v                       ] = 0.; 
            local_dot_product[v  +   invec->traits.ncols] = 0.;
            local_dot_product[v  + 2*invec->traits.ncols] = 0.;
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
ghost_error_t dd_SELL_kernel_SSE_32_multivec_rm(ghost_sparsemat_t *mat, ghost_densemat_t* res, ghost_densemat_t* invec, ghost_spmv_flags_t spmvmOptions,va_list argp)
{
#ifdef GHOST_HAVE_SSE
    ghost_idx_t j,c,col;
    ghost_nnz_t offs;
    double *mval = (double *)SELL(mat)->val;
    double *local_dot_product = NULL;
    double *partsums = NULL;
    __m128d rhs;
    int nthreads = 1, i;
    
    unsigned clsize;
    ghost_machine_cacheline_size(&clsize);
    unsigned padding = clsize/sizeof(double);

    double sscale = 1., sbeta = 1.;
    double *sshift = NULL;
    __m128d shift, scale, beta;

    GHOST_SPMV_PARSE_ARGS(spmvmOptions,argp,sscale,sbeta,sshift,local_dot_product,double);
    scale = _mm_load1_pd(&sscale);
    beta = _mm_load1_pd(&sbeta);
    
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
        #GHOST_UNROLL#__m128d tmp@;#32

        ghost_idx_t remainder;
        ghost_idx_t donecols;

#pragma omp for schedule(runtime)
        for (c=0; c<mat->nrowsPadded>>5; c++) 
        { // loop over chunks
            remainder = invec->traits.ncols;
            donecols = 0;
            double *lval = (double *)res->val[c*32];
            double *rval = (double *)invec->val[c*32];

            while(remainder >= 2) { // this is done multiple times
                #GHOST_UNROLL#tmp@ = _mm_setzero_pd();#32
                offs = SELL(mat)->chunkStart[c];

                for (j=0; j<SELL(mat)->chunkLen[c]; j++) { // loop inside chunk
                    #GHOST_UNROLL#rhs = _mm_load_pd((double *)invec->val[SELL(mat)->col[offs]]+donecols);tmp@ = _mm_add_pd(tmp@,_mm_mul_pd(_mm_load1_pd(&mval[offs++]),rhs));#32
                }
              
                if (spmvmOptions & (GHOST_SPMV_SHIFT | GHOST_SPMV_VSHIFT)) {
                    if (spmvmOptions & GHOST_SPMV_SHIFT) {
                        shift = _mm_load1_pd(&sshift[0]);
                    } else {
                        shift = _mm_load_pd(&sshift[donecols]);
                    }
                    #GHOST_UNROLL#tmp@ = _mm_sub_pd(tmp@,_mm_mul_pd(shift,_mm_load_pd((double *)invec->val[c*32+@]+donecols)));#32
                }
                if (spmvmOptions & GHOST_SPMV_SCALE) {
                    #GHOST_UNROLL#tmp@ = _mm_mul_pd(scale,tmp@);#32
                }
                if (spmvmOptions & GHOST_SPMV_AXPY) {
                    #GHOST_UNROLL#_mm_store_pd(&lval[invec->traits.ncols*@+donecols],_mm_add_pd(tmp@,_mm_load_pd(&lval[invec->traits.ncols*@+donecols])));#32
                } else if (spmvmOptions & GHOST_SPMV_AXPBY) {
                    #GHOST_UNROLL#_mm_store_pd(&lval[invec->traits.ncols*@+donecols],_mm_add_pd(tmp@,_mm_mul_pd(_mm_load_pd(&lval[invec->traits.ncols*@+donecols]),beta)));#32
                } else {
                    #GHOST_UNROLL#_mm_store_pd(&lval[invec->traits.ncols*@+donecols],tmp@);#32
                }
                if (spmvmOptions & GHOST_SPMV_DOT) {
                    for (col = donecols; col<donecols+2; col++) {
                        #GHOST_UNROLL#partsums[((padding+3*invec->traits.ncols)*tid)+3*col+0] += lval[col+@*invec->traits.ncols]*lval[col+@*invec->traits.ncols];#32
                        #GHOST_UNROLL#partsums[((padding+3*invec->traits.ncols)*tid)+3*col+1] += lval[col+@*invec->traits.ncols]*rval[col+@*invec->traits.ncols];#32
                        #GHOST_UNROLL#partsums[((padding+3*invec->traits.ncols)*tid)+3*col+2] += rval[col+@*invec->traits.ncols]*rval[col+@*invec->traits.ncols];#32
                    }
                }
                
                donecols += 2; 
                remainder -= 2;
            }
            if (remainder) {
                #GHOST_UNROLL#tmp@ = _mm_setzero_pd();#32
                offs = SELL(mat)->chunkStart[c];

                for (j=0; j<SELL(mat)->chunkLen[c]; j++) { // loop inside chunk
                    #GHOST_UNROLL#rhs = _mm_load_sd((double *)invec->val[SELL(mat)->col[offs]]+donecols);tmp@ = _mm_add_pd(tmp@,_mm_mul_pd(_mm_load1_pd(&mval[offs++]),rhs));#32
                }
              
                if (spmvmOptions & (GHOST_SPMV_SHIFT | GHOST_SPMV_VSHIFT)) {
                    if (spmvmOptions & GHOST_SPMV_SHIFT) {
                        shift = _mm_load1_pd(&sshift[0]);
                    } else {
                        shift = _mm_load_sd(&sshift[donecols]);
                    }
                    #GHOST_UNROLL#tmp@ = _mm_sub_pd(tmp@,_mm_mul_pd(shift,_mm_load_sd((double *)invec->val[c*32+@]+donecols)));#32
                }
                if (spmvmOptions & GHOST_SPMV_SCALE) {
                    #GHOST_UNROLL#tmp@ = _mm_mul_pd(scale,tmp@);#32
                }
                if (spmvmOptions & GHOST_SPMV_AXPY) {
                    #GHOST_UNROLL#_mm_store_sd(&lval[invec->traits.ncols*@+donecols],_mm_add_pd(tmp@,_mm_load_sd(&lval[invec->traits.ncols*@+donecols])));#32
                } else if (spmvmOptions & GHOST_SPMV_AXPBY) {
                    #GHOST_UNROLL#_mm_store_sd(&lval[invec->traits.ncols*@+donecols],_mm_add_pd(tmp@,_mm_mul_pd(_mm_load_sd(&lval[invec->traits.ncols*@+donecols]),beta)));#32
                } else {
                    #GHOST_UNROLL#_mm_store_sd(&lval[invec->traits.ncols*@+donecols],tmp@);#32
                }
                if (spmvmOptions & GHOST_SPMV_DOT) {
                    for (col = donecols; col<donecols+1; col++) {
                        #GHOST_UNROLL#partsums[((padding+3*invec->traits.ncols)*tid)+3*col+0] += lval[col+@*invec->traits.ncols]*lval[col+@*invec->traits.ncols];#32
                        #GHOST_UNROLL#partsums[((padding+3*invec->traits.ncols)*tid)+3*col+1] += lval[col+@*invec->traits.ncols]*rval[col+@*invec->traits.ncols];#32
                        #GHOST_UNROLL#partsums[((padding+3*invec->traits.ncols)*tid)+3*col+2] += rval[col+@*invec->traits.ncols]*rval[col+@*invec->traits.ncols];#32
                    }
                }
            }
        }
    }
    if (spmvmOptions & GHOST_SPMV_DOT) {
        for (col=0; col<invec->traits.ncols; col++) {
            local_dot_product[col                       ] = 0.; 
            local_dot_product[col  +   invec->traits.ncols] = 0.;
            local_dot_product[col  + 2*invec->traits.ncols] = 0.;
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

ghost_error_t dd_SELL_kernel_AVX_32_multivec_cm(ghost_sparsemat_t *mat, ghost_densemat_t* res, ghost_densemat_t* invec, ghost_spmv_flags_t spmvmOptions,va_list argp)
{
#ifdef GHOST_HAVE_AVX
    ghost_idx_t j,c,v,i;
    ghost_nnz_t offs;
    double *lval = NULL, *rval = NULL;
    double *mval = (double *)SELL(mat)->val;
    double *local_dot_product = NULL;
    double *partsums = NULL;
    __m256d tmp0,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7;
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

#pragma omp parallel private(v,c,j,tmp0,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,val,offs,rhs,rhstmp) shared(partsums)
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
                #GHOST_UNROLL#tmp@ = _mm256_setzero_pd();#8
                lval = (double *)res->val[v];
                rval = (double *)invec->val[v];
                offs = SELL(mat)->chunkStart[c];

                for (j=0; j<(SELL(mat)->chunkStart[c+1]-SELL(mat)->chunkStart[c])>>5; j++) 
                { // loop inside chunk

                    #GHOST_UNROLL#val    = _mm256_load_pd(&mval[offs]);rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);tmp@    = _mm256_add_pd(tmp@,_mm256_mul_pd(val,rhs));#8
                }

                if (spmvmOptions & (GHOST_SPMV_SHIFT | GHOST_SPMV_VSHIFT)) {
                    if (spmvmOptions & GHOST_SPMV_SHIFT) {
                        shift = _mm256_broadcast_sd(&sshift[0]);
                    } else {
                        shift = _mm256_broadcast_sd(&sshift[v]);
                    }
                    #GHOST_UNROLL#tmp@ = _mm256_sub_pd(tmp@,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*32+4*@])));#8
                }
                if (spmvmOptions & GHOST_SPMV_SCALE) {
                    #GHOST_UNROLL#tmp@ = _mm256_mul_pd(scale,tmp@);#8
                }
                if (spmvmOptions & GHOST_SPMV_AXPY) {
                    #GHOST_UNROLL#_mm256_store_pd(&lval[c*32+4*@],_mm256_add_pd(tmp@,_mm256_load_pd(&lval[c*32+4*@])));#8
                } else if (spmvmOptions & GHOST_SPMV_AXPBY) {
                    #GHOST_UNROLL#_mm256_store_pd(&lval[c*32+4*@],_mm256_add_pd(tmp@,_mm256_mul_pd(beta,_mm256_load_pd(&lval[c*32+4*@]))));#8
                } else {
                    #GHOST_UNROLL#_mm256_stream_pd(&lval[c*32+4*@],tmp@);#8
                }

                if (spmvmOptions & GHOST_SPMV_DOT) {
                    if ((c+1)*32 <= mat->nrows) {
                        #GHOST_UNROLL#dot1[v] = _mm256_add_pd(dot1[v],_mm256_mul_pd(_mm256_load_pd(&lval[c*32+4*@]),_mm256_load_pd(&lval[c*32+4*@])));#8
                        #GHOST_UNROLL#dot2[v] = _mm256_add_pd(dot2[v],_mm256_mul_pd(_mm256_load_pd(&rval[c*32+4*@]),_mm256_load_pd(&lval[c*32+4*@])));#8
                        #GHOST_UNROLL#dot3[v] = _mm256_add_pd(dot3[v],_mm256_mul_pd(_mm256_load_pd(&rval[c*32+4*@]),_mm256_load_pd(&rval[c*32+4*@])));#8
                    } else {
                        ghost_idx_t rem;
                        for (rem=0; rem<mat->nrows-c*32; rem++) {
                            partsums[((padding+3*invec->traits.ncols)*tid)+3*v+0] += lval[c*32+rem]*lval[c*32+rem];
                            partsums[((padding+3*invec->traits.ncols)*tid)+3*v+1] += lval[c*32+rem]*rval[c*32+rem];
                            partsums[((padding+3*invec->traits.ncols)*tid)+3*v+2] += rval[c*32+rem]*rval[c*32+rem];
                        }
                    }

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
            local_dot_product[v                       ] = 0.; 
            local_dot_product[v  +   invec->traits.ncols] = 0.;
            local_dot_product[v  + 2*invec->traits.ncols] = 0.;
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

ghost_error_t dd_SELL_kernel_AVX_32_multivec_rm(ghost_sparsemat_t *mat, ghost_densemat_t* res, ghost_densemat_t* invec, ghost_spmv_flags_t spmvmOptions,va_list argp)
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
            local_dot_product[col                       ] = 0.; 
            local_dot_product[col  +   invec->traits.ncols] = 0.;
            local_dot_product[col  + 2*invec->traits.ncols] = 0.;
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
