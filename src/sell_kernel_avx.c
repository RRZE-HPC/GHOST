#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/sell.h"
#include "ghost/util.h"
#include "ghost/instr.h"
#include "ghost/machine.h"
#include "ghost/omp.h"
#include <immintrin.h>

#GHOST_FUNC_BEGIN#1,2,4,8,16,32
ghost_error_t dd_SELL_kernel_AVX_$_multivec_cm(ghost_sparsemat_t *mat, ghost_densemat_t* res, ghost_densemat_t* invec, ghost_spmv_flags_t spmvmOptions,va_list argp)
{
#ifdef GHOST_HAVE_AVX
    ghost_idx_t j,c,v,i;
    ghost_nnz_t offs;
    double *lval = NULL, *rval = NULL;
    double *mval = (double *)SELL(mat)->val;
    double *local_dot_product = NULL;
    double *partsums = NULL;
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
    int padding = (int)clsize/sizeof(double);
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

#pragma omp parallel private(v,c,j,val,offs,rhs,rhstmp) shared(partsums)
    {
        #GHOST_UNROLL#__m256d tmp@;#$/4
        int tid = ghost_omp_threadnum();
        __m256d dot1[invec->traits.ncols],dot2[invec->traits.ncols],dot3[invec->traits.ncols];
        for (v=0; v<invec->traits.ncols; v++) {
            dot1[v] = _mm256_setzero_pd();
            dot2[v] = _mm256_setzero_pd();
            dot3[v] = _mm256_setzero_pd();
        }
#pragma omp for schedule(runtime)
        for (c=0; c<mat->nrowsPadded/$; c++) 
        { // loop over chunks

            for (v=0; v<invec->traits.ncols; v++)
            {
                #GHOST_UNROLL#tmp@ = _mm256_setzero_pd();#$/4
                lval = (double *)res->val[v];
                rval = (double *)invec->val[v];
                offs = SELL(mat)->chunkStart[c];

                for (j=0; j<SELL(mat)->chunkLenPadded[c]; j++) 
                { // loop inside chunk

                    #GHOST_UNROLL#val    = _mm256_load_pd(&mval[offs]);rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);rhstmp = _mm_loadl_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);rhstmp = _mm_loadh_pd(rhstmp,&rval[(SELL(mat)->col[offs++])]);rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);tmp@    = _mm256_add_pd(tmp@,_mm256_mul_pd(val,rhs));#$/4
                }

                if (spmvmOptions & (GHOST_SPMV_SHIFT | GHOST_SPMV_VSHIFT)) {
                    if (spmvmOptions & GHOST_SPMV_SHIFT) {
                        shift = _mm256_broadcast_sd(&sshift[0]);
                    } else {
                        shift = _mm256_broadcast_sd(&sshift[v]);
                    }
                    #GHOST_UNROLL#tmp@ = _mm256_sub_pd(tmp@,_mm256_mul_pd(shift,_mm256_load_pd(&rval[c*$+4*@])));#$/4
                }
                if (spmvmOptions & GHOST_SPMV_SCALE) {
                    #GHOST_UNROLL#tmp@ = _mm256_mul_pd(scale,tmp@);#$/4
                }
                if (spmvmOptions & GHOST_SPMV_AXPY) {
                    #GHOST_UNROLL#_mm256_store_pd(&lval[c*$+4*@],_mm256_add_pd(tmp@,_mm256_load_pd(&lval[c*$+4*@])));#$/4
                } else if (spmvmOptions & GHOST_SPMV_AXPBY) {
                    #GHOST_UNROLL#_mm256_store_pd(&lval[c*$+4*@],_mm256_add_pd(tmp@,_mm256_mul_pd(beta,_mm256_load_pd(&lval[c*$+4*@]))));#$/4
                } else {
                    #GHOST_UNROLL#_mm256_stream_pd(&lval[c*$+4*@],tmp@);#$/4
                }

                if (spmvmOptions & GHOST_SPMV_DOT) {
                    if ((c+1)*$ <= mat->nrows) {
                        #GHOST_UNROLL#dot1[v] = _mm256_add_pd(dot1[v],_mm256_mul_pd(_mm256_load_pd(&lval[c*$+4*@]),_mm256_load_pd(&lval[c*$+4*@])));#$/4
                        #GHOST_UNROLL#dot2[v] = _mm256_add_pd(dot2[v],_mm256_mul_pd(_mm256_load_pd(&rval[c*$+4*@]),_mm256_load_pd(&lval[c*$+4*@])));#$/4
                        #GHOST_UNROLL#dot3[v] = _mm256_add_pd(dot3[v],_mm256_mul_pd(_mm256_load_pd(&rval[c*$+4*@]),_mm256_load_pd(&rval[c*$+4*@])));#$/4
                    } else {
                        ghost_idx_t rem;
                        for (rem=0; rem<mat->nrows-c*$; rem++) {
                            partsums[((padding+3*invec->traits.ncols)*tid)+3*v+0] += lval[c*$+rem]*lval[c*$+rem];
                            partsums[((padding+3*invec->traits.ncols)*tid)+3*v+1] += lval[c*$+rem]*rval[c*$+rem];
                            partsums[((padding+3*invec->traits.ncols)*tid)+3*v+2] += rval[c*$+rem]*rval[c*$+rem];
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
#GHOST_FUNC_END

#GHOST_FUNC_BEGIN#1,2,4,8,16,32
ghost_error_t dd_SELL_kernel_AVX_$_multivec_rm(ghost_sparsemat_t *mat, ghost_densemat_t* res, ghost_densemat_t* invec, ghost_spmv_flags_t spmvmOptions,va_list argp)
{
#ifdef GHOST_HAVE_AVX
    ghost_idx_t j,c,col;
    ghost_nnz_t offs;
    double *mval = (double *)SELL(mat)->val;
    double *local_dot_product = NULL;
    double *partsums = NULL;
    __m256d rhs;
    int nthreads = 1, i;
    int ncolspadded = PAD(invec->traits.ncols,4);
    
    unsigned clsize;
    ghost_machine_cacheline_size(&clsize);
    int padding = (int)clsize/sizeof(double);

    UNUSED(argp);
    
    double sscale = 1., sbeta = 1.;
    double *sshift = NULL;
    __m256d shift, scale, beta;

    GHOST_SPMV_PARSE_ARGS(spmvmOptions,argp,sscale,sbeta,sshift,local_dot_product,double);
    scale = _mm256_broadcast_sd(&sscale);
    beta = _mm256_broadcast_sd(&sbeta);
    int axpy = spmvmOptions & (GHOST_SPMV_AXPY | GHOST_SPMV_AXPBY);
    
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
        #GHOST_UNROLL#__m256d tmp@;#$
        double *tmpresult = NULL;
        if (!axpy) {
            ghost_malloc((void **)&tmpresult,sizeof(double)*$*ncolspadded);
        }

        ghost_idx_t donecols;

#pragma omp for schedule(runtime)
        for (c=0; c<mat->nrowsPadded/$; c++) 
        { // loop over chunks
            double *lval = (double *)res->val[c*$];
            double *rval = (double *)invec->val[c*$];

            for (donecols = 0; donecols < ncolspadded; donecols+=4) {
                #GHOST_UNROLL#tmp@ = _mm256_setzero_pd();#$
                offs = SELL(mat)->chunkStart[c];

                for (j=0; j<SELL(mat)->chunkLen[c]; j++) { // loop inside chunk
                    #GHOST_UNROLL#rhs = _mm256_load_pd((double *)invec->val[SELL(mat)->col[offs]]+donecols);tmp@ = _mm256_add_pd(tmp@,_mm256_mul_pd(_mm256_broadcast_sd(&mval[offs++]),rhs));#$
                }
              
                if (spmvmOptions & (GHOST_SPMV_SHIFT | GHOST_SPMV_VSHIFT)) {
                    if (spmvmOptions & GHOST_SPMV_SHIFT) {
                        shift = _mm256_broadcast_sd(&sshift[0]);
                    } else {
                        shift = _mm256_load_pd(&sshift[donecols]);
                    }
                    #GHOST_UNROLL#tmp@ = _mm256_sub_pd(tmp@,_mm256_mul_pd(shift,_mm256_load_pd((double *)invec->val[c*$+@]+donecols)));#$
                }
                if (spmvmOptions & GHOST_SPMV_SCALE) {
                    #GHOST_UNROLL#tmp@ = _mm256_mul_pd(scale,tmp@);#$
                }
                if (axpy) {
                    if (spmvmOptions & GHOST_SPMV_AXPY) {
                        #GHOST_UNROLL#_mm256_store_pd(&lval[invec->traits.ncolspadded*@+donecols],_mm256_add_pd(tmp@,_mm256_load_pd(&lval[invec->traits.ncolspadded*@+donecols])));#$
                    } else if (spmvmOptions & GHOST_SPMV_AXPBY) {
                        #GHOST_UNROLL#_mm256_store_pd(&lval[invec->traits.ncolspadded*@+donecols],_mm256_add_pd(tmp@,_mm256_mul_pd(_mm256_load_pd(&lval[invec->traits.ncolspadded*@+donecols]),beta)));#$
                    }
                    if (spmvmOptions & GHOST_SPMV_DOT) {
                        for (col = donecols; col<donecols+4; col++) {
                            #GHOST_UNROLL#partsums[((padding+3*invec->traits.ncols)*tid)+3*col+0] += lval[col+@*invec->traits.ncolspadded]*lval[col+@*invec->traits.ncolspadded];#$
                            #GHOST_UNROLL#partsums[((padding+3*invec->traits.ncols)*tid)+3*col+1] += lval[col+@*invec->traits.ncolspadded]*rval[col+@*invec->traits.ncolspadded];#$
                            #GHOST_UNROLL#partsums[((padding+3*invec->traits.ncols)*tid)+3*col+2] += rval[col+@*invec->traits.ncolspadded]*rval[col+@*invec->traits.ncolspadded];#$
                        }
                    }
                } else { 
                    #GHOST_UNROLL#_mm256_store_pd(&tmpresult[@*ncolspadded+donecols],tmp@);#$
                    // TODO if non-AXPY cxompute DOT from tmp instead of lval
                }
            }
            if (!axpy) {
                #GHOST_UNROLL#for (donecols = 0; donecols < ncolspadded; donecols+=4) {tmp@ = _mm256_load_pd(&tmpresult[@*ncolspadded+donecols]); _mm256_stream_pd(&lval[@*ncolspadded+donecols],tmp@);}#$
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
#GHOST_FUNC_END
