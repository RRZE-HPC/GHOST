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
    int padding = (int)clsize/sizeof(double);
    int axpy = spmvmOptions & (GHOST_SPMV_AXPY | GHOST_SPMV_AXPBY);
    int ncolspadded = PAD(invec->traits.ncols,2);

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
        double *tmpresult = NULL;
        if (!axpy) {
            ghost_malloc((void **)&tmpresult,sizeof(double)*32*ncolspadded);
        }

        ghost_idx_t donecols;

#pragma omp for schedule(runtime)
        for (c=0; c<mat->nrowsPadded>>5; c++) 
        { // loop over chunks
            double *lval = (double *)res->val[c*32];
            double *rval = (double *)invec->val[c*32];

            for (donecols = 0; donecols < ncolspadded; donecols+=2) {
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
                if (axpy) {
                    if (spmvmOptions & GHOST_SPMV_AXPY) {
                        #GHOST_UNROLL#_mm_store_pd(&lval[invec->traits.ncolspadded*@+donecols],_mm_add_pd(tmp@,_mm_load_pd(&lval[invec->traits.ncolspadded*@+donecols])));#32
                    } else if (spmvmOptions & GHOST_SPMV_AXPBY) {
                        #GHOST_UNROLL#_mm_store_pd(&lval[invec->traits.ncolspadded*@+donecols],_mm_add_pd(tmp@,_mm_mul_pd(_mm_load_pd(&lval[invec->traits.ncolspadded*@+donecols]),beta)));#32
                    }                 if (spmvmOptions & GHOST_SPMV_DOT) {
                        for (col = donecols; col<donecols+2; col++) {
                            #GHOST_UNROLL#partsums[((padding+3*invec->traits.ncols)*tid)+3*col+0] += lval[col+@*invec->traits.ncolspadded]*lval[col+@*invec->traits.ncolspadded];#32
                            #GHOST_UNROLL#partsums[((padding+3*invec->traits.ncols)*tid)+3*col+1] += lval[col+@*invec->traits.ncolspadded]*rval[col+@*invec->traits.ncolspadded];#32
                            #GHOST_UNROLL#partsums[((padding+3*invec->traits.ncols)*tid)+3*col+2] += rval[col+@*invec->traits.ncolspadded]*rval[col+@*invec->traits.ncolspadded];#32
                        }
                    }
                } else {
                    #GHOST_UNROLL#_mm_store_pd(&tmpresult[@*ncolspadded+donecols],tmp@);#32
                    // TODO if non-AXPY cxompute DOT from tmp instead of lval
                }

            }
            if (!axpy) {
                #GHOST_UNROLL#for (donecols = 0; donecols < ncolspadded; donecols+=2) {tmp@ = _mm_load_pd(&tmpresult[@*ncolspadded+donecols]); _mm_stream_pd(&lval[@*ncolspadded+donecols],tmp@);}#32
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
