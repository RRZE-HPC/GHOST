#include "ghost/config.h"

#if GHOST_HAVE_MPI
#include <mpi.h> //mpi.h has to be included before stdio.h
#endif
#include <cstdlib>
#include <iostream>
#include <stdio.h>

#include "ghost/complex.h"
#include "ghost/util.h"
#include "ghost/vec.h"
#include "ghost/math.h"
#include "ghost/constants.h"
#include "ghost/affinity.h"
#include "ghost/blas_mangle.h"




template <typename v_t> void ghost_normalizeVector_tmpl(ghost_vec_t *vec)
{
    ghost_vidx_t v;
    v_t *s = (v_t *)ghost_malloc(vec->traits->nvecs*sizeof(v_t));
    ghost_dotProduct(vec,vec,s);

    for (v=0; v<vec->traits->nvecs; v++)
    {
        s[v] = (v_t)sqrt(s[v]);
        s[v] = (v_t)(((v_t)1.)/s[v]);
    }
    vec->vscale(vec,s);

}

template <typename v_t> void ghost_vec_dotprod_tmpl(ghost_vec_t *vec, ghost_vec_t *vec2, void *res)
{ // the parallelization is done manually because reduction does not work with ghost_complex numbers
    if (vec->traits->nrows != vec2->traits->nrows) {
        WARNING_LOG("The input vectors of the dot product have different numbers of rows");
    }
    if (vec->traits->nvecs != vec2->traits->nvecs) {
        WARNING_LOG("The input vectors of the dot product have different numbers of columns");
    }
    ghost_vidx_t v;
    ghost_blas_idx_t nr = MIN(vec->traits->nrows,vec2->traits->nrows);
    ghost_blas_idx_t incx = 1;
    ghost_blas_idx_t incy = 1;

    for (v=0; v<MIN(vec->traits->nvecs,vec2->traits->nvecs); v++) {
        v_t localres;
        if (vec->traits->datatype & GHOST_BINCRS_DT_COMPLEX) 
        {
            if (vec->traits->datatype & GHOST_BINCRS_DT_DOUBLE) 
            {
              zdotc((ghost_blas_idx_t*)&nr,(BLAS_Complex16*)VECVAL(vec,vec->val,v,0),&incx,(BLAS_Complex16*)VECVAL(vec2,vec2->val,v,0),&incy,(BLAS_Complex16*)&localres);
            }
            else 
            {
              cdotc((ghost_blas_idx_t*)&nr,(BLAS_Complex8*) VECVAL(vec,vec->val,v,0),&incx,(BLAS_Complex8*) VECVAL(vec2,vec2->val,v,0),&incy,(BLAS_Complex8*)&localres);
            }
        } 
        else 
        {
            if (vec->traits->datatype & GHOST_BINCRS_DT_DOUBLE) 
            {
              *(double*)&localres = ddot((ghost_blas_idx_t*)&nr,(double*)VECVAL(vec,vec->val,v,0),&incx,(double*)VECVAL(vec2,vec2->val,v,0),&incy);
            }
            else 
            {
              *(float*)&localres = sdot((ghost_blas_idx_t*)&nr,(float*) VECVAL(vec,vec->val,v,0),&incx,(float*) VECVAL(vec2,vec2->val,v,0),&incy);
            }
        }
        ((v_t *)res)[v] = localres;
    }

/*
    ghost_vidx_t i,v;
    ghost_vidx_t nr = MIN(vec->traits->nrows,vec2->traits->nrows);

    int nthreads;
#pragma omp parallel
    nthreads = ghost_ompGetNumThreads();

    v_t *partsums = (v_t *)ghost_malloc(16*nthreads*sizeof(v_t));
    for (v=0; v<MIN(vec->traits->nvecs,vec2->traits->nvecs); v++) {
        v_t sum = 0;
        for (i=0; i<nthreads*16; i++) partsums[i] = (v_t)0.;

#pragma omp parallel for 
        for (i=0; i<nr; i++) {
            partsums[ghost_ompGetThreadNum()*16] += 
                *(v_t *)VECVAL(vec2,vec2->val,v,i)*
                conjugate((v_t *)(VECVAL(vec,vec->val,v,i)));
        }

        for (i=0; i<nthreads; i++) sum += partsums[i*16];

        ((v_t *)res)[v] = sum;
    }
    free(partsums);
*/
}

template <typename v_t> void ghost_vec_vaxpy_tmpl(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale)
{
    ghost_vidx_t i,v;
    v_t *s = (v_t *)scale;
    ghost_vidx_t nr = MIN(vec->traits->nrows,vec2->traits->nrows);

    for (v=0; v<MIN(vec->traits->nvecs,vec2->traits->nvecs); v++) {
#pragma omp parallel for 
        for (i=0; i<nr; i++) {
            *(v_t *)VECVAL(vec,vec->val,v,i) += *(v_t *)VECVAL(vec2,vec2->val,v,i) * s[v];
        }
    }
}

template <typename v_t> void ghost_vec_vaxpby_tmpl(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale, void *b_)
{
    ghost_vidx_t i,v;
    v_t *s = (v_t *)scale;
    v_t *b = (v_t *)b_;
    ghost_vidx_t nr = MIN(vec->traits->nrows,vec2->traits->nrows);

    for (v=0; v<MIN(vec->traits->nvecs,vec2->traits->nvecs); v++) {
#pragma omp parallel for 
        for (i=0; i<nr; i++) {
            *(v_t *)VECVAL(vec,vec->val,v,i) = *(v_t *)VECVAL(vec2,vec2->val,v,i) * s[v] + 
                *(v_t *)VECVAL(vec,vec->val,v,i) * b[v];
        }
    }
}

template<typename v_t> void ghost_vec_vscale_tmpl(ghost_vec_t *vec, void *scale)
{
    ghost_vidx_t i,v;
    v_t *s = (v_t *)scale;
    ghost_blas_idx_t incx = 1;
    ghost_vidx_t nr = vec->traits->nrows;

    for (v=0; v<vec->traits->nvecs; v++)
    {
        if (vec->traits->datatype & GHOST_BINCRS_DT_COMPLEX) 
        {
            if (vec->traits->datatype & GHOST_BINCRS_DT_DOUBLE) 
                zscal((ghost_blas_idx_t*)&nr,(BLAS_Complex16*)&s[v],(BLAS_Complex16*)VECVAL(vec,vec->val,v,0),&incx);
            else
                cscal((ghost_blas_idx_t*)&nr,(BLAS_Complex8*)&s[v],(BLAS_Complex8*)VECVAL(vec,vec->val,v,0),&incx);
        }
        else
        {
            if (vec->traits->datatype & GHOST_BINCRS_DT_DOUBLE) 
                dscal((ghost_blas_idx_t*)&nr,(double*)&s[v],(double*)VECVAL(vec,vec->val,v,0),&incx);
            else
                sscal((ghost_blas_idx_t*)&nr,(float*)&s[v],(float*)VECVAL(vec,vec->val,v,0),&incx);
        }
    }

    //for (v=0; v<vec->traits->nvecs; v++) {
//#pragma omp parallel for 
        //for (i=0; i<vec->traits->nrows; i++) {
            //*(v_t *)VECVAL(vec,vec->val,v,i) *= s[v];
        //}
    //}
}

// thread-safe type generic random function, returns pseudo-random numbers between -1 and 1.
template <typename v_t>
void my_rand(unsigned int* state, v_t* result)
{
    // default implementation
    static const v_t scal = (v_t)2.0/(v_t)RAND_MAX;
    static const v_t shift=(v_t)(-1.0);
    *result=(v_t)rand_r(state)*scal+shift;
}

template <typename float_type>
void my_rand(unsigned int* state, std::complex<float_type>* result)
{
    float_type* ft_res = (float_type*)result;
    my_rand(state,&ft_res[0]);
    my_rand(state,&ft_res[1]);
}

template <typename float_type>
void my_rand(unsigned int* state, ghost_complex<float_type>* result)
{
    my_rand(state,result);
    my_rand(state,((char *)result)+sizeof(float_type));
}



template <typename v_t> void ghost_vec_fromRand_tmpl(ghost_vec_t *vec)
{
    ghost_vec_malloc(vec);
    DEBUG_LOG(1,"Filling vector with random values");
    ghost_vidx_t i,v;

#pragma omp parallel private (v,i)
    {
    unsigned int* state = ghost_getRandState();
        for (v=0; v<vec->traits->nvecs; v++) 
        {
#pragma omp for
            for (i=0; i<vec->traits->nrows; i++) 
            {
                my_rand(state,(v_t *)VECVAL(vec,vec->val,v,i));
            }
        }
    }
    vec->upload(vec);
}


template <typename v_t> void ghost_vec_print_tmpl(ghost_vec_t *vec)
{
    char prefix[16];
#ifdef GHOST_HAVE_MPI
    if (vec->context != NULL && vec->context->mpicomm != MPI_COMM_NULL) {
        int rank = ghost_getRank(vec->context->mpicomm);
        int ndigits = (int)floor(log10(abs(rank))) + 1;
        snprintf(prefix,4+ndigits,"PE%d: ",rank);
    } else {
        snprintf(prefix,1,"");
    }
#else
    snprintf(prefix,1,"");
#endif

    ghost_vidx_t i,v;
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    for (i=0; i<vec->traits->nrows; i++) {
        std::cout << prefix << "\t";
        for (v=0; v<vec->traits->nvecs; v++) {
            v_t val = 0.;
            if (vec->traits->flags & GHOST_VEC_DEVICE)
            {
#if GHOST_HAVE_CUDA
                CU_copyDeviceToHost(&val,&(((v_t *)vec->CU_val)[v*vec->traits->nrowspadded+i]),sizeof(v_t));
#endif
            }
            else if (vec->traits->flags & GHOST_VEC_HOST)
            {
                val = *(v_t *)VECVAL(vec,vec->val,v,i);
            }
            std::cout << val << "\t";
        }
        std::cout << std::endl;
    }
}


extern "C" void d_ghost_printVector(ghost_vec_t *vec) 
{ return ghost_vec_print_tmpl< double >(vec); }

extern "C" void s_ghost_printVector(ghost_vec_t *vec) 
{ return ghost_vec_print_tmpl< float >(vec); }

extern "C" void z_ghost_printVector(ghost_vec_t *vec) 
{ return ghost_vec_print_tmpl< ghost_complex<double> >(vec); }

extern "C" void c_ghost_printVector(ghost_vec_t *vec) 
{ return ghost_vec_print_tmpl< ghost_complex<float> >(vec); }

extern "C" void d_ghost_normalizeVector(ghost_vec_t *vec) 
{ return ghost_normalizeVector_tmpl< double >(vec); }

extern "C" void s_ghost_normalizeVector(ghost_vec_t *vec) 
{ return ghost_normalizeVector_tmpl< float >(vec); }

extern "C" void z_ghost_normalizeVector(ghost_vec_t *vec) 
{ return ghost_normalizeVector_tmpl< ghost_complex<double> >(vec); }

extern "C" void c_ghost_normalizeVector(ghost_vec_t *vec) 
{ return ghost_normalizeVector_tmpl< ghost_complex<float> >(vec); }

extern "C" void d_ghost_vec_dotprod(ghost_vec_t *vec, ghost_vec_t *vec2, void *res) 
{ return ghost_vec_dotprod_tmpl< double >(vec,vec2,res); }

extern "C" void s_ghost_vec_dotprod(ghost_vec_t *vec, ghost_vec_t *vec2, void *res) 
{ return ghost_vec_dotprod_tmpl< float >(vec,vec2,res); }

extern "C" void z_ghost_vec_dotprod(ghost_vec_t *vec, ghost_vec_t *vec2, void *res) 
{ return ghost_vec_dotprod_tmpl< ghost_complex<double> >(vec,vec2,res); }

extern "C" void c_ghost_vec_dotprod(ghost_vec_t *vec, ghost_vec_t *vec2, void *res) 
{ return ghost_vec_dotprod_tmpl< ghost_complex<float> >(vec,vec2,res); }

extern "C" void d_ghost_vec_vscale(ghost_vec_t *vec, void *scale) 
{ return ghost_vec_vscale_tmpl< double >(vec, scale); }

extern "C" void s_ghost_vec_vscale(ghost_vec_t *vec, void *scale) 
{ return ghost_vec_vscale_tmpl< float  >(vec, scale); }

extern "C" void z_ghost_vec_vscale(ghost_vec_t *vec, void *scale) 
{ return ghost_vec_vscale_tmpl< ghost_complex<double> >(vec, scale); }

extern "C" void c_ghost_vec_vscale(ghost_vec_t *vec, void *scale) 
{ return ghost_vec_vscale_tmpl< ghost_complex<float> >(vec, scale); }

extern "C" void d_ghost_vec_vaxpy(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale) 
{ return ghost_vec_vaxpy_tmpl< double >(vec, vec2, scale); }

extern "C" void s_ghost_vec_vaxpy(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale) 
{ return ghost_vec_vaxpy_tmpl< float >(vec, vec2, scale); }

extern "C" void z_ghost_vec_vaxpy(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale) 
{ return ghost_vec_vaxpy_tmpl< ghost_complex<double> >(vec, vec2, scale); }

extern "C" void c_ghost_vec_vaxpy(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale) 
{ return ghost_vec_vaxpy_tmpl< ghost_complex<float> >(vec, vec2, scale); }

extern "C" void d_ghost_vec_vaxpby(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale, void *b) 
{ return ghost_vec_vaxpby_tmpl< double >(vec, vec2, scale, b); }

extern "C" void s_ghost_vec_vaxpby(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale, void *b) 
{ return ghost_vec_vaxpby_tmpl< float >(vec, vec2, scale, b); }

extern "C" void z_ghost_vec_vaxpby(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale, void *b) 
{ return ghost_vec_vaxpby_tmpl< ghost_complex<double> >(vec, vec2, scale, b); }

extern "C" void c_ghost_vec_vaxpby(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale, void *b) 
{ return ghost_vec_vaxpby_tmpl< ghost_complex<float> >(vec, vec2, scale, b); }

extern "C" void d_ghost_vec_fromRand(ghost_vec_t *vec) 
{ return ghost_vec_fromRand_tmpl< double >(vec); }

extern "C" void s_ghost_vec_fromRand(ghost_vec_t *vec) 
{ return ghost_vec_fromRand_tmpl< float >(vec); }

extern "C" void z_ghost_vec_fromRand(ghost_vec_t *vec) 
{ return ghost_vec_fromRand_tmpl< ghost_complex<double> >(vec); }

extern "C" void c_ghost_vec_fromRand(ghost_vec_t *vec) 
{ return ghost_vec_fromRand_tmpl< ghost_complex<float> >(vec); }

