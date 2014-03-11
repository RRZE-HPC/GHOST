#include "ghost/config.h"
#include "ghost/omp.h"

#ifdef GHOST_HAVE_MPI
#include <mpi.h> //mpi.h has to be included before stdio.h
#endif
#include <cstdlib>
#include <iostream>
#include <complex>
#include <stdio.h>

#include "ghost/complex.h"
#include "ghost/rand.h"
#include "ghost/util.h"
#include "ghost/densemat.h"
#include "ghost/math.h"
#include "ghost/locality.h"
#include "ghost/log.h"

using namespace std;


template <typename v_t> 
static ghost_error_t ghost_normalizeVector_tmpl(ghost_densemat_t *vec)
{
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_idx_t v;
    v_t *s = NULL;

    GHOST_CALL_GOTO(ghost_malloc((void **)&s,vec->traits.ncols*sizeof(v_t)),err,ret);
    GHOST_CALL_GOTO(ghost_dot(s,vec,vec),err,ret);

    for (v=0; v<vec->traits.ncols; v++)
    {
        s[v] = (v_t)sqrt(s[v]);
        s[v] = (v_t)(((v_t)1.)/s[v]);
    }
    GHOST_CALL_GOTO(vec->vscale(vec,s),err,ret);

    goto out;
err:

out:
    free(s);

    return ret;

}

template <typename v_t> 
static ghost_error_t ghost_vec_dotprod_tmpl(ghost_densemat_t *vec, void *res, ghost_densemat_t *vec2)
{ // the parallelization is done manually because reduction does not work with ghost_complex numbers
    if (vec->traits.nrows != vec2->traits.nrows) {
        WARNING_LOG("The input vectors of the dot product have different numbers of rows");
    }
    if (vec->traits.ncols != vec2->traits.ncols) {
        WARNING_LOG("The input vectors of the dot product have different numbers of columns");
    }
    ghost_idx_t i,v;
    ghost_idx_t nr = MIN(vec->traits.nrows,vec2->traits.nrows);

    int nthreads;
#pragma omp parallel
#pragma omp single
    nthreads = ghost_omp_nthread();

    v_t *partsums;
    GHOST_CALL_RETURN(ghost_malloc((void **)&partsums,16*nthreads*sizeof(v_t)));

    for (v=0; v<MIN(vec->traits.ncols,vec2->traits.ncols); v++) {
        v_t sum = 0;
        for (i=0; i<nthreads*16; i++) partsums[i] = (v_t)0.;

#pragma omp parallel 
        {
            int tid = ghost_omp_threadnum();
#pragma omp for schedule(runtime)
            for (i=0; i<nr; i++) {
                partsums[tid*16] += 
                    *(v_t *)VECVAL(vec2,vec2->val,v,i)*
                    conjugate((v_t *)(VECVAL(vec,vec->val,v,i)));
            }
        }

        for (i=0; i<nthreads; i++) sum += partsums[i*16];

        ((v_t *)res)[v] = sum;
    }

    free(partsums);
    
    return GHOST_SUCCESS;
}

template <typename v_t> 
static ghost_error_t ghost_vec_vaxpy_tmpl(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale)
{
    ghost_idx_t i,v;
    v_t *s = (v_t *)scale;
    ghost_idx_t nr = MIN(vec->traits.nrows,vec2->traits.nrows);

    for (v=0; v<MIN(vec->traits.ncols,vec2->traits.ncols); v++) {
#pragma omp parallel for schedule(runtime) 
        for (i=0; i<nr; i++) {
            *(v_t *)VECVAL(vec,vec->val,v,i) += *(v_t *)VECVAL(vec2,vec2->val,v,i) * s[v];
        }
    }
    return GHOST_SUCCESS;
}

template <typename v_t> 
static ghost_error_t ghost_vec_vaxpby_tmpl(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale, void *b_)
{
    ghost_idx_t i,v;
    v_t *s = (v_t *)scale;
    v_t *b = (v_t *)b_;
    ghost_idx_t nr = MIN(vec->traits.nrows,vec2->traits.nrows);

    for (v=0; v<MIN(vec->traits.ncols,vec2->traits.ncols); v++) {
#pragma omp parallel for schedule(runtime) 
        for (i=0; i<nr; i++) {
            *(v_t *)VECVAL(vec,vec->val,v,i) = *(v_t *)VECVAL(vec2,vec2->val,v,i) * s[v] + 
                *(v_t *)VECVAL(vec,vec->val,v,i) * b[v];
        }
    }
    return GHOST_SUCCESS;
}

template<typename v_t> 
static ghost_error_t ghost_vec_vscale_tmpl(ghost_densemat_t *vec, void *scale)
{
    ghost_idx_t i,v;
    v_t *s = (v_t *)scale;

    for (v=0; v<vec->traits.ncols; v++) {
#pragma omp parallel for schedule(runtime) 
        for (i=0; i<vec->traits.nrows; i++) {
            *(v_t *)VECVAL(vec,vec->val,v,i) *= s[v];
        }
    }
    return GHOST_SUCCESS;
}

// thread-safe type generic random function, returns pseudo-random numbers between -1 and 1.
template <typename v_t>
static void my_rand(unsigned int* state, v_t* result)
{
    // default implementation
    static const v_t scal = (v_t)2.0/(v_t)RAND_MAX;
    static const v_t shift=(v_t)(-1.0);
    *result=(v_t)rand_r(state)*scal+shift;
}

template <typename float_type>
static void my_rand(unsigned int* state, std::complex<float_type>* result)
{
    float_type* ft_res = (float_type*)result;
    my_rand(state,&ft_res[0]);
    my_rand(state,&ft_res[1]);
}

template <typename float_type>
static void my_rand(unsigned int* state, ghost_complex<float_type>* result)
{
    my_rand<float_type>(state,(float_type *)result);
    my_rand<float_type>(state,((float_type *)result)+1);
}



template <typename v_t> 
static ghost_error_t ghost_vec_fromRand_tmpl(ghost_densemat_t *vec)
{
    ghost_vec_malloc(vec);
    DEBUG_LOG(1,"Filling vector with random values");

#pragma omp parallel
    {
    ghost_idx_t i,v;
    unsigned int *state;
    ghost_rand_get(&state);
        for (v=0; v<vec->traits.ncols; v++) 
        {
#pragma omp for schedule(runtime)
            for (i=0; i<vec->traits.nrows; i++) 
            {
                my_rand(state,(v_t *)VECVAL(vec,vec->val,v,i));
            }
        }
    }
    vec->upload(vec);

    return GHOST_SUCCESS;
}


template <typename v_t> 
static ghost_error_t ghost_vec_string_tmpl(char **str, ghost_densemat_t *vec)
{
    stringstream buffer;

    ghost_idx_t i,v;
    for (i=0; i<vec->traits.nrows; i++) {
        for (v=0; v<vec->traits.ncols; v++) {
            v_t val = 0.;
            if (vec->traits.flags & GHOST_DENSEMAT_DEVICE)
            {
#ifdef GHOST_HAVE_CUDA
                ghost_cu_download(&val,&(((v_t *)vec->cu_val)[v*vec->traits.nrowspadded+i]),sizeof(v_t));
#endif
            }
            else if (vec->traits.flags & GHOST_DENSEMAT_HOST)
            {
                val = *(v_t *)VECVAL(vec,vec->val,v,i);
            }
            buffer << val << "\t";
        }
        if (i<vec->traits.nrows-1) {
            buffer << std::endl;
        }
    }
    GHOST_CALL_RETURN(ghost_malloc((void **)str,buffer.str().length()+1));
    strcpy(*str,buffer.str().c_str());

    return GHOST_SUCCESS;
}


extern "C" ghost_error_t d_ghost_densemat_string(char **str, ghost_densemat_t *vec) 
{ return ghost_vec_string_tmpl< double >(str,vec); }

extern "C" ghost_error_t s_ghost_densemat_string(char **str, ghost_densemat_t *vec) 
{ return ghost_vec_string_tmpl< float >(str,vec); }


extern "C" ghost_error_t z_ghost_densemat_string(char **str, ghost_densemat_t *vec) 
{ return ghost_vec_string_tmpl< ghost_complex<double> >(str,vec); }

extern "C" ghost_error_t c_ghost_densemat_string(char **str, ghost_densemat_t *vec) 
{ return ghost_vec_string_tmpl< ghost_complex<float> >(str,vec); }

extern "C" ghost_error_t d_ghost_normalizeVector(ghost_densemat_t *vec) 
{ return ghost_normalizeVector_tmpl< double >(vec); }

extern "C" ghost_error_t s_ghost_normalizeVector(ghost_densemat_t *vec) 
{ return ghost_normalizeVector_tmpl< float >(vec); }

extern "C" ghost_error_t z_ghost_normalizeVector(ghost_densemat_t *vec) 
{ return ghost_normalizeVector_tmpl< ghost_complex<double> >(vec); }

extern "C" ghost_error_t c_ghost_normalizeVector(ghost_densemat_t *vec) 
{ return ghost_normalizeVector_tmpl< ghost_complex<float> >(vec); }

extern "C" ghost_error_t d_ghost_vec_dotprod(ghost_densemat_t *vec, void *res, ghost_densemat_t *vec2) 
{ return ghost_vec_dotprod_tmpl< double >(vec,res,vec2); }

extern "C" ghost_error_t s_ghost_vec_dotprod(ghost_densemat_t *vec, void *res, ghost_densemat_t *vec2) 
{ return ghost_vec_dotprod_tmpl< float >(vec,res,vec2); }

extern "C" ghost_error_t z_ghost_vec_dotprod(ghost_densemat_t *vec, void *res, ghost_densemat_t *vec2) 
{ return ghost_vec_dotprod_tmpl< ghost_complex<double> >(vec,res,vec2); }

extern "C" ghost_error_t c_ghost_vec_dotprod(ghost_densemat_t *vec, void *res, ghost_densemat_t *vec2) 
{ return ghost_vec_dotprod_tmpl< ghost_complex<float> >(vec,res,vec2); }

extern "C" ghost_error_t d_ghost_vec_vscale(ghost_densemat_t *vec, void *scale) 
{ return ghost_vec_vscale_tmpl< double >(vec, scale); }

extern "C" ghost_error_t s_ghost_vec_vscale(ghost_densemat_t *vec, void *scale) 
{ return ghost_vec_vscale_tmpl< float  >(vec, scale); }

extern "C" ghost_error_t z_ghost_vec_vscale(ghost_densemat_t *vec, void *scale) 
{ return ghost_vec_vscale_tmpl< ghost_complex<double> >(vec, scale); }

extern "C" ghost_error_t c_ghost_vec_vscale(ghost_densemat_t *vec, void *scale) 
{ return ghost_vec_vscale_tmpl< ghost_complex<float> >(vec, scale); }

extern "C" ghost_error_t d_ghost_vec_vaxpy(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale) 
{ return ghost_vec_vaxpy_tmpl< double >(vec, vec2, scale); }

extern "C" ghost_error_t s_ghost_vec_vaxpy(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale) 
{ return ghost_vec_vaxpy_tmpl< float >(vec, vec2, scale); }

extern "C" ghost_error_t z_ghost_vec_vaxpy(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale) 
{ return ghost_vec_vaxpy_tmpl< ghost_complex<double> >(vec, vec2, scale); }

extern "C" ghost_error_t c_ghost_vec_vaxpy(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale) 
{ return ghost_vec_vaxpy_tmpl< ghost_complex<float> >(vec, vec2, scale); }

extern "C" ghost_error_t d_ghost_vec_vaxpby(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale, void *b) 
{ return ghost_vec_vaxpby_tmpl< double >(vec, vec2, scale, b); }

extern "C" ghost_error_t s_ghost_vec_vaxpby(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale, void *b) 
{ return ghost_vec_vaxpby_tmpl< float >(vec, vec2, scale, b); }

extern "C" ghost_error_t z_ghost_vec_vaxpby(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale, void *b) 
{ return ghost_vec_vaxpby_tmpl< ghost_complex<double> >(vec, vec2, scale, b); }

extern "C" ghost_error_t c_ghost_vec_vaxpby(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale, void *b) 
{ return ghost_vec_vaxpby_tmpl< ghost_complex<float> >(vec, vec2, scale, b); }

extern "C" ghost_error_t d_ghost_vec_fromRand(ghost_densemat_t *vec) 
{ return ghost_vec_fromRand_tmpl< double >(vec); }

extern "C" ghost_error_t s_ghost_vec_fromRand(ghost_densemat_t *vec) 
{ return ghost_vec_fromRand_tmpl< float >(vec); }

extern "C" ghost_error_t z_ghost_vec_fromRand(ghost_densemat_t *vec) 
{ return ghost_vec_fromRand_tmpl< ghost_complex<double> >(vec); }

extern "C" ghost_error_t c_ghost_vec_fromRand(ghost_densemat_t *vec) 
{ return ghost_vec_fromRand_tmpl< ghost_complex<float> >(vec); }

