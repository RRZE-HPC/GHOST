#include <ghost.h>
#include <ghost_util.h>
#include <ghost_vec.h>
#include <cmath>
#include <cstdio>
#include "ghost_complex.h"
#include <omp.h>


template <typename v_t> void ghost_normalizeVector_tmpl(ghost_vec_t *vec)
{
	v_t s;
	ghost_vec_dotprod_tmpl<v_t>(vec,vec,&s);
	s = (v_t)sqrt(s);
	s = (v_t)(((v_t)1.)/s);
	ghost_vec_scale_tmpl<v_t>(vec,&s);

#ifdef OPENCL
	vec->CLupload(vec);
#endif
#ifdef CUDA
	vec->CUupload(vec);
#endif
}

template <typename v_t> void ghost_vec_dotprod_tmpl(ghost_vec_t *vec, ghost_vec_t *vec2, void *res)
{ // the parallelization is done manually because reduction does not work with ghost_complex numbers
	ghost_vidx_t i,v;
	ghost_vidx_t nr = MIN(vec->traits->nrows,vec2->traits->nrows);

	int nthreads;
#pragma omp parallel
	nthreads = omp_get_num_threads();

	for (v=0; v<MIN(vec->traits->nvecs,vec2->traits->nvecs); v++) {
		v_t sum = 0;
		v_t partsums[nthreads];
		for (i=0; i<nthreads; i++) partsums[i] = (v_t)0.;

#pragma omp parallel for 
		for (i=0; i<nr; i++) {
			partsums[omp_get_thread_num()] += ((v_t *)(vec->val))[i]*((v_t *)(vec2->val))[i];
		}

		for (i=0; i<nthreads; i++) sum += partsums[i];

		((v_t *)res)[v] = sum;
	}
}

template <typename v_t> void ghost_vec_axpy_tmpl(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale)
{
	ghost_vidx_t i;
	v_t s = *(v_t *)scale;
	ghost_vidx_t nr = MIN(vec->traits->nrows,vec2->traits->nrows);

#pragma omp parallel for 
	for (i=0; i<nr; i++) {
		((v_t *)(vec->val))[i] += ((v_t *)(vec2->val))[i] * s;
	}
}

template<typename v_t> void ghost_vec_scale_tmpl(ghost_vec_t *vec, void *scale)
{
	ghost_vidx_t i;
	v_t s = *(v_t *)scale;

#pragma omp parallel for 
	for (i=0; i<vec->traits->nrows; i++) {
		((v_t *)(vec->val))[i] *= s;
	}
}

template <typename v_t> void ghost_vec_fromRand_tmpl(ghost_vec_t *vec, ghost_context_t * ctx)
{
	DEBUG_LOG(1,"Filling vector with random values");
	getNrowsFromContext(vec,ctx);

	vec->val = ghost_malloc(vec->traits->nvecs*vec->traits->nrowspadded*ghost_sizeofDataType(vec->traits->datatype));
	int i,v;

	// TODO fuse loops but preserve randomness

	if (vec->traits->nvecs > 1) {
#pragma omp parallel for schedule(runtime) private(i)
		for (v=0; v<vec->traits->nvecs; v++) {
			for (i=0; i<vec->traits->nrows; i++) {
				((v_t *)(vec->val))[v*vec->traits->nrowspadded+i] = (v_t)0;
			}
		}
	} else {
#pragma omp parallel for schedule(runtime)
		for (i=0; i<vec->traits->nrows; i++) {
			((v_t *)(vec->val))[i] = (v_t)0;
		}
	}

	for (v=0; v<vec->traits->nvecs; v++) {
		for (i=0; i<vec->traits->nrows; i++) {
			((v_t *)(vec->val))[v*vec->traits->nrowspadded+i] = (v_t)(rand()*1./RAND_MAX); // TODO imag
		}
	}
}

template<typename v_t> int ghost_vecEquals_tmpl(ghost_vec_t *a, ghost_vec_t *b)
{
	double tol = 1e-5; // TODO as argument?
	int i;
	for (i=0; i<a->traits->nrows; i++) {
		if (fabs(real((std::complex<double>)VAL(a,i) - (std::complex<double>)VAL(b,i))) > tol ||
				fabs(imag((std::complex<double>)VAL(a,i) - (std::complex<double>)VAL(b,i))) > tol)
			return 0;
	}
	return 1;
}

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

extern "C" void d_ghost_vec_scale(ghost_vec_t *vec, void *scale) 
{ return ghost_vec_scale_tmpl< double >(vec, scale); }

extern "C" void s_ghost_vec_scale(ghost_vec_t *vec, void *scale) 
{ return ghost_vec_scale_tmpl< float  >(vec, scale); }

extern "C" void z_ghost_vec_scale(ghost_vec_t *vec, void *scale) 
{ return ghost_vec_scale_tmpl< ghost_complex<double> >(vec, scale); }

extern "C" void c_ghost_vec_scale(ghost_vec_t *vec, void *scale) 
{ return ghost_vec_scale_tmpl< ghost_complex<float> >(vec, scale); }

extern "C" void d_ghost_vec_axpy(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale) 
{ return ghost_vec_axpy_tmpl< double >(vec, vec2, scale); }

extern "C" void s_ghost_vec_axpy(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale) 
{ return ghost_vec_axpy_tmpl< float >(vec, vec2, scale); }

extern "C" void z_ghost_vec_axpy(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale) 
{ return ghost_vec_axpy_tmpl< ghost_complex<double> >(vec, vec2, scale); }

extern "C" void c_ghost_vec_axpy(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale) 
{ return ghost_vec_axpy_tmpl< ghost_complex<float> >(vec, vec2, scale); }

extern "C" void d_ghost_vec_fromRand(ghost_vec_t *vec, ghost_context_t *ctx) 
{ return ghost_vec_fromRand_tmpl< double >(vec,ctx); }

extern "C" void s_ghost_vec_fromRand(ghost_vec_t *vec, ghost_context_t *ctx) 
{ return ghost_vec_fromRand_tmpl< float >(vec,ctx); }

extern "C" void z_ghost_vec_fromRand(ghost_vec_t *vec, ghost_context_t *ctx) 
{ return ghost_vec_fromRand_tmpl< ghost_complex<double> >(vec,ctx); }

extern "C" void c_ghost_vec_fromRand(ghost_vec_t *vec, ghost_context_t *ctx) 
{ return ghost_vec_fromRand_tmpl< ghost_complex<float> >(vec,ctx); }

extern "C" int d_ghost_vecEquals(ghost_vec_t *vec, ghost_vec_t *vec2) 
{ return ghost_vecEquals_tmpl< double >(vec,vec2); }

extern "C" int s_ghost_vecEquals(ghost_vec_t *vec, ghost_vec_t *vec2) 
{ return ghost_vecEquals_tmpl< float >(vec,vec2); }

extern "C" int z_ghost_vecEquals(ghost_vec_t *vec, ghost_vec_t *vec2) 
{ return ghost_vecEquals_tmpl< ghost_complex<double> >(vec,vec2); }

extern "C" int c_ghost_vecEquals(ghost_vec_t *vec, ghost_vec_t *vec2) 
{ return ghost_vecEquals_tmpl< ghost_complex<float> >(vec,vec2); }
