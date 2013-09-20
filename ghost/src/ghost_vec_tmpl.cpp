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
	vec->scale(vec,&s);

#ifdef OPENCL
	vec->CLupload(vec);
#endif
#ifdef CUDA
	vec->CUupload(vec);
#endif
}

template <typename v_t> void ghost_vec_dotprod_tmpl(ghost_vec_t *vec, ghost_vec_t *vec2, void *res)
{ // the parallelization is done manually because reduction does not work with ghost_complex numbers
	if (vec->traits->nrows != vec2->traits->nrows) {
		WARNING_LOG("The input vectors of the dot product have different numbers of rows");
	}
	if (vec->traits->nvecs != vec2->traits->nvecs) {
		WARNING_LOG("The input vectors of the dot product have different numbers of columns");
	}
	ghost_vidx_t i,v;
	ghost_vidx_t nr = MIN(vec->traits->nrows,vec2->traits->nrows);

	int nthreads;
#pragma omp parallel
	nthreads = ghost_ompGetNumThreads();

	for (v=0; v<MIN(vec->traits->nvecs,vec2->traits->nvecs); v++) {
		v_t sum = 0;
		v_t partsums[nthreads];
		for (i=0; i<nthreads; i++) partsums[i] = (v_t)0.;

#pragma omp parallel for 
		for (i=0; i<nr; i++) {
			partsums[ghost_ompGetThreadNum()] += 
				((v_t *)(vec->val))[i+vec->traits->nrowspadded*v]*((v_t *)(vec2->val))[i+vec->traits->nrowspadded*v];
		}

		for (i=0; i<nthreads; i++) sum += partsums[i];

		((v_t *)res)[v] = sum;
	}

}

template <typename v_t> void ghost_vec_vaxpy_tmpl(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale)
{
	ghost_vidx_t i,v;
	v_t *s = (v_t *)scale;
	ghost_vidx_t nr = MIN(vec->traits->nrows,vec2->traits->nrows);

	for (v=0; v<MIN(vec->traits->nvecs,vec2->traits->nvecs); v++) {
#pragma omp parallel for 
		for (i=0; i<nr; i++) {
			((v_t *)(vec->val))[i+vec->traits->nrowspadded*v] += ((v_t *)(vec2->val))[i+vec->traits->nrowspadded*v] * s[v];
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
			((v_t *)(vec->val))[i+vec->traits->nrowspadded*v] = ((v_t *)(vec2->val))[i+vec->traits->nrowspadded*v] * s[v] + 
				((v_t *)(vec->val))[i+vec->traits->nrowspadded*v] * b[v];
		}
	}
}

template<typename v_t> void ghost_vec_vscale_tmpl(ghost_vec_t *vec, void *scale)
{
	ghost_vidx_t i,v;
	v_t *s = (v_t *)scale;

	for (v=0; v<vec->traits->nvecs; v++) {
#pragma omp parallel for 
		for (i=0; i<vec->traits->nrows; i++) {
			((v_t *)(vec->val))[i] *= s[v];
		}
	}
}

template <typename v_t> void ghost_vec_fromRand_tmpl(ghost_vec_t *vec)
{
	vec_malloc(vec);
	DEBUG_LOG(1,"Filling vector with random values");
	size_t sizeofdt = ghost_sizeofDataType(vec->traits->datatype);

	vec->val = ghost_malloc(vec->traits->nvecs*vec->traits->nrowspadded*sizeofdt);
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
	vec->upload(vec);

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

