#include <ghost_config.h>

#if GHOST_HAVE_MPI
#include <mpi.h> //mpi.h has to be included before stdio.h
#endif
#include <stdio.h>

#include <ghost_complex.h>
#include <ghost_util.h>
#include <ghost_vec.h>
#include <cstdio>
#include <iostream>
#include <omp.h>

double conjugate(double * c) {
	return *c;
}

float conjugate(float * c) {
	return *c;
}

template <typename T>
ghost_complex<T> conjugate(ghost_complex<T>* c) {
	return ghost_complex<T>(std::real(*c),-std::imag(*c));
}

template <typename v_t> void ghost_normalizeVector_tmpl(ghost_vec_t *vec)
{
	v_t s;
	ghost_vec_dotprod_tmpl<v_t>(vec,vec,&s);
	s = (v_t)sqrt(s);
	s = (v_t)(((v_t)1.)/s);
	vec->scale(vec,&s);

#ifdef GHOST_HAVE_OPENCL
	vec->CLupload(vec);
#endif
#ifdef GHOST_HAVE_CUDA
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
				*(v_t *)VECVAL(vec2,vec2->val,v,i)*
				conjugate((v_t *)(VECVAL(vec,vec->val,v,i)));
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

	for (v=0; v<vec->traits->nvecs; v++) {
#pragma omp parallel for 
		for (i=0; i<vec->traits->nrows; i++) {
			*(v_t *)VECVAL(vec,vec->val,v,i) *= s[v];
		}
	}
}

template <typename v_t> void ghost_vec_fromRand_tmpl(ghost_vec_t *vec)
{
	vec_malloc(vec);
	DEBUG_LOG(1,"Filling vector with random values");
	size_t sizeofdt = ghost_sizeofDataType(vec->traits->datatype);

	int i,v;

	// TODO fuse loops but preserve randomness

	for (v=0; v<vec->traits->nvecs; v++) {
#pragma omp parallel for schedule(runtime)
		for (i=0; i<vec->traits->nrows; i++) {
			*(v_t *)VECVAL(vec,vec->val,v,i) = (v_t)0;
		}
	}

	for (v=0; v<vec->traits->nvecs; v++) {
		for (i=0; i<vec->traits->nrows; i++) {
			*(v_t *)VECVAL(vec,vec->val,v,i) = (v_t)(rand()*1./RAND_MAX); // TODO imag
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
		snprintf(prefix,1,"\0");
	}
#else
	snprintf(prefix,1,"\0");
#endif
	
	ghost_vidx_t i,v;

	std::cout.setf(std::ios::fixed, std::ios::floatfield);
	for (i=0; i<vec->traits->nrows; i++) {
		std::cout << prefix << "\t";
		for (v=0; v<vec->traits->nvecs; v++) {
			std::cout << *(v_t *)(VECVAL(vec,vec->val,v,i)) << "\t";
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

