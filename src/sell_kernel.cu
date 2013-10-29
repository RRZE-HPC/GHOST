#define CUDAKERNEL
#include <ghost.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <ghost_util.h>
#include <ghost_types.h>
#include <sell.h>
#include "ghost_complex.h"
#include <cuComplex.h>

#define CHOOSE_KERNEL(dt1,dt2,ch, ...) \
	switch(ch) { \
		case 1: \
				SELL_kernel_CU_tmpl< dt1, dt2, 1 > <<< (int)ceil(SELL(mat)->cumat->nrows/(double)SELL_CUDA_BLOCKSIZE),SELL_CUDA_BLOCKSIZE >>> ( __VA_ARGS__ ); \
		break; \
		case 2: \
				SELL_kernel_CU_tmpl< dt1, dt2, 2 > <<< (int)ceil(SELL(mat)->cumat->nrows/(double)SELL_CUDA_BLOCKSIZE),SELL_CUDA_BLOCKSIZE >>> ( __VA_ARGS__ ); \
		break; \
		case 4: \
				SELL_kernel_CU_tmpl< dt1, dt2, 4 > <<< (int)ceil(SELL(mat)->cumat->nrows/(double)SELL_CUDA_BLOCKSIZE),SELL_CUDA_BLOCKSIZE >>> ( __VA_ARGS__ ); \
		break; \
		case 8: \
				SELL_kernel_CU_tmpl< dt1, dt2, 8 > <<< (int)ceil(SELL(mat)->cumat->nrows/(double)SELL_CUDA_BLOCKSIZE),SELL_CUDA_BLOCKSIZE >>> ( __VA_ARGS__ ); \
		break; \
		case 16: \
				 SELL_kernel_CU_tmpl< dt1, dt2, 16 > <<< (int)ceil(SELL(mat)->cumat->nrows/(double)SELL_CUDA_BLOCKSIZE),SELL_CUDA_BLOCKSIZE >>> ( __VA_ARGS__ ); \
		break; \
		case 32: \
				 SELL_kernel_CU_tmpl< dt1, dt2, 32 > <<< (int)ceil(SELL(mat)->cumat->nrows/(double)SELL_CUDA_BLOCKSIZE),SELL_CUDA_BLOCKSIZE >>> ( __VA_ARGS__ ); \
		break; \
		case 64: \
				 SELL_kernel_CU_tmpl< dt1, dt2, 64 > <<< (int)ceil(SELL(mat)->cumat->nrows/(double)SELL_CUDA_BLOCKSIZE),SELL_CUDA_BLOCKSIZE >>> ( __VA_ARGS__ ); \
		break; \
		case 256: \
				 SELL_kernel_CU_tmpl< dt1, dt2, 256 > <<< (int)ceil(SELL(mat)->cumat->nrows/(double)SELL_CUDA_BLOCKSIZE),SELL_CUDA_BLOCKSIZE >>> ( __VA_ARGS__ ); \
		break; \
		default: \
				 DEBUG_LOG(2,"Calling ELLPACK kernel"); \
				 SELL_kernel_CU_ELLPACK_tmpl< dt1, dt2 > <<< (int)ceil(SELL(mat)->cumat->nrows/(double)SELL_CUDA_BLOCKSIZE),SELL_CUDA_BLOCKSIZE >>> ( __VA_ARGS__ ); \
		}
	/*	default: \
				 return SELL_kernel_CU_ELLPACK_tmpl< dt1, dt2 > <<< (int)ceil(SELL(mat)->cumat->nrows/(double)ch),ch >>> ( __VA_ARGS__ ); \
		break; \
	}*/

template<typename T>
__device__ inline void zero(T &val)
{
	val = 0.;
}

template<>
__device__ inline void zero<cuFloatComplex>(cuFloatComplex &val)
{
	val = make_cuFloatComplex(0.,0.);
}

template<>
__device__ inline void zero<cuDoubleComplex>(cuDoubleComplex &val)
{
	val = make_cuDoubleComplex(0.,0.);
}

// val += val2*val3
template<typename T, typename T2>
__device__ inline T axpy(T val, T val2, T2 val3)
{
	return val+val2*val3;
}

template<>
__device__ inline cuFloatComplex axpy<cuFloatComplex,cuFloatComplex>(cuFloatComplex val, cuFloatComplex val2, cuFloatComplex val3)
{
	return cuCaddf(val,cuCmulf(val2,val3));
}

template<>
__device__ inline cuFloatComplex axpy<cuFloatComplex,double>(cuFloatComplex val, cuFloatComplex val2, double val3)
{
	return cuCaddf(val,cuCmulf(val2,make_cuFloatComplex((float)val3,0.f)));
}

template<>
__device__ inline cuFloatComplex axpy<cuFloatComplex,float>(cuFloatComplex val, cuFloatComplex val2, float val3)
{
	return cuCaddf(val,cuCmulf(val2,make_cuFloatComplex(val3,0.f)));
}

template<>
__device__ inline cuFloatComplex axpy<cuFloatComplex,cuDoubleComplex>(cuFloatComplex val, cuFloatComplex val2, cuDoubleComplex val3)
{
	return cuCaddf(val,cuCmulf(val2,make_cuFloatComplex((float)(cuCreal(val3)),(float)(cuCimag(val3)))));
}

template<>
__device__ inline cuDoubleComplex axpy<cuDoubleComplex,double>(cuDoubleComplex val, cuDoubleComplex val2, double val3)
{
	return cuCadd(val,cuCmul(val2,make_cuDoubleComplex(val3,0.)));
}

template<>
__device__ inline cuDoubleComplex axpy<cuDoubleComplex,float>(cuDoubleComplex val, cuDoubleComplex val2, float val3)
{
	return cuCadd(val,cuCmul(val2,make_cuDoubleComplex((double)val3,0.)));
}

template<>
__device__ inline cuDoubleComplex axpy<cuDoubleComplex,cuDoubleComplex>(cuDoubleComplex val, cuDoubleComplex val2, cuDoubleComplex val3)
{
	return cuCadd(val,cuCmul(val2,val3));
}

template<>
__device__ inline cuDoubleComplex axpy<cuDoubleComplex,cuFloatComplex>(cuDoubleComplex val, cuDoubleComplex val2, cuFloatComplex val3)
{
	return cuCadd(val,cuCmul(val2,make_cuDoubleComplex((double)(cuCrealf(val3)),(double)(cuCimagf(val3)))));
}

template<>
__device__ inline double axpy<double,cuFloatComplex>(double val, double val2, cuFloatComplex val3)
{
	return val+val2*(double)cuCrealf(val3);
}


template<>
__device__ inline double axpy<double,cuDoubleComplex>(double val, double val2, cuDoubleComplex val3)
{
	return val+val2*cuCreal(val3);
}

template<>
__device__ inline float axpy<float,cuFloatComplex>(float val, float val2, cuFloatComplex val3)
{
	return val+val2*cuCrealf(val3);
}


template<>
__device__ inline float axpy<float,cuDoubleComplex>(float val, float val2, cuDoubleComplex val3)
{
	return val+val2*(float)cuCreal(val3);
}

template<typename m_t, typename v_t>  
__global__ void SELL_kernel_CU_ELLPACK_tmpl(v_t *lhs, v_t *rhs, int options, int nrows, int nrowspadded, ghost_midx_t *rowlen, ghost_midx_t *col, m_t *val, ghost_mnnz_t *chunkstart, ghost_midx_t *chunklen)
{
	int i = threadIdx.x+blockIdx.x*blockDim.x;

	if (i<nrows) {
		int j;
		v_t tmp;
		zero<v_t>(tmp);

		for (j=0; j<rowlen[i]; j++) {
			tmp = axpy<v_t,m_t>(tmp, rhs[col[i + j*nrowspadded]], val[i + j*nrowspadded]);
		}
		if (options & GHOST_SPMVM_AXPY)
			lhs[i] = axpy<v_t,float>(lhs[i],tmp,1.f);
		else 
			lhs[i] = tmp;
	}
}

template<typename m_t, typename v_t, int chunkHeight>  
__global__ void SELL_kernel_CU_tmpl(v_t *lhs, v_t *rhs, int options, int nrows, int nrowspadded, ghost_midx_t *rowlen, ghost_midx_t *col, m_t *val, ghost_mnnz_t *chunkstart, ghost_midx_t *chunklen)
{
	int i = threadIdx.x+blockIdx.x*blockDim.x;

//	printf(">>> %d\n",nrows);
	if (i<nrows) {
		int cs, tid;
		if (chunkHeight == SELL_CUDA_BLOCKSIZE) {
		cs = chunkstart[blockIdx.x];
		tid = threadIdx.x;
		} else {
		cs = chunkstart[i/chunkHeight];
		tid = threadIdx.x%chunkHeight;
		}
		int j;
		v_t tmp;
		zero<v_t>(tmp);

		for (j=0; j<rowlen[i]; j++) {
//			printf("%d/%d: %f*%f\n",i,j,rhs[col[cs + tid + j*chunkHeight]], val[cs + tid + j*chunkHeight]);
			tmp = axpy<v_t,m_t>(tmp, rhs[col[cs + tid + j*chunkHeight]], val[cs + tid + j*chunkHeight]);
		}
		if (options & GHOST_SPMVM_AXPY)
			lhs[i] = axpy<v_t,float>(lhs[i],tmp,1.f);
		else 
			lhs[i] = tmp;
	}
}

/*template<typename m_t>  
__global__ void SELL_kernel_CU_cvec_tmpl(cuFloatComplex *lhs, cuFloatComplex *rhs, int options, int nrows, int nrowspadded, ghost_midx_t *rowlen, ghost_midx_t *col, m_t *val, ghost_mnnz_t *chunkstart, ghost_midx_t *chunklen)
{
	int i = threadIdx.x+blockIdx.x*blockDim.x;

	if (i<nrows) {
		int cs = chunkstart[blockIdx.x];
		int j;
		cuFloatComplex tmp = make_cuFloatComplex(0.,0.);


		for (j=0; j<rowlen[i]; j++) {
			tmp += make_cuFloatComplex(val[cs + threadIdx.x + j*SELL_LEN])  // TODO cast besser machen
				* rhs[col[cs + threadIdx.x + j*SELL_LEN]];
		}
		if (options & GHOST_SPMVM_AXPY)
			lhs[i] += tmp;
		else 
			lhs[i] = tmp;
	}
}*/

extern "C" void dd_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(double,double,SELL(mat)->chunkHeight,(double *)lhs->CU_val,(double *)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(double *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen) }

extern "C" void ds_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(double,float,SELL(mat)->chunkHeight,(float *)lhs->CU_val,(float *)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(double *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen) }

extern "C" void dc_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(double,cuFloatComplex,SELL(mat)->chunkHeight,(cuFloatComplex *)lhs->CU_val,(cuFloatComplex *)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(double *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen) }

extern "C" void dz_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(double,cuDoubleComplex,SELL(mat)->chunkHeight,(cuDoubleComplex *)lhs->CU_val,(cuDoubleComplex *)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(double *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen) }

extern "C" void sd_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(float,double,SELL(mat)->chunkHeight,(double *)lhs->CU_val,(double *)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(float *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen) }

extern "C" void ss_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(float,float,SELL(mat)->chunkHeight,(float *)lhs->CU_val,(float *)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(float *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen) }

extern "C" void sc_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(float,cuFloatComplex,SELL(mat)->chunkHeight,(cuFloatComplex *)lhs->CU_val,(cuFloatComplex *)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(float *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen) }

extern "C" void sz_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(float,cuDoubleComplex,SELL(mat)->chunkHeight,(cuDoubleComplex *)lhs->CU_val,(cuDoubleComplex *)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(float *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen) }

extern "C" void cd_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(cuFloatComplex,double,SELL(mat)->chunkHeight,(double *)lhs->CU_val,(double *)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(cuFloatComplex *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen) }

extern "C" void cs_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(cuFloatComplex,float,SELL(mat)->chunkHeight,(float *)lhs->CU_val,(float *)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(cuFloatComplex *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen) }

extern "C" void cc_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(cuFloatComplex,cuFloatComplex,SELL(mat)->chunkHeight,(cuFloatComplex *)lhs->CU_val,(cuFloatComplex *)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(cuFloatComplex *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen) }

extern "C" void cz_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(cuFloatComplex,cuDoubleComplex,SELL(mat)->chunkHeight,(cuDoubleComplex *)lhs->CU_val,(cuDoubleComplex *)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(cuFloatComplex *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen) }

extern "C" void zd_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(cuDoubleComplex,double,SELL(mat)->chunkHeight,(double *)lhs->CU_val,(double *)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(cuDoubleComplex *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen) }

extern "C" void zs_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(cuDoubleComplex,float,SELL(mat)->chunkHeight,(float *)lhs->CU_val,(float *)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(cuDoubleComplex *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen) }

extern "C" void zc_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(cuDoubleComplex,cuFloatComplex,SELL(mat)->chunkHeight,(cuFloatComplex *)lhs->CU_val,(cuFloatComplex *)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(cuDoubleComplex *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen) }

extern "C" void zz_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(cuDoubleComplex,cuDoubleComplex,SELL(mat)->chunkHeight,(cuDoubleComplex *)lhs->CU_val,(cuDoubleComplex *)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(cuDoubleComplex *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen) }

/*extern "C" void ds_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return SELL_kernel_CU_tmpl< double,float > <<<(int)ceil(SELL(mat)->cumat->nrows/256.),256>>> ((float *)lhs->CU_val,(float *)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(double *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen); }

extern "C" void dc_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options){ return SELL_kernel_CU_tmpl< double > <<<(int)ceil(SELL(mat)->cumat->nrows/256.),256>>> ((cuFloatComplex*)lhs->CU_val,(cuFloatComplex*)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(double *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen); }

extern "C" void dz_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return SELL_kernel_CU_tmpl< double,cuDoubleComplex > <<<(int)ceil(SELL(mat)->cumat->nrows/256.),256>>> ((cuDoubleComplex*)lhs->CU_val,(cuDoubleComplex*)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(double *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen); }

extern "C" void sd_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return SELL_kernel_CU_tmpl< float,double > <<<(int)ceil(SELL(mat)->cumat->nrows/256.),256>>> ((double *)lhs->CU_val,(double *)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(float *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen); }

extern "C" void ss_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return SELL_kernel_CU_tmpl< float,float > <<<(int)ceil(SELL(mat)->cumat->nrows/256.),256>>> ((float *)lhs->CU_val,(float *)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(float *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen); }

extern "C" void sc_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return SELL_kernel_CU_tmpl< float,cuComplex > <<<(int)ceil(SELL(mat)->cumat->nrows/256.),256>>> ((cuComplex*)lhs->CU_val,(cuComplex*)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(float *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen); }

extern "C" void sz_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return SELL_kernel_CU_tmpl< float,cuDoubleComplex > <<<(int)ceil(SELL(mat)->cumat->nrows/256.),256>>> ((cuDoubleComplex*)lhs->CU_val,(cuDoubleComplex*)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(float *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen); }

extern "C" void cd_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return SELL_kernel_CU_tmpl< cuFloatComplex,double > <<<(int)ceil(SELL(mat)->cumat->nrows/256.),256>>> ((double *)lhs->CU_val,(double *)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(cuComplex*)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen); }

extern "C" void cs_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return SELL_kernel_CU_tmpl< cuFloatComplex,float > <<<(int)ceil(SELL(mat)->cumat->nrows/256.),256>>> ((float *)lhs->CU_val,(float *)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(cuComplex*)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen); }

extern "C" void cc_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return SELL_kernel_CU_tmpl< cuFloatComplex,cuComplex > <<<(int)ceil(SELL(mat)->cumat->nrows/256.),256>>> ((cuComplex*)lhs->CU_val,(cuComplex*)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(cuComplex*)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen); }

extern "C" void cz_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return SELL_kernel_CU_tmpl< cuFloatComplex,cuDoubleComplex > <<<(int)ceil(SELL(mat)->cumat->nrows/256.),256>>> ((cuDoubleComplex*)lhs->CU_val,(cuDoubleComplex*)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(cuComplex*)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen); }

extern "C" void zd_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return SELL_kernel_CU_tmpl< cuDoubleComplex,double > <<<(int)ceil(SELL(mat)->cumat->nrows/256.),256>>> ((double *)lhs->CU_val,(double *)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(cuDoubleComplex*)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen); }

extern "C" void zs_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return SELL_kernel_CU_tmpl< cuDoubleComplex,float > <<<(int)ceil(SELL(mat)->cumat->nrows/256.),256>>> ((float *)lhs->CU_val,(float *)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(cuDoubleComplex*)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen); }

extern "C" void zc_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return SELL_kernel_CU_tmpl< cuDoubleComplex,cuFloatComplex > <<<(int)ceil(SELL(mat)->cumat->nrows/256.),256>>> ((cuComplex*)lhs->CU_val,(cuComplex*)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(cuDoubleComplex*)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen); }

extern "C" void zz_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return SELL_kernel_CU_tmpl< cuDoubleComplex,cuDoubleComplex > <<<(int)ceil(SELL(mat)->cumat->nrows/256.),256>>> ((cuDoubleComplex*)lhs->CU_val,(cuDoubleComplex*)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(cuDoubleComplex*)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen); }
*/
