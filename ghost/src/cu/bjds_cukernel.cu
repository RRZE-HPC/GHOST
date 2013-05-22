#define CUDAKERNEL
#include <ghost.h>
#include <cuda_runtime.h>
#include <ghost_util.h>
#include <ghost_types.h>
#include <bjds.h>
#include <ghost_cu_types_generic.h>
#include "ghost_complex.h"
#include <cuComplex.h>

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
__global__ void BJDS_kernel_CU_tmpl(v_t *lhs, v_t *rhs, int options, int nrows, int nrowspadded, ghost_midx_t *rowlen, ghost_midx_t *col, m_t *val, ghost_mnnz_t *chunkstart, ghost_midx_t *chunklen)
{
	int i = threadIdx.x+blockIdx.x*blockDim.x;

	if (i<nrows) {
		int cs = chunkstart[blockIdx.x];
		int j;
		v_t tmp;
		zero<v_t>(tmp);

		for (j=0; j<rowlen[i]; j++) {
			tmp = axpy<v_t,m_t>(tmp, rhs[col[cs + threadIdx.x + j*BJDS_LEN]], val[cs + threadIdx.x + j*BJDS_LEN]);
		}
		if (options & GHOST_SPMVM_AXPY)
			lhs[i] = axpy<v_t,float>(lhs[i],tmp,1.f);
		else 
			lhs[i] = tmp;
	}
}

/*template<typename m_t>  
__global__ void BJDS_kernel_CU_cvec_tmpl(cuFloatComplex *lhs, cuFloatComplex *rhs, int options, int nrows, int nrowspadded, ghost_midx_t *rowlen, ghost_midx_t *col, m_t *val, ghost_mnnz_t *chunkstart, ghost_midx_t *chunklen)
{
	int i = threadIdx.x+blockIdx.x*blockDim.x;

	if (i<nrows) {
		int cs = chunkstart[blockIdx.x];
		int j;
		cuFloatComplex tmp = make_cuFloatComplex(0.,0.);


		for (j=0; j<rowlen[i]; j++) {
			tmp += make_cuFloatComplex(val[cs + threadIdx.x + j*BJDS_LEN])  // TODO cast besser machen
				* rhs[col[cs + threadIdx.x + j*BJDS_LEN]];
		}
		if (options & GHOST_SPMVM_AXPY)
			lhs[i] += tmp;
		else 
			lhs[i] = tmp;
	}
}*/

extern "C" void dd_BJDS_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_CU_tmpl< double,double > <<<ceil(BJDS(mat)->cumat->nrows/256.),256>>> ((double *)lhs->CU_val,(double *)rhs->CU_val,options,BJDS(mat)->cumat->nrows,BJDS(mat)->cumat->nrowsPadded,BJDS(mat)->cumat->rowLen,BJDS(mat)->cumat->col,(double *)BJDS(mat)->cumat->val,BJDS(mat)->cumat->chunkStart,BJDS(mat)->cumat->chunkLen); }

extern "C" void ds_BJDS_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_CU_tmpl< double,float > <<<ceil(BJDS(mat)->cumat->nrows/256.),256>>> ((float *)lhs->CU_val,(float *)rhs->CU_val,options,BJDS(mat)->cumat->nrows,BJDS(mat)->cumat->nrowsPadded,BJDS(mat)->cumat->rowLen,BJDS(mat)->cumat->col,(double *)BJDS(mat)->cumat->val,BJDS(mat)->cumat->chunkStart,BJDS(mat)->cumat->chunkLen); }

extern "C" void dc_BJDS_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options){ return BJDS_kernel_CU_tmpl< double > <<<ceil(BJDS(mat)->cumat->nrows/256.),256>>> ((cuFloatComplex*)lhs->CU_val,(cuFloatComplex*)rhs->CU_val,options,BJDS(mat)->cumat->nrows,BJDS(mat)->cumat->nrowsPadded,BJDS(mat)->cumat->rowLen,BJDS(mat)->cumat->col,(double *)BJDS(mat)->cumat->val,BJDS(mat)->cumat->chunkStart,BJDS(mat)->cumat->chunkLen); }

extern "C" void dz_BJDS_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_CU_tmpl< double,cuDoubleComplex > <<<ceil(BJDS(mat)->cumat->nrows/256.),256>>> ((cuDoubleComplex*)lhs->CU_val,(cuDoubleComplex*)rhs->CU_val,options,BJDS(mat)->cumat->nrows,BJDS(mat)->cumat->nrowsPadded,BJDS(mat)->cumat->rowLen,BJDS(mat)->cumat->col,(double *)BJDS(mat)->cumat->val,BJDS(mat)->cumat->chunkStart,BJDS(mat)->cumat->chunkLen); }

extern "C" void sd_BJDS_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_CU_tmpl< float,double > <<<ceil(BJDS(mat)->cumat->nrows/256.),256>>> ((double *)lhs->CU_val,(double *)rhs->CU_val,options,BJDS(mat)->cumat->nrows,BJDS(mat)->cumat->nrowsPadded,BJDS(mat)->cumat->rowLen,BJDS(mat)->cumat->col,(float *)BJDS(mat)->cumat->val,BJDS(mat)->cumat->chunkStart,BJDS(mat)->cumat->chunkLen); }

extern "C" void ss_BJDS_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_CU_tmpl< float,float > <<<ceil(BJDS(mat)->cumat->nrows/256.),256>>> ((float *)lhs->CU_val,(float *)rhs->CU_val,options,BJDS(mat)->cumat->nrows,BJDS(mat)->cumat->nrowsPadded,BJDS(mat)->cumat->rowLen,BJDS(mat)->cumat->col,(float *)BJDS(mat)->cumat->val,BJDS(mat)->cumat->chunkStart,BJDS(mat)->cumat->chunkLen); }

extern "C" void sc_BJDS_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_CU_tmpl< float,cuComplex > <<<ceil(BJDS(mat)->cumat->nrows/256.),256>>> ((cuComplex*)lhs->CU_val,(cuComplex*)rhs->CU_val,options,BJDS(mat)->cumat->nrows,BJDS(mat)->cumat->nrowsPadded,BJDS(mat)->cumat->rowLen,BJDS(mat)->cumat->col,(float *)BJDS(mat)->cumat->val,BJDS(mat)->cumat->chunkStart,BJDS(mat)->cumat->chunkLen); }

extern "C" void sz_BJDS_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_CU_tmpl< float,cuDoubleComplex > <<<ceil(BJDS(mat)->cumat->nrows/256.),256>>> ((cuDoubleComplex*)lhs->CU_val,(cuDoubleComplex*)rhs->CU_val,options,BJDS(mat)->cumat->nrows,BJDS(mat)->cumat->nrowsPadded,BJDS(mat)->cumat->rowLen,BJDS(mat)->cumat->col,(float *)BJDS(mat)->cumat->val,BJDS(mat)->cumat->chunkStart,BJDS(mat)->cumat->chunkLen); }

extern "C" void cd_BJDS_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_CU_tmpl< cuFloatComplex,double > <<<ceil(BJDS(mat)->cumat->nrows/256.),256>>> ((double *)lhs->CU_val,(double *)rhs->CU_val,options,BJDS(mat)->cumat->nrows,BJDS(mat)->cumat->nrowsPadded,BJDS(mat)->cumat->rowLen,BJDS(mat)->cumat->col,(cuComplex*)BJDS(mat)->cumat->val,BJDS(mat)->cumat->chunkStart,BJDS(mat)->cumat->chunkLen); }

extern "C" void cs_BJDS_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_CU_tmpl< cuFloatComplex,float > <<<ceil(BJDS(mat)->cumat->nrows/256.),256>>> ((float *)lhs->CU_val,(float *)rhs->CU_val,options,BJDS(mat)->cumat->nrows,BJDS(mat)->cumat->nrowsPadded,BJDS(mat)->cumat->rowLen,BJDS(mat)->cumat->col,(cuComplex*)BJDS(mat)->cumat->val,BJDS(mat)->cumat->chunkStart,BJDS(mat)->cumat->chunkLen); }

extern "C" void cc_BJDS_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_CU_tmpl< cuFloatComplex,cuComplex > <<<ceil(BJDS(mat)->cumat->nrows/256.),256>>> ((cuComplex*)lhs->CU_val,(cuComplex*)rhs->CU_val,options,BJDS(mat)->cumat->nrows,BJDS(mat)->cumat->nrowsPadded,BJDS(mat)->cumat->rowLen,BJDS(mat)->cumat->col,(cuComplex*)BJDS(mat)->cumat->val,BJDS(mat)->cumat->chunkStart,BJDS(mat)->cumat->chunkLen); }

extern "C" void cz_BJDS_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_CU_tmpl< cuFloatComplex,cuDoubleComplex > <<<ceil(BJDS(mat)->cumat->nrows/256.),256>>> ((cuDoubleComplex*)lhs->CU_val,(cuDoubleComplex*)rhs->CU_val,options,BJDS(mat)->cumat->nrows,BJDS(mat)->cumat->nrowsPadded,BJDS(mat)->cumat->rowLen,BJDS(mat)->cumat->col,(cuComplex*)BJDS(mat)->cumat->val,BJDS(mat)->cumat->chunkStart,BJDS(mat)->cumat->chunkLen); }

extern "C" void zd_BJDS_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_CU_tmpl< cuDoubleComplex,double > <<<ceil(BJDS(mat)->cumat->nrows/256.),256>>> ((double *)lhs->CU_val,(double *)rhs->CU_val,options,BJDS(mat)->cumat->nrows,BJDS(mat)->cumat->nrowsPadded,BJDS(mat)->cumat->rowLen,BJDS(mat)->cumat->col,(cuDoubleComplex*)BJDS(mat)->cumat->val,BJDS(mat)->cumat->chunkStart,BJDS(mat)->cumat->chunkLen); }

extern "C" void zs_BJDS_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_CU_tmpl< cuDoubleComplex,float > <<<ceil(BJDS(mat)->cumat->nrows/256.),256>>> ((float *)lhs->CU_val,(float *)rhs->CU_val,options,BJDS(mat)->cumat->nrows,BJDS(mat)->cumat->nrowsPadded,BJDS(mat)->cumat->rowLen,BJDS(mat)->cumat->col,(cuDoubleComplex*)BJDS(mat)->cumat->val,BJDS(mat)->cumat->chunkStart,BJDS(mat)->cumat->chunkLen); }

extern "C" void zc_BJDS_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_CU_tmpl< cuDoubleComplex,cuFloatComplex > <<<ceil(BJDS(mat)->cumat->nrows/256.),256>>> ((cuComplex*)lhs->CU_val,(cuComplex*)rhs->CU_val,options,BJDS(mat)->cumat->nrows,BJDS(mat)->cumat->nrowsPadded,BJDS(mat)->cumat->rowLen,BJDS(mat)->cumat->col,(cuDoubleComplex*)BJDS(mat)->cumat->val,BJDS(mat)->cumat->chunkStart,BJDS(mat)->cumat->chunkLen); }

extern "C" void zz_BJDS_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_CU_tmpl< cuDoubleComplex,cuDoubleComplex > <<<ceil(BJDS(mat)->cumat->nrows/256.),256>>> ((cuDoubleComplex*)lhs->CU_val,(cuDoubleComplex*)rhs->CU_val,options,BJDS(mat)->cumat->nrows,BJDS(mat)->cumat->nrowsPadded,BJDS(mat)->cumat->rowLen,BJDS(mat)->cumat->col,(cuDoubleComplex*)BJDS(mat)->cumat->val,BJDS(mat)->cumat->chunkStart,BJDS(mat)->cumat->chunkLen); }

