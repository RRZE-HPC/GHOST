#define CUDAKERNEL
#include <ghost.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <ghost_util.h>
#include <ghost_types.h>
#include <bjds.h>
#include <ghost_cu_types_generic.h>

#define vecdt float
#define prefix s
#define prefixID GHOST_DT_S_IDX
#include "bjds_cukernel.def"
#undef vecdt
#undef prefix
#undef prefixID

#define vecdt double
#define prefix d
#define prefixID GHOST_DT_D_IDX
#include "bjds_cukernel.def"
#undef vecdt
#undef prefix
#undef prefixID

#define vecdt cuDoubleComplex
#define prefix z
#define prefixID GHOST_DT_Z_IDX
#include "bjds_cukernel.def"
#undef vecdt
#undef prefix
#undef prefixID

#define vecdt cuFloatComplex
#define prefix c
#define prefixID GHOST_DT_C_IDX
#include "bjds_cukernel.def"
#undef vecdt
#undef prefix
#undef prefixID

/*
__global__ void BJDS_kernel(ghost_cu_dt *lhs, ghost_cu_dt *rhs, int options, int nrows, int nrowspadded, ghost_midx_t *rowlen, ghost_midx_t *col, ghost_cu_dt *val, ghost_mnnz_t *chunkstart, ghost_midx_t *chunklen)
{
	int i = threadIdx.x+blockIdx.x*blockDim.x;

	if (i<nrows) {
		int cs = chunkstart[blockIdx.x];
		int j;
		ghost_cu_dt tmp = ZERO;

		for (j=0; j<rowlen[i]; j++) {
			tmp = ADD(tmp, MUL(val[cs + threadIdx.x + j*BJDS_LEN], rhs[col[cs + threadIdx.x + j*BJDS_LEN]]));
		}
		if (options & GHOST_SPMVM_AXPY)
			lhs[i] = ADD(lhs[i],tmp);
		else 
			lhs[i] = tmp;

	}
}	

extern "C" void BJDS_kernel_wrap(ghost_cu_dt *lhs, ghost_cu_dt *rhs, int options, int nrows, int nrowspadded, ghost_midx_t *rowlen, ghost_midx_t *col, ghost_cu_dt *val, ghost_mnnz_t *chunkstart, ghost_midx_t *chunklen)
{
	BJDS_kernel<<<nrowspadded/BJDS_LEN,BJDS_LEN>>>(lhs, rhs, options, nrows, nrowspadded, rowlen, col, val, chunkstart, chunklen);
	CU_checkerror();
}*/
