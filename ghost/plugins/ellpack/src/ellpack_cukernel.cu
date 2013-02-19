#define CUDAKERNEL
#include <ghost.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <ghost_util.h>
#include <ghost_types.h>
#include <ellpack.h>
#include <ghost_cu_types_generic.h>

#define vecdt float
#define prefix s
#define prefixID GHOST_DT_S_IDX
#include "ellpack_cukernel.def"
#undef vecdt
#undef prefix
#undef prefixID

#define vecdt double
#define prefix d
#define prefixID GHOST_DT_D_IDX
#include "ellpack_cukernel.def"
#undef vecdt
#undef prefix
#undef prefixID

#define vecdt cuDoubleComplex
#define prefix z
#define prefixID GHOST_DT_Z_IDX
#include "ellpack_cukernel.def"
#undef vecdt
#undef prefix
#undef prefixID

#define vecdt cuFloatComplex
#define prefix c
#define prefixID GHOST_DT_C_IDX
#include "ellpack_cukernel.def"
#undef vecdt
#undef prefix
#undef prefixID

/*
__global__ void ELLPACK_kernel(ghost_cu_dt *lhs, ghost_cu_dt *rhs, int options, int nrows, int nrowspadded, ghost_midx_t *rowlen, ghost_midx_t *col, ghost_cu_dt *val)
{
	int i = threadIdx.x+blockIdx.x*blockDim.x;

	if (i<nrows) {
		int j;
		ghost_cu_dt tmp = ZERO;


		for (j=0; j<rowlen[i]; j++) {
			tmp = ADD(tmp, MUL(val[i + j*nrowspadded], rhs[col[i + j*nrowspadded]]));
		}
		if (options & GHOST_SPMVM_AXPY)
			lhs[i] = ADD(lhs[i], tmp);
		else 
			lhs[i] = tmp;

	}
}	

extern "C" void ELLPACK_kernel_wrap(ghost_cu_dt *lhs, ghost_cu_dt *rhs, int options, int nrows, int nrowspadded, ghost_midx_t *rowlen, ghost_midx_t *col, ghost_cu_dt *val)
{
	ELLPACK_kernel<<<nrowspadded/ELLPACK_WGXSIZE,ELLPACK_WGXSIZE>>>(lhs, rhs, options, nrows, nrowspadded, rowlen, col, val);
	CU_checkerror();
	
}*/


