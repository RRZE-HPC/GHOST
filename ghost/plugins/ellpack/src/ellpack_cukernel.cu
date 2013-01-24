#include <ghost.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <ghost_util.h>
#include <ghost_types.h>


__global__ void ELLPACK_kernel(ghost_vdat_t *lhs, ghost_vdat_t *rhs, int options, int nrows, int nrowspadded, ghost_midx_t *rowlen, ghost_midx_t *col, ghost_mdat_t *val)
{
	int i = threadIdx.x+blockIdx.x*blockDim.x;

	if (i<nrows) {
		int j;
		ghost_mdat_t tmp = 0.;

		for (j=0; j<rowlen[i]; j++) {
			tmp += val[i + j*nrowspadded] * rhs[col[i + j*nrowspadded]];
		}
		if (options & GHOST_SPMVM_AXPY)
			lhs[i] += tmp;
		else 
			lhs[i] = tmp;

	}
}	

extern "C" void ELLPACK_kernel_wrap(ghost_vdat_t *lhs, ghost_vdat_t *rhs, int options, int nrows, int nrowspadded, ghost_midx_t *rowlen, ghost_midx_t *col, ghost_mdat_t *val)
{
	ELLPACK_kernel<<<nrowspadded/512,512>>>(lhs, rhs, options, nrows, nrowspadded, rowlen, col, val);
	cudaThreadSynchronize();
}


