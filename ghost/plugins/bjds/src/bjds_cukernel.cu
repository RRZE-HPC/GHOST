#define CUDAKERNEL
#include <ghost.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <ghost_util.h>
#include <ghost_types.h>
#include <spm_format_bjds.h>


__global__ void BJDS_kernel(ghost_vdat_t *lhs, ghost_vdat_t *rhs, int options, int nrows, int nrowspadded, ghost_midx_t *rowlen, ghost_midx_t *col, ghost_mdat_t *val, ghost_mnnz_t *chunkstart, ghost_midx_t *chunklen)
{
	int i = threadIdx.x+blockIdx.x*blockDim.x;

	if (i<nrows) {
		int cs = chunkstart[blockIdx.x];
		int j;
		ghost_mdat_t tmp = 0.;

		for (j=0; j<rowlen[i]; j++) {
			tmp += val[cs + threadIdx.x + j*BJDS_LEN] * 
				rhs[col[cs + threadIdx.x + j*BJDS_LEN]];
		}
		if (options & GHOST_SPMVM_AXPY)
			lhs[i] += tmp;
		else 
			lhs[i] = tmp;

	}
}	

extern "C" void BJDS_kernel_wrap(ghost_vdat_t *lhs, ghost_vdat_t *rhs, int options, int nrows, int nrowspadded, ghost_midx_t *rowlen, ghost_midx_t *col, ghost_mdat_t *val, ghost_mnnz_t *chunkstart, ghost_midx_t *chunklen)
{
	BJDS_kernel<<<nrowspadded/BJDS_LEN,BJDS_LEN>>>(lhs, rhs, options, nrows, nrowspadded, rowlen, col, val, chunkstart, chunklen);
	CU_checkerror();
}
