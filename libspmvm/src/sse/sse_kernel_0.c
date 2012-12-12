#include "kernel.h"
#include "ghost_util.h"
#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>

void sse_kernel_0_intr(ghost_vec_t* res, BJDS_TYPE* bjds, ghost_vec_t* invec, int spmvmOptions)
{
	ghost_midx_t c,j;
	ghost_mnnz_t offs;
	__m128d tmp;
	__m128d val;
	__m128d rhs;

//#pragma omp parallel for schedule(runtime) private(j,tmp,val,rhs,offs)
	for (c=0; c<bjds->nrowsPadded>>1; c++) 
	{ // loop over chunks
		tmp = _mm_setzero_pd(); // tmp = 0
		offs = bjds->chunkStart[c];

		for (j=0; j<(bjds->chunkStart[c+1]-bjds->chunkStart[c])>>1; j++) 
		{ // loop inside chunk
			val    = _mm_load_pd(&bjds->val[offs]);                      // load values
			rhs    = _mm_loadl_pd(rhs,&invec->val[(bjds->col[offs++])]); // load first 128 bits of RHS
			rhs    = _mm_loadh_pd(rhs,&invec->val[(bjds->col[offs++])]);
			tmp    = _mm_add_pd(tmp,_mm_mul_pd(val,rhs));           // accumulate
		}
		if (spmvmOptions & GHOST_OPTION_AXPY) {
			_mm_store_pd(&res->val[c*BJDS_LEN],_mm_add_pd(tmp,_mm_load_pd(&res->val[c*BJDS_LEN])));
		} else {
			_mm_stream_pd(&res->val[c*BJDS_LEN],tmp);
		}
	}
}


