#include "kernel_helper.h"
#include "kernel.h"
#include <stdio.h>

#ifdef LIKWID
#include <likwid.h>
#endif
void mic_kernel_0(VECTOR_TYPE* res, MICVEC_TYPE* mv, VECTOR_TYPE* invec, int spmvmOptions)
{
	int c,j;
	data_t tmp[MICVEC_LEN]; 

#pragma omp parallel for schedule(runtime) private(j,tmp)
	for (c=0; c<mv->nRowsPadded/MICVEC_LEN; c++) 
	{ // loop over chunks
		tmp[0] = 0.;
		tmp[1] = 0.;
		tmp[2] = 0.;
		tmp[3] = 0.;
#if MICVEC_LEN > 4
		tmp[4] = 0.;
		tmp[5] = 0.;
		tmp[6] = 0.;
		tmp[7] = 0.;
#endif

		for (j=0; j<(mv->chunkStart[c+1]-mv->chunkStart[c])/MICVEC_LEN; j++) 
		{ // loop inside chunk
			tmp[0] += mv->val[mv->chunkStart[c]+j*MICVEC_LEN  ] * invec->val[mv->col[mv->chunkStart[c]+j*MICVEC_LEN  ]];
			tmp[1] += mv->val[mv->chunkStart[c]+j*MICVEC_LEN+1] * invec->val[mv->col[mv->chunkStart[c]+j*MICVEC_LEN+1]];
			tmp[2] += mv->val[mv->chunkStart[c]+j*MICVEC_LEN+2] * invec->val[mv->col[mv->chunkStart[c]+j*MICVEC_LEN+2]];
			tmp[3] += mv->val[mv->chunkStart[c]+j*MICVEC_LEN+3] * invec->val[mv->col[mv->chunkStart[c]+j*MICVEC_LEN+3]];
#if MICVEC_LEN > 4
			tmp[4] += mv->val[mv->chunkStart[c]+j*MICVEC_LEN+4] * invec->val[mv->col[mv->chunkStart[c]+j*MICVEC_LEN+4]];
			tmp[5] += mv->val[mv->chunkStart[c]+j*MICVEC_LEN+5] * invec->val[mv->col[mv->chunkStart[c]+j*MICVEC_LEN+5]];
			tmp[6] += mv->val[mv->chunkStart[c]+j*MICVEC_LEN+6] * invec->val[mv->col[mv->chunkStart[c]+j*MICVEC_LEN+6]];
			tmp[7] += mv->val[mv->chunkStart[c]+j*MICVEC_LEN+7] * invec->val[mv->col[mv->chunkStart[c]+j*MICVEC_LEN+7]];
#endif

		}
		if (spmvmOptions & SPMVM_OPTION_AXPY) { 
			res->val[c*MICVEC_LEN  ] += tmp[0];
			res->val[c*MICVEC_LEN+1] += tmp[1];
			res->val[c*MICVEC_LEN+2] += tmp[2];
			res->val[c*MICVEC_LEN+3] += tmp[3];
#if MICVEC_LEN > 4
			res->val[c*MICVEC_LEN+4] += tmp[4];
			res->val[c*MICVEC_LEN+5] += tmp[5];
			res->val[c*MICVEC_LEN+6] += tmp[6];
			res->val[c*MICVEC_LEN+7] += tmp[7];
#endif
		} else {
			res->val[c*MICVEC_LEN  ] = tmp[0];
			res->val[c*MICVEC_LEN+1] = tmp[1];
			res->val[c*MICVEC_LEN+2] = tmp[2];
			res->val[c*MICVEC_LEN+3] = tmp[3];
#if MICVEC_LEN > 4
			res->val[c*MICVEC_LEN+4] = tmp[4];
			res->val[c*MICVEC_LEN+5] = tmp[5];
			res->val[c*MICVEC_LEN+6] = tmp[6];
			res->val[c*MICVEC_LEN+7] = tmp[7];
#endif
		}
	}


}
