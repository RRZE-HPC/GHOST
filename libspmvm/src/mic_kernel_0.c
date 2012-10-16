#include "kernel_helper.h"
#include "kernel.h"
#include <stdio.h>
#include <xmmintrin.h>
#include <immintrin.h>

#ifdef LIKWID
#include <likwid.h>
#endif
void mic_kernel_0(VECTOR_TYPE* res, BJDS_TYPE* mv, VECTOR_TYPE* invec, int spmvmOptions)
{
	int c,j,i;
	mat_data_t tmp[BJDS_LEN]; 

#pragma omp parallel for schedule(runtime) private(j,tmp,i)
	for (c=0; c<mv->nRowsPadded/BJDS_LEN; c++) 
	{ // loop over chunks
		for (i=0; i<BJDS_LEN; i++)
		{
			tmp[i] = 0;
		}

		for (j=0; j<(mv->chunkStart[c+1]-mv->chunkStart[c])/BJDS_LEN; j++) 
		{ // loop inside chunk
			for (i=0; i<BJDS_LEN; i++)
			{
				tmp[i] += mv->val[mv->chunkStart[c]+j*BJDS_LEN+i] * invec->val[mv->col[mv->chunkStart[c]+j*BJDS_LEN+i]];

			}
			if (spmvmOptions & SPMVM_OPTION_AXPY) { 
				for (i=0; i<BJDS_LEN; i++)
				{
					res->val[c*BJDS_LEN+i] += tmp[i];
				}

			} else {
				for (i=0; i<BJDS_LEN; i++)
				{
					res->val[c*BJDS_LEN+i] = tmp[i];
				}
			}
		}
	}
}

void mic_kernel_0_unr(VECTOR_TYPE* res, BJDS_TYPE* mv, VECTOR_TYPE* invec, int spmvmOptions)
{
	int c,j;
	mat_data_t tmp[BJDS_LEN]; 

#pragma omp parallel for schedule(runtime) private(j,tmp)
	for (c=0; c<mv->nRowsPadded/BJDS_LEN; c++) 
	{ // loop over chunks
		tmp[0] = 0.;
		tmp[1] = 0.;
		tmp[2] = 0.;
		tmp[3] = 0.;
#if BJDS_LEN > 4
		tmp[4] = 0.;
		tmp[5] = 0.;
		tmp[6] = 0.;
		tmp[7] = 0.;
#endif

		for (j=0; j<(mv->chunkStart[c+1]-mv->chunkStart[c])/BJDS_LEN; j++) 
		{ // loop inside chunk
			tmp[0] += mv->val[mv->chunkStart[c]+j*BJDS_LEN  ] * invec->val[mv->col[mv->chunkStart[c]+j*BJDS_LEN  ]];
			tmp[1] += mv->val[mv->chunkStart[c]+j*BJDS_LEN+1] * invec->val[mv->col[mv->chunkStart[c]+j*BJDS_LEN+1]];
			tmp[2] += mv->val[mv->chunkStart[c]+j*BJDS_LEN+2] * invec->val[mv->col[mv->chunkStart[c]+j*BJDS_LEN+2]];
			tmp[3] += mv->val[mv->chunkStart[c]+j*BJDS_LEN+3] * invec->val[mv->col[mv->chunkStart[c]+j*BJDS_LEN+3]];
#if BJDS_LEN > 4
			tmp[4] += mv->val[mv->chunkStart[c]+j*BJDS_LEN+4] * invec->val[mv->col[mv->chunkStart[c]+j*BJDS_LEN+4]];
			tmp[5] += mv->val[mv->chunkStart[c]+j*BJDS_LEN+5] * invec->val[mv->col[mv->chunkStart[c]+j*BJDS_LEN+5]];
			tmp[6] += mv->val[mv->chunkStart[c]+j*BJDS_LEN+6] * invec->val[mv->col[mv->chunkStart[c]+j*BJDS_LEN+6]];
			tmp[7] += mv->val[mv->chunkStart[c]+j*BJDS_LEN+7] * invec->val[mv->col[mv->chunkStart[c]+j*BJDS_LEN+7]];
#endif

		}
		if (spmvmOptions & SPMVM_OPTION_AXPY) { 
			res->val[c*BJDS_LEN  ] += tmp[0];
			res->val[c*BJDS_LEN+1] += tmp[1];
			res->val[c*BJDS_LEN+2] += tmp[2];
			res->val[c*BJDS_LEN+3] += tmp[3];
#if BJDS_LEN > 4
			res->val[c*BJDS_LEN+4] += tmp[4];
			res->val[c*BJDS_LEN+5] += tmp[5];
			res->val[c*BJDS_LEN+6] += tmp[6];
			res->val[c*BJDS_LEN+7] += tmp[7];
#endif
		} else {
			res->val[c*BJDS_LEN  ] = tmp[0];
			res->val[c*BJDS_LEN+1] = tmp[1];
			res->val[c*BJDS_LEN+2] = tmp[2];
			res->val[c*BJDS_LEN+3] = tmp[3];
#if BJDS_LEN > 4
			res->val[c*BJDS_LEN+4] = tmp[4];
			res->val[c*BJDS_LEN+5] = tmp[5];
			res->val[c*BJDS_LEN+6] = tmp[6];
			res->val[c*BJDS_LEN+7] = tmp[7];
#endif
		}
	}
}

void mic_kernel_0_intr(VECTOR_TYPE* res, BJDS_TYPE* mv, VECTOR_TYPE* invec, int spmvmOptions)
{
	int c,j,offs;
	__m512d tmp;
	__m512d val;
	__m512d rhs;
	__m512i idx; // TODO lower und upper index, shuffle ueberlappen

	// TODO relaxed store
#pragma omp parallel for schedule(runtime) private(j,tmp,val,rhs,idx,offs)
	for (c=0; c<mv->nRowsPadded>>3; c++) 
	{ // loop over chunks
		tmp = _mm512_setzero_pd(); // tmp = 0
		//		int offset = mv->chunkStart[c];
		offs = mv->chunkStart[c];

		for (j=0; j<(mv->chunkStart[c+1]-mv->chunkStart[c])>>3; j+=2) 
		{ // loop inside chunk
			val = _mm512_load_pd(&mv->val[offs]);
			idx = _mm512_load_epi32(&mv->col[offs]);
			rhs = _mm512_i32logather_pd(idx,invec->val,8);
			tmp = _mm512_add_pd(tmp,_mm512_mul_pd(val,rhs));

			offs += 8;

			val = _mm512_load_pd(&mv->val[offs]);
			idx = _mm512_permute4f128_epi32(idx,_MM_PERM_BADC);
			rhs = _mm512_i32logather_pd(idx,invec->val,8);
			tmp = _mm512_add_pd(tmp,_mm512_mul_pd(val,rhs));

			offs += 8;
		}
		if (spmvmOptions & SPMVM_OPTION_AXPY) {
			_mm512_extstore_pd(&res->val[c*BJDS_LEN],_mm512_add_pd(tmp,_mm512_load_pd(&res->val[c*BJDS_LEN])),_MM_DOWNCONV_PD_NONE,_MM_HINT_NT);
		} else {
			_mm512_extstore_pd(&res->val[c*BJDS_LEN],tmp,_MM_DOWNCONV_PD_NONE,_MM_HINT_NT);
		}
	}
}
