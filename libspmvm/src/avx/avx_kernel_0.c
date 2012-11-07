#include "kernel.h"
#include "spmvm_util.h"
#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>

void avx_kernel_0_intr(VECTOR_TYPE* res, BJDS_TYPE* bjds, VECTOR_TYPE* invec, int spmvmOptions)
{
	int c,j,offs;
	__m256d tmp;
	__m256d val;
	__m256d rhs;
	__m128d rhstmp;

#pragma omp parallel for schedule(runtime) private(j,tmp,val,rhs,offs,rhstmp)
	for (c=0; c<bjds->nRowsPadded>>2; c++) 
	{ // loop over chunks
		tmp = _mm256_setzero_pd(); // tmp = 0
		offs = bjds->chunkStart[c];

		for (j=0; j<(bjds->chunkStart[c+1]-bjds->chunkStart[c])>>2; j++) 
		{ // loop inside chunk

			val    = _mm256_load_pd(&bjds->val[offs]);                      // load values
			rhstmp = _mm_loadl_pd(rhstmp,&invec->val[(bjds->col[offs++])]); // load first 128 bits of RHS
			rhstmp = _mm_loadh_pd(rhstmp,&invec->val[(bjds->col[offs++])]);
			rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);                  // insert to RHS
			rhstmp = _mm_loadl_pd(rhstmp,&invec->val[(bjds->col[offs++])]); // load second 128 bits of RHS
			rhstmp = _mm_loadh_pd(rhstmp,&invec->val[(bjds->col[offs++])]);
			rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);                  // insert to RHS
			tmp    = _mm256_add_pd(tmp,_mm256_mul_pd(val,rhs));           // accumulate
		}
		if (spmvmOptions & SPMVM_OPTION_AXPY) {
			_mm256_store_pd(&res->val[c*BJDS_LEN],_mm256_add_pd(tmp,_mm256_load_pd(&res->val[c*BJDS_LEN])));
		} else {
			_mm256_stream_pd(&res->val[c*BJDS_LEN],tmp);
		}
	}
}

void avx_kernel_0_intr_rem(VECTOR_TYPE* res, BJDS_TYPE* bjds, VECTOR_TYPE* invec, int spmvmOptions)
{
	int c,j,offs;
	__m256d tmp;
	__m256d val;
	__m256d rhs;
	__m128d rhstmp;

#pragma omp parallel for schedule(runtime) private(j,tmp,val,rhs,offs,rhstmp)
	for (c=0; c<bjds->nRowsPadded>>2; c++) 
	{ // loop over chunks
		tmp = _mm256_setzero_pd(); // tmp = 0
		offs = bjds->chunkStart[c];

		for (j=0; j<bjds->chunkMin[c]; j++) 
		{ // loop inside chunk
			val    = _mm256_load_pd(&bjds->val[offs]);                      // load values
			rhstmp = _mm_loadl_pd(rhstmp,&invec->val[(bjds->col[offs++])]); // load first 128 bits of RHS
			rhstmp = _mm_loadh_pd(rhstmp,&invec->val[(bjds->col[offs++])]);
			rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);                  // insert to RHS
			rhstmp = _mm_loadl_pd(rhstmp,&invec->val[(bjds->col[offs++])]); // load second 128 bits of RHS
			rhstmp = _mm_loadh_pd(rhstmp,&invec->val[(bjds->col[offs++])]);
			rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);                  // insert to RHS
			tmp    = _mm256_add_pd(tmp,_mm256_mul_pd(val,rhs));           // accumulate
		}
			/*printf("rem 1 %d..%d\n",bjds->chunkMin[c],bjds->rowLen[c*BJDS_LEN]);
			printf("rem 2 %d..%d\n",bjds->chunkMin[c],bjds->rowLen[c*BJDS_LEN+1]);
			printf("rem 3 %d..%d\n",bjds->chunkMin[c],bjds->rowLen[c*BJDS_LEN+2]);
		printf("rem 4 %d..%d\n",bjds->chunkMin[c],bjds->rowLen[c*BJDS_LEN+3]);*/
		for (j=bjds->chunkMin[c]; j<bjds->rowLen[c*BJDS_LEN]; j++)
		{
			res->val[c*BJDS_LEN] += bjds->val[offs] * invec->val[bjds->col[offs++]];
		}
		for (j=bjds->chunkMin[c]; j<bjds->rowLen[c*BJDS_LEN+1]; j++)
		{
			res->val[c*BJDS_LEN+1] += bjds->val[offs] * invec->val[bjds->col[offs++]];
		}
		for (j=bjds->chunkMin[c]; j<bjds->rowLen[c*BJDS_LEN+2]; j++)
		{
			res->val[c*BJDS_LEN+2] += bjds->val[offs] * invec->val[bjds->col[offs++]];
		}
		for (j=bjds->chunkMin[c]; j<bjds->rowLen[c*BJDS_LEN+3]; j++)
		{
			res->val[c*BJDS_LEN+3] += bjds->val[offs] * invec->val[bjds->col[offs++]];
		}


		if (spmvmOptions & SPMVM_OPTION_AXPY) {
			_mm256_store_pd(&res->val[c*BJDS_LEN],_mm256_add_pd(tmp,_mm256_load_pd(&res->val[c*BJDS_LEN])));
		} else {
			_mm256_stream_pd(&res->val[c*BJDS_LEN],tmp);
		}
	}
}

void avx_kernel_0_intr_rem_if(VECTOR_TYPE* res, BJDS_TYPE* bjds, VECTOR_TYPE* invec, int spmvmOptions)
{
	int c,j,offs;
	__m256d tmp;
	__m256d val;
	__m256d rhs;
	__m128d rhstmp;

#pragma omp parallel for schedule(runtime) private(j,tmp,val,rhs,offs,rhstmp)
	for (c=0; c<bjds->nRowsPadded>>2; c++) 
	{ // loop over chunks
		tmp = _mm256_setzero_pd(); // tmp = 0
		offs = bjds->chunkStart[c];

		for (j=0; j<bjds->chunkMin[c]; j++) 
		{ // loop inside chunk
			val    = _mm256_load_pd(&bjds->val[offs]);                      // load values
			rhstmp = _mm_loadl_pd(rhstmp,&invec->val[(bjds->col[offs++])]); // load first 128 bits of RHS
			rhstmp = _mm_loadh_pd(rhstmp,&invec->val[(bjds->col[offs++])]);
			rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);                  // insert to RHS
			rhstmp = _mm_loadl_pd(rhstmp,&invec->val[(bjds->col[offs++])]); // load second 128 bits of RHS
			rhstmp = _mm_loadh_pd(rhstmp,&invec->val[(bjds->col[offs++])]);
			rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);                  // insert to RHS
			tmp    = _mm256_add_pd(tmp,_mm256_mul_pd(val,rhs));           // accumulate
		}

		for (j=bjds->chunkMin[c]; j<bjds->chunkLen[c]; j++)
		{
			if (j<bjds->rowLen[c*BJDS_LEN])
				res->val[c*BJDS_LEN] += bjds->val[offs] * invec->val[bjds->col[offs++]];
			if (j<bjds->rowLen[c*BJDS_LEN+1])
				res->val[c*BJDS_LEN+1] += bjds->val[offs] * invec->val[bjds->col[offs++]];
			if (j<bjds->rowLen[c*BJDS_LEN+2])
				res->val[c*BJDS_LEN+2] += bjds->val[offs] * invec->val[bjds->col[offs++]];
			if (j<bjds->rowLen[c*BJDS_LEN+3])
				res->val[c*BJDS_LEN+3] += bjds->val[offs] * invec->val[bjds->col[offs++]];
		}


		if (spmvmOptions & SPMVM_OPTION_AXPY) {
			_mm256_store_pd(&res->val[c*BJDS_LEN],_mm256_add_pd(tmp,_mm256_load_pd(&res->val[c*BJDS_LEN])));
		} else {
			_mm256_stream_pd(&res->val[c*BJDS_LEN],tmp);
		}
	}
}
