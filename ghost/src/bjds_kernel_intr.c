#include <immintrin.h>
#include "bjds.h"
#include <stdio.h>
#include "ghost_util.h"

void dd_BJDS_kernel_SSE (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * invec, int options)
{
#ifdef SSE_INTR
	ghost_midx_t c,j;
	ghost_mnnz_t offs;
	double *mval = (double *)BJDS(mat)->val;
	double *lval = lhs->val;
	double *rval = invec->val;
	__m128d tmp;
	__m128d val;
	__m128d rhs;

//	printf("in SSE kernel %d %d %d\n",BJDS(mat)->nrowsPadded,lhs->traits->nrows,BJDS(mat)->chunkHeight);
#pragma omp parallel for schedule(runtime) private(j,tmp,val,rhs,offs)
	for (c=0; c<BJDS(mat)->nrowsPadded>>1; c++) 
	{ // loop over chunks
		tmp = _mm_setzero_pd(); // tmp = 0
		offs = BJDS(mat)->chunkStart[c];

		for (j=0; j<(BJDS(mat)->chunkStart[c+1]-BJDS(mat)->chunkStart[c])>>1; j++) 
		{ // loop inside chunk
			val    = _mm_load_pd(&mval[offs]);                      // load values
			rhs    = _mm_loadl_pd(rhs,&rval[(BJDS(mat)->col[offs++])]); // load first 128 bits of RHS
			rhs    = _mm_loadh_pd(rhs,&rval[(BJDS(mat)->col[offs++])]);
			tmp    = _mm_add_pd(tmp,_mm_mul_pd(val,rhs));           // accumulate
		}
		if (options & GHOST_SPMVM_AXPY) {
			_mm_store_pd(&lval[c*2],_mm_add_pd(tmp,_mm_load_pd(&lval[c*2])));
		} else {
			_mm_stream_pd(&lval[c*2],tmp);
		}
	}
#else
	UNUSED(mat);
	UNUSED(lhs);
	UNUSED(invec);
	UNUSED(options);
#endif
}

void dd_BJDS_kernel_AVX(ghost_mat_t *mat, ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions)
{
#ifdef AVX_INTR
	ghost_midx_t c,j;
	ghost_mnnz_t offs;
	double *mval = (double *)BJDS(mat)->val;
	double *lval = res->val;
	double *rval = invec->val;
	__m256d tmp;
	__m256d val;
	__m256d rhs;
	__m128d rhstmp;

#pragma omp parallel for schedule(runtime) private(j,tmp,val,offs,rhs,rhstmp)
	for (c=0; c<BJDS(mat)->nrowsPadded>>2; c++) 
	{ // loop over chunks
		tmp = _mm256_setzero_pd(); // tmp = 0
		offs = BJDS(mat)->chunkStart[c];

		for (j=0; j<(BJDS(mat)->chunkStart[c+1]-BJDS(mat)->chunkStart[c])>>2; j++) 
		{ // loop inside chunk

			val    = _mm256_load_pd(&mval[offs]);                      // load values
			rhstmp = _mm_loadl_pd(rhstmp,&rval[(BJDS(mat)->col[offs++])]); // load first 128 bits of RHS
			rhstmp = _mm_loadh_pd(rhstmp,&rval[(BJDS(mat)->col[offs++])]);
			rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);                  // insert to RHS
			rhstmp = _mm_loadl_pd(rhstmp,&rval[(BJDS(mat)->col[offs++])]); // load second 128 bits of RHS
			rhstmp = _mm_loadh_pd(rhstmp,&rval[(BJDS(mat)->col[offs++])]);
			rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);                  // insert to RHS
			tmp    = _mm256_add_pd(tmp,_mm256_mul_pd(val,rhs));           // accumulate
		}
		if (spmvmOptions & GHOST_SPMVM_AXPY) {
			_mm256_store_pd(&lval[c*BJDS(mat)->chunkHeight],_mm256_add_pd(tmp,_mm256_load_pd(&lval[c*BJDS(mat)->chunkHeight])));
		} else {
			_mm256_stream_pd(&lval[c*BJDS(mat)->chunkHeight],tmp);
		}
	}
#else
	UNUSED(mat);
	UNUSED(res);
	UNUSED(invec);
	UNUSED(spmvmOptions);
#endif
}

void dd_BJDS_kernel_MIC_16(ghost_mat_t *mat, ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions)
{
#ifdef MIC_INTR
	ghost_midx_t c,j;
	ghost_mnnz_t offs;
	double *mval = (double *)BJDS(mat)->val;
	double *lval = res->val;
	double *rval = invec->val;
	__m512d tmp1;
	__m512d tmp2;
	__m512d val;
	__m512d rhs;
	__m512i idx;

#pragma omp parallel for schedule(runtime) private(j,tmp1,tmp2,idx,val,rhs,offs)
	for (c=0; c<BJDS(mat)->nrowsPadded>>4; c++) 
	{ // loop over chunks
		tmp1 = _mm512_setzero_pd(); // tmp1 = 0
		tmp2 = _mm512_setzero_pd(); // tmp2 = 0
		offs = BJDS(mat)->chunkStart[c];

		for (j=0; j<(BJDS(mat)->chunkStart[c+1]-BJDS(mat)->chunkStart[c])>>4; j++) 
		{ // loop inside chunk
			val = _mm512_load_pd(&mval[offs]);
			idx = _mm512_load_epi32(&BJDS(mat)->col[offs]);
			rhs = _mm512_i32logather_pd(idx,rval,8);
			//rhs = _mm512_set1_pd(invec->val[j]);
			tmp1 = _mm512_add_pd(tmp1,_mm512_mul_pd(val,rhs));

			offs += 8;

			val = _mm512_load_pd(&mval[offs]);
			idx = _mm512_permute4f128_epi32(idx,_MM_PERM_BADC);
			rhs = _mm512_i32logather_pd(idx,rval,8);
			//rhs = _mm512_set1_pd(invec->val[j]);
			tmp2 = _mm512_add_pd(tmp2,_mm512_mul_pd(val,rhs));

			offs += 8;
		}
		if (spmvmOptions & GHOST_SPMVM_AXPY) {
		//	_mm512_storenrngo_pd(&res->val[c*BJDS_LEN],_mm512_add_pd(tmp1,_mm512_load_pd(&res->val[c*BJDS_LEN])));
		//	_mm512_storenrngo_pd(&res->val[c*BJDS_LEN+8],_mm512_add_pd(tmp2,_mm512_load_pd(&res->val[c*BJDS_LEN+8])));
			_mm512_store_pd(&lval[c*BJDS(mat)->chunkHeight],_mm512_add_pd(tmp1,_mm512_load_pd(&lval[c*BJDS(mat)->chunkHeight])));
			_mm512_store_pd(&lval[c*BJDS(mat)->chunkHeight+8],_mm512_add_pd(tmp2,_mm512_load_pd(&lval[c*BJDS(mat)->chunkHeight+8])));
		} else {
			_mm512_storenrngo_pd(&lval[c*BJDS(mat)->chunkHeight],tmp1);
			_mm512_storenrngo_pd(&lval[c*BJDS(mat)->chunkHeight+8],tmp2);
		}
	}
#else 
	UNUSED(mat);
	UNUSED(res);
	UNUSED(invec);
	UNUSED(spmvmOptions);
#endif
}
