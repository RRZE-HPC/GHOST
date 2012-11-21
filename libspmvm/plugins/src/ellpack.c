#include "spm_format_ellpack.h"
#include "matricks.h"
#include "ghost_util.h"
#include "kernel.h"

#include <immintrin.h>

#define ELLPACK(mat) ((ELLPACK_TYPE *)(mat->data))

char name[] = "ELLPACK plugin for ghost";
char version[] = "0.1a";
char formatID[] = "ELLPACK";

static mat_nnz_t ELLPACK_nnz(ghost_mat_t *mat);
static mat_idx_t ELLPACK_nrows(ghost_mat_t *mat);
static mat_idx_t ELLPACK_ncols(ghost_mat_t *mat);
static void ELLPACK_printInfo(ghost_mat_t *mat);
static char * ELLPACK_formatName(ghost_mat_t *mat);
static mat_idx_t ELLPACK_rowLen (ghost_mat_t *mat, mat_idx_t i);
static mat_data_t ELLPACK_entry (ghost_mat_t *mat, mat_idx_t i, mat_idx_t j);
static size_t ELLPACK_byteSize (ghost_mat_t *mat);
static void ELLPACK_fromCRS(ghost_mat_t *mat, CR_TYPE *cr, mat_trait_t traits);
static void ELLPACK_fromBin(ghost_mat_t *mat, char *, mat_trait_t traits);
static void ELLPACK_kernel_plain (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);

void init(ghost_mat_t *mat)
{
	DEBUG_LOG(1,"Setting functions for ELLPACK matrix");

	mat->fromBin = &ELLPACK_fromBin;
	mat->printInfo = &ELLPACK_printInfo;
	mat->formatName = &ELLPACK_formatName;
	mat->rowLen     = &ELLPACK_rowLen;
	mat->entry      = &ELLPACK_entry;
	mat->byteSize   = &ELLPACK_byteSize;
	mat->kernel     = &ELLPACK_kernel_plain;
	mat->nnz      = &ELLPACK_nnz;
	mat->nrows    = &ELLPACK_nrows;
	mat->ncols    = &ELLPACK_ncols;
}

static mat_nnz_t ELLPACK_nnz(ghost_mat_t *mat)
{
	return ELLPACK(mat)->nnz;
}
static mat_idx_t ELLPACK_nrows(ghost_mat_t *mat)
{
	return ELLPACK(mat)->nrows;
}
static mat_idx_t ELLPACK_ncols(ghost_mat_t *mat)
{
	UNUSED(mat);
	return 0;
}

static void ELLPACK_printInfo(ghost_mat_t *mat)
{
	return;
}

static char * ELLPACK_formatName(ghost_mat_t *mat)
{
	UNUSED(mat);
	return "ELLPACK";
}

static mat_idx_t ELLPACK_rowLen (ghost_mat_t *mat, mat_idx_t i)
{
	if (mat->trait.flags & GHOST_SPM_SORTED)
		i = mat->rowPerm[i];

	return ELLPACK(mat)->rowLen[i];
}

static mat_data_t ELLPACK_entry (ghost_mat_t *mat, mat_idx_t i, mat_idx_t j)
{
	mat_idx_t e;
	mat_idx_t eInRow;

	if (mat->trait.flags & GHOST_SPM_SORTED)
		i = mat->rowPerm[i];
	if (mat->trait.flags & GHOST_SPM_PERMUTECOLIDX)
		j = mat->rowPerm[j];

	for (e=i, eInRow = 0; eInRow<ELLPACK(mat)->rowLen[i]; e+=ELLPACK(mat)->nrowsPadded, eInRow++) {
		if (ELLPACK(mat)->col[e] == j)
			return ELLPACK(mat)->val[e];
	}
	return 0.;
}

static size_t ELLPACK_byteSize (ghost_mat_t *mat)
{
	return (size_t)((ELLPACK(mat)->nrowsPadded)*sizeof(mat_idx_t) + 
			ELLPACK(mat)->nrowsPadded*ELLPACK(mat)->maxRowLen*(sizeof(mat_idx_t)+sizeof(mat_data_t)));
}

static void ELLPACK_fromBin(ghost_mat_t *mat, char *matrixPath, mat_trait_t traits)
{
	// TODO
	ghost_mat_t *crsMat = SpMVM_initMatrix("CRS");
	mat_trait_t crsTraits = {.format = "CRS",.flags=GHOST_SPM_DEFAULT,NULL};
	crsMat->fromBin(crsMat,matrixPath,crsTraits);
	
	ELLPACK_fromCRS(mat,crsMat->data,traits);
}

static void ELLPACK_fromCRS(ghost_mat_t *mat, CR_TYPE *cr, mat_trait_t trait)
{
	DEBUG_LOG(1,"Creating ELLPACK matrix");
	mat_idx_t i,j,c;
	unsigned int flags = trait.flags;

	mat_idx_t *rowPerm = NULL;
	mat_idx_t *invRowPerm = NULL;

	JD_SORT_TYPE* rowSort;

	mat->data = (ELLPACK_TYPE *)allocateMemory(sizeof(ELLPACK_TYPE),"ELLPACK(mat)");
	mat->data = ELLPACK(mat);
	mat->trait = trait;
//	mat->nrows = cr->nrows;
//	mat->ncols = cr->ncols;
//	mat->nnz = cr->nEnts;
	mat->rowPerm = rowPerm;
	mat->invRowPerm = invRowPerm;
	/**thisMat = (ghost_mat_t)MATRIX_INIT(
	  .trait = trait, 
	  .nrows = cr->nrows, 
	  .ncols = cr->ncols, 
	  .nnz = cr->nEnts,
	  .rowPerm = rowPerm,
	  .invRowPerm = invRowPerm,	   
	  .data = ELLPACK(mat));
	 */
	if (trait.flags & GHOST_SPM_SORTED) {
		rowPerm = (mat_idx_t *)allocateMemory(cr->nrows*sizeof(mat_idx_t),"ELLPACK(mat)->rowPerm");
		invRowPerm = (mat_idx_t *)allocateMemory(cr->nrows*sizeof(mat_idx_t),"ELLPACK(mat)->invRowPerm");

		mat->rowPerm = rowPerm;
		mat->invRowPerm = invRowPerm;
		unsigned int sortBlock = *(unsigned int *)(trait.aux);
		if (sortBlock == 0)
			sortBlock = cr->nrows;

		DEBUG_LOG(1,"Sorting matrix with a sorting block size of %u",sortBlock);

		/* get max number of entries in one row ###########################*/
		rowSort = (JD_SORT_TYPE*) allocateMemory( cr->nrows * sizeof( JD_SORT_TYPE ),
				"rowSort" );

		for (c=0; c<cr->nrows/sortBlock; c++)  
		{
			for( i = c*sortBlock; i < (c+1)*sortBlock; i++ ) 
			{
				rowSort[i].row = i;
				rowSort[i].nEntsInRow = cr->rpt[i+1] - cr->rpt[i];
			} 

			qsort( rowSort+c*sortBlock, sortBlock, sizeof( JD_SORT_TYPE  ), compareNZEPerRow );
		}
		for( i = c*sortBlock; i < cr->nrows; i++ ) 
		{ // remainder
			rowSort[i].row = i;
			rowSort[i].nEntsInRow = cr->rpt[i+1] - cr->rpt[i];
		}

		/* sort within same rowlength with asceding row number #################### */
		/*i=0;
		while(i < cr->nrows) {
			mat_idx_t start = i;

			j = rowSort[start].nEntsInRow;
			while( i<cr->nrows && rowSort[i].nEntsInRow >= j ) 
				++i;

			DEBUG_LOG(1,"sorting over %"PRmatIDX" rows (%"PRmatIDX"): %"PRmatIDX" - %"PRmatIDX,i-start,j, start, i-1);
			qsort( &rowSort[start], i-start, sizeof(JD_SORT_TYPE), compareNZEOrgPos );
		}

		for(i=1; i < cr->nrows; ++i) {
			if( rowSort[i].nEntsInRow == rowSort[i-1].nEntsInRow && rowSort[i].row < rowSort[i-1].row)
				printf("Error in row %"PRmatIDX": descending row number\n",i);
		}*/
		for(i=0; i < cr->nrows; ++i) {
			/* invRowPerm maps an index in the permuted system to the original index,
			 * rowPerm gets the original index and returns the corresponding permuted position.
			 */
			if( rowSort[i].row >= cr->nrows ) DEBUG_LOG(0,"error: invalid row number %"PRmatIDX" in %"PRmatIDX,rowSort[i].row, i); 

			(invRowPerm)[i] = rowSort[i].row;
			(rowPerm)[rowSort[i].row] = i;
		}
	}




	ELLPACK(mat)->nrows = cr->nrows;
	ELLPACK(mat)->nnz = cr->nEnts;
	ELLPACK(mat)->nEnts = 0;
	ELLPACK(mat)->nrowsPadded = pad(ELLPACK(mat)->nrows,ELLPACK_LEN);

	mat_idx_t nChunks = ELLPACK(mat)->nrowsPadded/ELLPACK_LEN;
	ELLPACK(mat)->chunkStart = (mat_nnz_t *)allocateMemory((nChunks+1)*sizeof(mat_nnz_t),"ELLPACK(mat)->chunkStart");
	ELLPACK(mat)->chunkMin = (mat_idx_t *)allocateMemory((nChunks)*sizeof(mat_idx_t),"ELLPACK(mat)->chunkMin");
	ELLPACK(mat)->chunkLen = (mat_idx_t *)allocateMemory((nChunks)*sizeof(mat_idx_t),"ELLPACK(mat)->chunkMin");
	ELLPACK(mat)->rowLen = (mat_idx_t *)allocateMemory((ELLPACK(mat)->nrowsPadded)*sizeof(mat_idx_t),"ELLPACK(mat)->chunkMin");
	ELLPACK(mat)->chunkStart[0] = 0;

	mat_idx_t chunkMin = cr->ncols;
	mat_idx_t chunkLen = 0;
	mat_idx_t curChunk = 1;
	ELLPACK(mat)->nu = 0.;

	for (i=0; i<ELLPACK(mat)->nrowsPadded; i++) {
		if (i<cr->nrows) {
			if (flags & GHOST_SPM_SORTED)
				ELLPACK(mat)->rowLen[i] = rowSort[i].nEntsInRow;
			else
				ELLPACK(mat)->rowLen[i] = cr->rpt[i+1]-cr->rpt[i];
		} else {
			ELLPACK(mat)->rowLen[i] = 0;
		}


		chunkMin = ELLPACK(mat)->rowLen[i]<chunkMin?ELLPACK(mat)->rowLen[i]:chunkMin;
		chunkLen = ELLPACK(mat)->rowLen[i]>chunkLen?ELLPACK(mat)->rowLen[i]:chunkLen;

		if ((i+1)%ELLPACK_LEN == 0) {
			ELLPACK(mat)->nEnts += ELLPACK_LEN*chunkLen;
			ELLPACK(mat)->chunkStart[curChunk] = ELLPACK(mat)->nEnts;
			ELLPACK(mat)->chunkMin[curChunk-1] = chunkMin;
			ELLPACK(mat)->chunkLen[curChunk-1] = chunkLen;

			ELLPACK(mat)->nu += (double)chunkMin/chunkLen;

			chunkMin = cr->ncols;
			chunkLen = 0;
			curChunk++;
		}
	}
	ELLPACK(mat)->nu /= (double)nChunks;

	ELLPACK(mat)->val = (mat_data_t *)allocateMemory(sizeof(mat_data_t)*ELLPACK(mat)->nEnts,"ELLPACK(mat)->val");
	ELLPACK(mat)->col = (mat_idx_t *)allocateMemory(sizeof(mat_idx_t)*ELLPACK(mat)->nEnts,"ELLPACK(mat)->col");

#pragma omp parallel for schedule(runtime) private(j,i)
	for (c=0; c<ELLPACK(mat)->nrowsPadded/ELLPACK_LEN; c++) 
	{ // loop over chunks

		for (j=0; j<(ELLPACK(mat)->chunkStart[c+1]-ELLPACK(mat)->chunkStart[c])/ELLPACK_LEN; j++)
		{
			for (i=0; i<ELLPACK_LEN; i++)
			{
				ELLPACK(mat)->val[ELLPACK(mat)->chunkStart[c]+j*ELLPACK_LEN+i] = 0.;
				ELLPACK(mat)->col[ELLPACK(mat)->chunkStart[c]+j*ELLPACK_LEN+i] = 0;
			}
		}
	}



	for (c=0; c<nChunks; c++) {

		for (j=0; j<ELLPACK(mat)->chunkLen[c]; j++) {

			for (i=0; i<ELLPACK_LEN; i++) {
				mat_idx_t row = c*ELLPACK_LEN+i;

				if (j<ELLPACK(mat)->rowLen[row]) {
					if (flags & GHOST_SPM_SORTED) {
						ELLPACK(mat)->val[ELLPACK(mat)->chunkStart[c]+j*ELLPACK_LEN+i] = cr->val[cr->rpt[(invRowPerm)[row]]+j];
						if (flags & GHOST_SPM_PERMUTECOLIDX)
							ELLPACK(mat)->col[ELLPACK(mat)->chunkStart[c]+j*ELLPACK_LEN+i] = (rowPerm)[cr->col[cr->rpt[(invRowPerm)[row]]+j]];
						else
							ELLPACK(mat)->col[ELLPACK(mat)->chunkStart[c]+j*ELLPACK_LEN+i] = cr->col[cr->rpt[(invRowPerm)[row]]+j];
					} else {
						ELLPACK(mat)->val[ELLPACK(mat)->chunkStart[c]+j*ELLPACK_LEN+i] = cr->val[cr->rpt[row]+j];
						ELLPACK(mat)->col[ELLPACK(mat)->chunkStart[c]+j*ELLPACK_LEN+i] = cr->col[cr->rpt[row]+j];
					}

				} else {
					ELLPACK(mat)->val[ELLPACK(mat)->chunkStart[c]+j*ELLPACK_LEN+i] = 0.0;
					ELLPACK(mat)->col[ELLPACK(mat)->chunkStart[c]+j*ELLPACK_LEN+i] = 0;
				}
				//	printf("%f ",ELLPACK(mat)->val[ELLPACK(mat)->chunkStart[c]+j*ELLPACK_LEN+i]);


			}
		}
	}


	DEBUG_LOG(1,"Successfully created ELLPACK");



}

static void ELLPACK_kernel_plain (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
	//	sse_kernel_0_intr(lhs, ELLPACK(mat), rhs, options);	
	mat_idx_t c,j,i;
	mat_data_t tmp[ELLPACK_LEN]; 

#pragma omp parallel for schedule(runtime) private(j,tmp,i)
	for (c=0; c<ELLPACK(mat)->nrowsPadded/ELLPACK_LEN; c++) 
	{ // loop over chunks
		for (i=0; i<ELLPACK_LEN; i++)
		{
			tmp[i] = 0;
		}

		for (j=0; j<(ELLPACK(mat)->chunkStart[c+1]-ELLPACK(mat)->chunkStart[c])/ELLPACK_LEN; j++) 
		{ // loop inside chunk
			for (i=0; i<ELLPACK_LEN; i++)
			{
				tmp[i] += ELLPACK(mat)->val[ELLPACK(mat)->chunkStart[c]+j*ELLPACK_LEN+i] * rhs->val[ELLPACK(mat)->col[ELLPACK(mat)->chunkStart[c]+j*ELLPACK_LEN+i]];
			}

		}
		for (i=0; i<ELLPACK_LEN; i++)
		{
			if (options & GHOST_OPTION_AXPY)
				lhs->val[c*ELLPACK_LEN+i] += tmp[i];
			else
				lhs->val[c*ELLPACK_LEN+i] = tmp[i];

		}
	}
}

#ifdef SSE
static void ELLPACK_kernel_SSE (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * invec, int options)
{
	mat_idx_t c,j;
	mat_nnz_t offs;
	__m128d tmp;
	__m128d val;
	__m128d rhs;

#pragma omp parallel for schedule(runtime) private(j,tmp,val,rhs,offs)
	for (c=0; c<ELLPACK(mat)->nrowsPadded>>1; c++) 
	{ // loop over chunks
		tmp = _mm_setzero_pd(); // tmp = 0
		offs = ELLPACK(mat)->chunkStart[c];

		for (j=0; j<(ELLPACK(mat)->chunkStart[c+1]-ELLPACK(mat)->chunkStart[c])>>1; j++) 
		{ // loop inside chunk
			val    = _mm_load_pd(&ELLPACK(mat)->val[offs]);                      // load values
			rhs    = _mm_loadl_pd(rhs,&invec->val[(ELLPACK(mat)->col[offs++])]); // load first 128 bits of RHS
			rhs    = _mm_loadh_pd(rhs,&invec->val[(ELLPACK(mat)->col[offs++])]);
			tmp    = _mm_add_pd(tmp,_mm_mul_pd(val,rhs));           // accumulate
		}
		if (options & GHOST_OPTION_AXPY) {
			_mm_store_pd(&lhs->val[c*ELLPACK_LEN],_mm_add_pd(tmp,_mm_load_pd(&lhs->val[c*ELLPACK_LEN])));
		} else {
			_mm_stream_pd(&lhs->val[c*ELLPACK_LEN],tmp);
		}
	}


}
#endif

#ifdef AVX
static void ELLPACK_kernel_AVX(ghost_mat_t *mat, ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions)
{
	mat_idx_t c,j;
	mat_nnz_t offs;
	__m256d tmp;
	__m256d val;
	__m256d rhs;
	__m128d rhstmp;

#pragma omp parallel for schedule(runtime) private(j,tmp,val,rhs,offs,rhstmp)
	for (c=0; c<ELLPACK(mat)->nrowsPadded>>2; c++) 
	{ // loop over chunks
		tmp = _mm256_setzero_pd(); // tmp = 0
		offs = ELLPACK(mat)->chunkStart[c];

		for (j=0; j<(ELLPACK(mat)->chunkStart[c+1]-ELLPACK(mat)->chunkStart[c])>>2; j++) 
		{ // loop inside chunk

			val    = _mm256_load_pd(&ELLPACK(mat)->val[offs]);                      // load values
			rhstmp = _mm_loadl_pd(rhstmp,&invec->val[(ELLPACK(mat)->col[offs++])]); // load first 128 bits of RHS
			rhstmp = _mm_loadh_pd(rhstmp,&invec->val[(ELLPACK(mat)->col[offs++])]);
			rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);                  // insert to RHS
			rhstmp = _mm_loadl_pd(rhstmp,&invec->val[(ELLPACK(mat)->col[offs++])]); // load second 128 bits of RHS
			rhstmp = _mm_loadh_pd(rhstmp,&invec->val[(ELLPACK(mat)->col[offs++])]);
			rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);                  // insert to RHS
			tmp    = _mm256_add_pd(tmp,_mm256_mul_pd(val,rhs));           // accumulate
		}
		if (spmvmOptions & GHOST_OPTION_AXPY) {
			_mm256_store_pd(&res->val[c*ELLPACK_LEN],_mm256_add_pd(tmp,_mm256_load_pd(&res->val[c*ELLPACK_LEN])));
		} else {
			_mm256_stream_pd(&res->val[c*ELLPACK_LEN],tmp);
		}
	}
}
#endif

#ifdef MIC
static void ELLPACK_kernel_MIC(ghost_mat_t *mat, ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions)
{
	mat_idx_t c,j;
	mat_nnz_t offs;
	__m512d tmp;
	__m512d val;
	__m512d rhs;
	__m512i idx;

#pragma omp parallel for schedule(runtime) private(j,tmp,val,rhs,idx,offs)
	for (c=0; c<ELLPACK(mat)->nrowsPadded>>3; c++) 
	{ // loop over chunks
		tmp = _mm512_setzero_pd(); // tmp = 0
		//		int offset = ELLPACK(mat)->chunkStart[c];
		offs = ELLPACK(mat)->chunkStart[c];

		for (j=0; j<(ELLPACK(mat)->chunkStart[c+1]-ELLPACK(mat)->chunkStart[c])>>3; j+=2) 
		{ // loop inside chunk
			val = _mm512_load_pd(&ELLPACK(mat)->val[offs]);
			idx = _mm512_load_epi32(&ELLPACK(mat)->col[offs]);
			rhs = _mm512_i32logather_pd(idx,invec->val,8);
			tmp = _mm512_add_pd(tmp,_mm512_mul_pd(val,rhs));

			offs += 8;

			val = _mm512_load_pd(&ELLPACK(mat)->val[offs]);
			idx = _mm512_permute4f128_epi32(idx,_MM_PERM_BADC);
			rhs = _mm512_i32logather_pd(idx,invec->val,8);
			tmp = _mm512_add_pd(tmp,_mm512_mul_pd(val,rhs));

			offs += 8;
		}
		if (spmvmOptions & GHOST_OPTION_AXPY) {
			_mm512_storenrngo_pd(&res->val[c*ELLPACK_LEN],_mm512_add_pd(tmp,_mm512_load_pd(&res->val[c*ELLPACK_LEN])));
		} else {
			_mm512_storenrngo_pd(&res->val[c*ELLPACK_LEN],tmp);
		}
	}
}

static void ELLPACK_kernel_MIC_16(ghost_mat_t *mat, ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions)
{
	mat_idx_t c,j;
	mat_nnz_t offs;
	__m512d tmp1;
	__m512d tmp2;
	__m512d val;
	__m512d rhs;
	__m512i idx;

#pragma omp parallel for schedule(runtime) private(j,tmp1,tmp2,val,rhs,idx,offs)
	for (c=0; c<ELLPACK(mat)->nrowsPadded>>4; c++) 
	{ // loop over chunks
		tmp1 = _mm512_setzero_pd(); // tmp1 = 0
		tmp2 = _mm512_setzero_pd(); // tmp2 = 0
		offs = ELLPACK(mat)->chunkStart[c];

		for (j=0; j<(ELLPACK(mat)->chunkStart[c+1]-ELLPACK(mat)->chunkStart[c])>>4; j++) 
		{ // loop inside chunk
			val = _mm512_load_pd(&ELLPACK(mat)->val[offs]);
			idx = _mm512_load_epi32(&ELLPACK(mat)->col[offs]);
			rhs = _mm512_i32logather_pd(idx,invec->val,8);
			tmp1 = _mm512_add_pd(tmp1,_mm512_mul_pd(val,rhs));

			offs += 8;

			val = _mm512_load_pd(&ELLPACK(mat)->val[offs]);
			idx = _mm512_permute4f128_epi32(idx,_MM_PERM_BADC);
			rhs = _mm512_i32logather_pd(idx,invec->val,8);
			tmp2 = _mm512_add_pd(tmp2,_mm512_mul_pd(val,rhs));

			offs += 8;
		}
		if (spmvmOptions & GHOST_OPTION_AXPY) {
			_mm512_storenrngo_pd(&res->val[c*ELLPACK_LEN],_mm512_add_pd(tmp1,_mm512_load_pd(&res->val[c*ELLPACK_LEN])));
			_mm512_storenrngo_pd(&res->val[c*ELLPACK_LEN+8],_mm512_add_pd(tmp2,_mm512_load_pd(&res->val[c*ELLPACK_LEN+8])));
		} else {
			_mm512_storenrngo_pd(&res->val[c*ELLPACK_LEN],tmp1);
			_mm512_storenrngo_pd(&res->val[c*ELLPACK_LEN+8],tmp2);
		}
	}
}
#endif
