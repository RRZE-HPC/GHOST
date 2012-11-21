#include "spm_format_bjds.h"
#include "matricks.h"
#include "ghost_util.h"
#include "kernel.h"

#include <immintrin.h>

#define BJDS(mat) ((BJDS_TYPE *)(mat->data))

char name[] = "BJDS plugin for ghost";
char version[] = "0.1a";
char formatID[] = "BJDS";

static mat_nnz_t BJDS_nnz(ghost_mat_t *mat);
static mat_idx_t BJDS_nrows(ghost_mat_t *mat);
static mat_idx_t BJDS_ncols(ghost_mat_t *mat);
static void BJDS_printInfo(ghost_mat_t *mat);
static char * BJDS_formatName(ghost_mat_t *mat);
static mat_idx_t BJDS_rowLen (ghost_mat_t *mat, mat_idx_t i);
static mat_data_t BJDS_entry (ghost_mat_t *mat, mat_idx_t i, mat_idx_t j);
static size_t BJDS_byteSize (ghost_mat_t *mat);
static void BJDS_fromCRS(ghost_mat_t *mat, CR_TYPE *cr, mat_trait_t traits);
static void BJDS_fromBin(ghost_mat_t *mat, char *, mat_trait_t traits);
static void BJDS_kernel_plain (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
#ifdef SSE
static void BJDS_kernel_SSE (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
#endif
#ifdef AVX
static void BJDS_kernel_AVX (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
#endif
#ifdef MIC
static void BJDS_kernel_MIC (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
static void BJDS_kernel_MIC_16 (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
#endif

//static ghost_mat_t *thisMat;
//static BJDS_TYPE *BJDS(mat);

void init(ghost_mat_t *mat)
{
	DEBUG_LOG(1,"Setting functions for TBJDS matrix");

	mat->fromBin = &BJDS_fromBin;
	mat->printInfo = &BJDS_printInfo;
	mat->formatName = &BJDS_formatName;
	mat->rowLen     = &BJDS_rowLen;
	mat->entry      = &BJDS_entry;
	mat->byteSize   = &BJDS_byteSize;
	mat->kernel     = &BJDS_kernel_plain;
#ifdef SSE
	mat->kernel   = &BJDS_kernel_SSE;
#endif
#ifdef AVX
	mat->kernel   = &BJDS_kernel_AVX;
#endif
#ifdef MIC
	mat->kernel   = &BJDS_kernel_MIC_16;
	UNUSED(&BJDS_kernel_MIC);
#endif
	mat->nnz      = &BJDS_nnz;
	mat->nrows    = &BJDS_nrows;
	mat->ncols    = &BJDS_ncols;
}

static mat_nnz_t BJDS_nnz(ghost_mat_t *mat)
{
	return BJDS(mat)->nnz;
}
static mat_idx_t BJDS_nrows(ghost_mat_t *mat)
{
	return BJDS(mat)->nrows;
}
static mat_idx_t BJDS_ncols(ghost_mat_t *mat)
{
	UNUSED(mat);
	return 0;
}

static void BJDS_printInfo(ghost_mat_t *mat)
{
	SpMVM_printLine("Vector block size",NULL,"%d",BJDS_LEN);
	SpMVM_printLine("Row length oscillation nu",NULL,"%f",BJDS(mat)->nu);
	if (mat->trait.flags & GHOST_SPM_SORTED) {
		SpMVM_printLine("Sorted",NULL,"yes");
		SpMVM_printLine("Sort block size",NULL,"%u",*(unsigned int *)(mat->trait.aux));
		SpMVM_printLine("Permuted columns",NULL,"%s",mat->trait.flags&GHOST_SPM_PERMUTECOLIDX?"yes":"no");
	} else {
		SpMVM_printLine("Sorted",NULL,"no");
	}
}

static char * BJDS_formatName(ghost_mat_t *mat)
{
	UNUSED(mat);
	return "BJDS";
}

static mat_idx_t BJDS_rowLen (ghost_mat_t *mat, mat_idx_t i)
{
	if (mat->trait.flags & GHOST_SPM_SORTED)
		i = mat->rowPerm[i];

	return BJDS(mat)->rowLen[i];
}

static mat_data_t BJDS_entry (ghost_mat_t *mat, mat_idx_t i, mat_idx_t j)
{
	mat_idx_t e;

	if (mat->trait.flags & GHOST_SPM_SORTED)
		i = mat->rowPerm[i];
	if (mat->trait.flags & GHOST_SPM_PERMUTECOLIDX)
		j = mat->rowPerm[j];

	for (e=BJDS(mat)->chunkStart[i/BJDS_LEN]+i%BJDS_LEN; 
			e<BJDS(mat)->chunkStart[i/BJDS_LEN+1]; 
			e+=BJDS_LEN) {
		if (BJDS(mat)->col[e] == j)
			return BJDS(mat)->val[e];
	}
	return 0.;
}

static size_t BJDS_byteSize (ghost_mat_t *mat)
{
	return (size_t)((BJDS(mat)->nrowsPadded/BJDS_LEN)*sizeof(mat_nnz_t) + 
			BJDS(mat)->nEnts*(sizeof(mat_idx_t)+sizeof(mat_data_t)));
}

static void BJDS_fromBin(ghost_mat_t *mat, char *matrixPath, mat_trait_t traits)
{
	// TODO
	ghost_mat_t *crsMat = SpMVM_initMatrix("CRS");
	mat_trait_t crsTraits = {.format = "CRS",.flags=GHOST_SPM_DEFAULT,NULL};
	crsMat->fromBin(crsMat,matrixPath,crsTraits);
	
	BJDS_fromCRS(mat,crsMat->data,traits);
}

static void BJDS_fromCRS(ghost_mat_t *mat, CR_TYPE *cr, mat_trait_t trait)
{
	DEBUG_LOG(1,"Creating BJDS matrix");
	mat_idx_t i,j,c;
	unsigned int flags = trait.flags;

	mat_idx_t *rowPerm = NULL;
	mat_idx_t *invRowPerm = NULL;

	JD_SORT_TYPE* rowSort;

	mat->data = (BJDS_TYPE *)allocateMemory(sizeof(BJDS_TYPE),"BJDS(mat)");
	mat->data = BJDS(mat);
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
	  .data = BJDS(mat));
	 */
	if (trait.flags & GHOST_SPM_SORTED) {
		rowPerm = (mat_idx_t *)allocateMemory(cr->nrows*sizeof(mat_idx_t),"BJDS(mat)->rowPerm");
		invRowPerm = (mat_idx_t *)allocateMemory(cr->nrows*sizeof(mat_idx_t),"BJDS(mat)->invRowPerm");

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




	BJDS(mat)->nrows = cr->nrows;
	BJDS(mat)->nnz = cr->nEnts;
	BJDS(mat)->nEnts = 0;
	BJDS(mat)->nrowsPadded = pad(BJDS(mat)->nrows,BJDS_LEN);

	mat_idx_t nChunks = BJDS(mat)->nrowsPadded/BJDS_LEN;
	BJDS(mat)->chunkStart = (mat_nnz_t *)allocateMemory((nChunks+1)*sizeof(mat_nnz_t),"BJDS(mat)->chunkStart");
	BJDS(mat)->chunkMin = (mat_idx_t *)allocateMemory((nChunks)*sizeof(mat_idx_t),"BJDS(mat)->chunkMin");
	BJDS(mat)->chunkLen = (mat_idx_t *)allocateMemory((nChunks)*sizeof(mat_idx_t),"BJDS(mat)->chunkMin");
	BJDS(mat)->rowLen = (mat_idx_t *)allocateMemory((BJDS(mat)->nrowsPadded)*sizeof(mat_idx_t),"BJDS(mat)->chunkMin");
	BJDS(mat)->chunkStart[0] = 0;

	mat_idx_t chunkMin = cr->ncols;
	mat_idx_t chunkLen = 0;
	mat_idx_t curChunk = 1;
	BJDS(mat)->nu = 0.;

	for (i=0; i<BJDS(mat)->nrowsPadded; i++) {
		if (i<cr->nrows) {
			if (flags & GHOST_SPM_SORTED)
				BJDS(mat)->rowLen[i] = rowSort[i].nEntsInRow;
			else
				BJDS(mat)->rowLen[i] = cr->rpt[i+1]-cr->rpt[i];
		} else {
			BJDS(mat)->rowLen[i] = 0;
		}


		chunkMin = BJDS(mat)->rowLen[i]<chunkMin?BJDS(mat)->rowLen[i]:chunkMin;
		chunkLen = BJDS(mat)->rowLen[i]>chunkLen?BJDS(mat)->rowLen[i]:chunkLen;

		if ((i+1)%BJDS_LEN == 0) {
			BJDS(mat)->nEnts += BJDS_LEN*chunkLen;
			BJDS(mat)->chunkStart[curChunk] = BJDS(mat)->nEnts;
			BJDS(mat)->chunkMin[curChunk-1] = chunkMin;
			BJDS(mat)->chunkLen[curChunk-1] = chunkLen;

			BJDS(mat)->nu += (double)chunkMin/chunkLen;

			chunkMin = cr->ncols;
			chunkLen = 0;
			curChunk++;
		}
	}
	BJDS(mat)->nu /= (double)nChunks;

	BJDS(mat)->val = (mat_data_t *)allocateMemory(sizeof(mat_data_t)*BJDS(mat)->nEnts,"BJDS(mat)->val");
	BJDS(mat)->col = (mat_idx_t *)allocateMemory(sizeof(mat_idx_t)*BJDS(mat)->nEnts,"BJDS(mat)->col");

#pragma omp parallel for schedule(runtime) private(j,i)
	for (c=0; c<BJDS(mat)->nrowsPadded/BJDS_LEN; c++) 
	{ // loop over chunks

		for (j=0; j<(BJDS(mat)->chunkStart[c+1]-BJDS(mat)->chunkStart[c])/BJDS_LEN; j++)
		{
			for (i=0; i<BJDS_LEN; i++)
			{
				BJDS(mat)->val[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = 0.;
				BJDS(mat)->col[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = 0;
			}
		}
	}



	for (c=0; c<nChunks; c++) {

		for (j=0; j<BJDS(mat)->chunkLen[c]; j++) {

			for (i=0; i<BJDS_LEN; i++) {
				mat_idx_t row = c*BJDS_LEN+i;

				if (j<BJDS(mat)->rowLen[row]) {
					if (flags & GHOST_SPM_SORTED) {
						BJDS(mat)->val[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = cr->val[cr->rpt[(invRowPerm)[row]]+j];
						if (flags & GHOST_SPM_PERMUTECOLIDX)
							BJDS(mat)->col[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = (rowPerm)[cr->col[cr->rpt[(invRowPerm)[row]]+j]];
						else
							BJDS(mat)->col[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = cr->col[cr->rpt[(invRowPerm)[row]]+j];
					} else {
						BJDS(mat)->val[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = cr->val[cr->rpt[row]+j];
						BJDS(mat)->col[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = cr->col[cr->rpt[row]+j];
					}

				} else {
					BJDS(mat)->val[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = 0.0;
					BJDS(mat)->col[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = 0;
				}
				//	printf("%f ",BJDS(mat)->val[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i]);


			}
		}
	}


	DEBUG_LOG(1,"Successfully created BJDS");



}

static void BJDS_kernel_plain (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
	//	sse_kernel_0_intr(lhs, BJDS(mat), rhs, options);	
	mat_idx_t c,j,i;
	mat_data_t tmp[BJDS_LEN]; 

#pragma omp parallel for schedule(runtime) private(j,tmp,i)
	for (c=0; c<BJDS(mat)->nrowsPadded/BJDS_LEN; c++) 
	{ // loop over chunks
		for (i=0; i<BJDS_LEN; i++)
		{
			tmp[i] = 0;
		}

		for (j=0; j<(BJDS(mat)->chunkStart[c+1]-BJDS(mat)->chunkStart[c])/BJDS_LEN; j++) 
		{ // loop inside chunk
			for (i=0; i<BJDS_LEN; i++)
			{
				tmp[i] += BJDS(mat)->val[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] * rhs->val[BJDS(mat)->col[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i]];
			}

		}
		for (i=0; i<BJDS_LEN; i++)
		{
			if (options & GHOST_OPTION_AXPY)
				lhs->val[c*BJDS_LEN+i] += tmp[i];
			else
				lhs->val[c*BJDS_LEN+i] = tmp[i];

		}
	}
}

#ifdef SSE
static void BJDS_kernel_SSE (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * invec, int options)
{
	mat_idx_t c,j;
	mat_nnz_t offs;
	__m128d tmp;
	__m128d val;
	__m128d rhs;

#pragma omp parallel for schedule(runtime) private(j,tmp,val,rhs,offs)
	for (c=0; c<BJDS(mat)->nrowsPadded>>1; c++) 
	{ // loop over chunks
		tmp = _mm_setzero_pd(); // tmp = 0
		offs = BJDS(mat)->chunkStart[c];

		for (j=0; j<(BJDS(mat)->chunkStart[c+1]-BJDS(mat)->chunkStart[c])>>1; j++) 
		{ // loop inside chunk
			val    = _mm_load_pd(&BJDS(mat)->val[offs]);                      // load values
			rhs    = _mm_loadl_pd(rhs,&invec->val[(BJDS(mat)->col[offs++])]); // load first 128 bits of RHS
			rhs    = _mm_loadh_pd(rhs,&invec->val[(BJDS(mat)->col[offs++])]);
			tmp    = _mm_add_pd(tmp,_mm_mul_pd(val,rhs));           // accumulate
		}
		if (options & GHOST_OPTION_AXPY) {
			_mm_store_pd(&lhs->val[c*BJDS_LEN],_mm_add_pd(tmp,_mm_load_pd(&lhs->val[c*BJDS_LEN])));
		} else {
			_mm_stream_pd(&lhs->val[c*BJDS_LEN],tmp);
		}
	}


}
#endif

#ifdef AVX
static void BJDS_kernel_AVX(ghost_mat_t *mat, ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions)
{
	mat_idx_t c,j;
	mat_nnz_t offs;
	__m256d tmp;
	__m256d val;
	__m256d rhs;
	__m128d rhstmp;

#pragma omp parallel for schedule(runtime) private(j,tmp,val,rhs,offs,rhstmp)
	for (c=0; c<BJDS(mat)->nrowsPadded>>2; c++) 
	{ // loop over chunks
		tmp = _mm256_setzero_pd(); // tmp = 0
		offs = BJDS(mat)->chunkStart[c];

		for (j=0; j<(BJDS(mat)->chunkStart[c+1]-BJDS(mat)->chunkStart[c])>>2; j++) 
		{ // loop inside chunk

			val    = _mm256_load_pd(&BJDS(mat)->val[offs]);                      // load values
			rhstmp = _mm_loadl_pd(rhstmp,&invec->val[(BJDS(mat)->col[offs++])]); // load first 128 bits of RHS
			rhstmp = _mm_loadh_pd(rhstmp,&invec->val[(BJDS(mat)->col[offs++])]);
			rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);                  // insert to RHS
			rhstmp = _mm_loadl_pd(rhstmp,&invec->val[(BJDS(mat)->col[offs++])]); // load second 128 bits of RHS
			rhstmp = _mm_loadh_pd(rhstmp,&invec->val[(BJDS(mat)->col[offs++])]);
			rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);                  // insert to RHS
			tmp    = _mm256_add_pd(tmp,_mm256_mul_pd(val,rhs));           // accumulate
		}
		if (spmvmOptions & GHOST_OPTION_AXPY) {
			_mm256_store_pd(&res->val[c*BJDS_LEN],_mm256_add_pd(tmp,_mm256_load_pd(&res->val[c*BJDS_LEN])));
		} else {
			_mm256_stream_pd(&res->val[c*BJDS_LEN],tmp);
		}
	}
}
#endif

#ifdef MIC
static void BJDS_kernel_MIC(ghost_mat_t *mat, ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions)
{
	mat_idx_t c,j;
	mat_nnz_t offs;
	__m512d tmp;
	__m512d val;
	__m512d rhs;
	__m512i idx;

#pragma omp parallel for schedule(runtime) private(j,tmp,val,rhs,idx,offs)
	for (c=0; c<BJDS(mat)->nrowsPadded>>3; c++) 
	{ // loop over chunks
		tmp = _mm512_setzero_pd(); // tmp = 0
		//		int offset = BJDS(mat)->chunkStart[c];
		offs = BJDS(mat)->chunkStart[c];

		for (j=0; j<(BJDS(mat)->chunkStart[c+1]-BJDS(mat)->chunkStart[c])>>3; j+=2) 
		{ // loop inside chunk
			val = _mm512_load_pd(&BJDS(mat)->val[offs]);
			idx = _mm512_load_epi32(&BJDS(mat)->col[offs]);
			rhs = _mm512_i32logather_pd(idx,invec->val,8);
			tmp = _mm512_add_pd(tmp,_mm512_mul_pd(val,rhs));

			offs += 8;

			val = _mm512_load_pd(&BJDS(mat)->val[offs]);
			idx = _mm512_permute4f128_epi32(idx,_MM_PERM_BADC);
			rhs = _mm512_i32logather_pd(idx,invec->val,8);
			tmp = _mm512_add_pd(tmp,_mm512_mul_pd(val,rhs));

			offs += 8;
		}
		if (spmvmOptions & GHOST_OPTION_AXPY) {
			_mm512_storenrngo_pd(&res->val[c*BJDS_LEN],_mm512_add_pd(tmp,_mm512_load_pd(&res->val[c*BJDS_LEN])));
		} else {
			_mm512_storenrngo_pd(&res->val[c*BJDS_LEN],tmp);
		}
	}
}

static void BJDS_kernel_MIC_16(ghost_mat_t *mat, ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions)
{
	mat_idx_t c,j;
	mat_nnz_t offs;
	__m512d tmp1;
	__m512d tmp2;
	__m512d val;
	__m512d rhs;
	__m512i idx;

#pragma omp parallel for schedule(runtime) private(j,tmp1,tmp2,val,rhs,idx,offs)
	for (c=0; c<BJDS(mat)->nrowsPadded>>4; c++) 
	{ // loop over chunks
		tmp1 = _mm512_setzero_pd(); // tmp1 = 0
		tmp2 = _mm512_setzero_pd(); // tmp2 = 0
		offs = BJDS(mat)->chunkStart[c];

		for (j=0; j<(BJDS(mat)->chunkStart[c+1]-BJDS(mat)->chunkStart[c])>>4; j++) 
		{ // loop inside chunk
			val = _mm512_load_pd(&BJDS(mat)->val[offs]);
			idx = _mm512_load_epi32(&BJDS(mat)->col[offs]);
			rhs = _mm512_i32logather_pd(idx,invec->val,8);
			tmp1 = _mm512_add_pd(tmp1,_mm512_mul_pd(val,rhs));

			offs += 8;

			val = _mm512_load_pd(&BJDS(mat)->val[offs]);
			idx = _mm512_permute4f128_epi32(idx,_MM_PERM_BADC);
			rhs = _mm512_i32logather_pd(idx,invec->val,8);
			tmp2 = _mm512_add_pd(tmp2,_mm512_mul_pd(val,rhs));

			offs += 8;
		}
		if (spmvmOptions & GHOST_OPTION_AXPY) {
			_mm512_storenrngo_pd(&res->val[c*BJDS_LEN],_mm512_add_pd(tmp1,_mm512_load_pd(&res->val[c*BJDS_LEN])));
			_mm512_storenrngo_pd(&res->val[c*BJDS_LEN+8],_mm512_add_pd(tmp2,_mm512_load_pd(&res->val[c*BJDS_LEN+8])));
		} else {
			_mm512_storenrngo_pd(&res->val[c*BJDS_LEN],tmp1);
			_mm512_storenrngo_pd(&res->val[c*BJDS_LEN+8],tmp2);
		}
	}
}
#endif
