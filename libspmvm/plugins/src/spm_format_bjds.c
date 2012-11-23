#include "spm_format_bjds.h"
#include "matricks.h"
#include "ghost_util.h"
#include "kernel.h"

#include <immintrin.h>

char name[] = "BJDS plugin for ghost";
char version[] = "0.1a";
char formatID[] = "BJDS";

static mat_nnz_t BJDS_nnz();
static mat_idx_t BJDS_nrows();
static mat_idx_t BJDS_ncols();
static void BJDS_printInfo();
static char * BJDS_formatName();
static mat_idx_t BJDS_rowLen (mat_idx_t i);
static mat_data_t BJDS_entry (mat_idx_t i, mat_idx_t j);
static size_t BJDS_byteSize (void);
static void BJDS_fromCRS(CR_TYPE *cr, mat_trait_t traits);
static void BJDS_fromBin(char *, mat_trait_t traits);
static void BJDS_kernel_plain (ghost_vec_t *, ghost_vec_t *, int);
#ifdef SSE
static void BJDS_kernel_SSE (ghost_vec_t *, ghost_vec_t *, int);
#endif
#ifdef AVX
static void BJDS_kernel_AVX (ghost_vec_t *, ghost_vec_t *, int);
#endif
#ifdef MIC
static void BJDS_kernel_MIC (ghost_vec_t *, ghost_vec_t *, int);
static void BJDS_kernel_MIC_16 (ghost_vec_t *, ghost_vec_t *, int);
#endif

static ghost_mat_t *thisMat;
static BJDS_TYPE *thisBJDS;

ghost_mat_t * init()
{
	DEBUG_LOG(1,"Setting functions for TBJDS matrix");
	ghost_mat_t *mat = (ghost_mat_t *)allocateMemory(sizeof(ghost_mat_t),"matrix");
	thisMat = mat;

	thisBJDS = (BJDS_TYPE *)(thisMat->data);

	thisMat->fromBin = &BJDS_fromBin;
	thisMat->printInfo = &BJDS_printInfo;
	thisMat->formatName = &BJDS_formatName;
	thisMat->rowLen     = &BJDS_rowLen;
	thisMat->entry      = &BJDS_entry;
	thisMat->byteSize   = &BJDS_byteSize;
	thisMat->kernel     = &BJDS_kernel_plain;
#ifdef SSE
	thisMat->kernel   = &BJDS_kernel_SSE;
#endif
#ifdef AVX
	thisMat->kernel   = &BJDS_kernel_AVX;
#endif
#ifdef MIC
	thisMat->kernel   = &BJDS_kernel_MIC_16;
	UNUSED(&BJDS_kernel_MIC);
#endif
	thisMat->nnz      = &BJDS_nnz;
	thisMat->nrows    = &BJDS_nrows;
	thisMat->ncols    = &BJDS_ncols;

	return mat;
}

static mat_nnz_t BJDS_nnz()
{
	return thisBJDS->nnz;
}
static mat_idx_t BJDS_nrows()
{
	return thisBJDS->nrows;
}
static mat_idx_t BJDS_ncols()
{
	return 0;
}

static void BJDS_printInfo()
{
	SpMVM_printLine("Vector block size",NULL,"%d",BJDS_LEN);
	SpMVM_printLine("Row length oscillation nu",NULL,"%f",thisBJDS->nu);
	if (thisMat->trait.flags & GHOST_SPM_SORTED) {
		SpMVM_printLine("Sorted",NULL,"yes");
		SpMVM_printLine("Sort block size",NULL,"%u",*(unsigned int *)(thisMat->trait.aux));
		SpMVM_printLine("Permuted columns",NULL,"%s",thisMat->trait.flags&GHOST_SPM_PERMUTECOLIDX?"yes":"no");
	} else {
		SpMVM_printLine("Sorted",NULL,"no");
	}
}

static char * BJDS_formatName()
{
	return "BJDS";
}

static mat_idx_t BJDS_rowLen (mat_idx_t i)
{
	if (thisMat->trait.flags & GHOST_SPM_SORTED)
		i = thisMat->rowPerm[i];

	return thisBJDS->rowLen[i];
}

static mat_data_t BJDS_entry (mat_idx_t i, mat_idx_t j)
{
	mat_idx_t e;

	if (thisMat->trait.flags & GHOST_SPM_SORTED)
		i = thisMat->rowPerm[i];
	if (thisMat->trait.flags & GHOST_SPM_PERMUTECOLIDX)
		j = thisMat->rowPerm[j];

	for (e=thisBJDS->chunkStart[i/BJDS_LEN]+i%BJDS_LEN; 
			e<thisBJDS->chunkStart[i/BJDS_LEN+1]; 
			e+=BJDS_LEN) {
		if (thisBJDS->col[e] == j)
			return thisBJDS->val[e];
	}
	return 0.;
}

static size_t BJDS_byteSize (void)
{
	return (size_t)((thisBJDS->nrowsPadded/BJDS_LEN)*sizeof(mat_nnz_t) + 
			thisBJDS->nEnts*(sizeof(mat_idx_t)+sizeof(mat_data_t)));
}

static void BJDS_fromBin(char *matrixPath, mat_trait_t traits)
{
	// TODO
	ghost_mat_t *crsMat = SpMVM_initMatrix("CRS");
	mat_trait_t crsTraits = {.format = "CRS",.flags=GHOST_SPM_DEFAULT,NULL};
	crsMat->fromBin(matrixPath,crsTraits);
	
	BJDS_fromCRS(crsMat->data,traits);
}

static void BJDS_fromCRS(CR_TYPE *cr, mat_trait_t trait)
{
	DEBUG_LOG(1,"Creating BJDS matrix");
	mat_idx_t i,j,c;
	unsigned int flags = trait.flags;

	mat_idx_t *rowPerm = NULL;
	mat_idx_t *invRowPerm = NULL;

	JD_SORT_TYPE* rowSort;

	thisBJDS = (BJDS_TYPE *)allocateMemory(sizeof(BJDS_TYPE),"thisBJDS");
	thisMat->data = thisBJDS;
	thisMat->trait = trait;
//	thisMat->nrows = cr->nrows;
//	thisMat->ncols = cr->ncols;
//	thisMat->nnz = cr->nEnts;
	thisMat->rowPerm = rowPerm;
	thisMat->invRowPerm = invRowPerm;
	/**thisMat = (ghost_mat_t)MATRIX_INIT(
	  .trait = trait, 
	  .nrows = cr->nrows, 
	  .ncols = cr->ncols, 
	  .nnz = cr->nEnts,
	  .rowPerm = rowPerm,
	  .invRowPerm = invRowPerm,	   
	  .data = thisBJDS);
	 */
	if (trait.flags & GHOST_SPM_SORTED) {
		rowPerm = (mat_idx_t *)allocateMemory(cr->nrows*sizeof(mat_idx_t),"thisBJDS->rowPerm");
		invRowPerm = (mat_idx_t *)allocateMemory(cr->nrows*sizeof(mat_idx_t),"thisBJDS->invRowPerm");

		(thisMat)->rowPerm = rowPerm;
		(thisMat)->invRowPerm = invRowPerm;
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




	thisBJDS->nrows = cr->nrows;
	thisBJDS->nnz = cr->nEnts;
	thisBJDS->nEnts = 0;
	thisBJDS->nrowsPadded = pad(thisBJDS->nrows,BJDS_LEN);

	mat_idx_t nChunks = thisBJDS->nrowsPadded/BJDS_LEN;
	thisBJDS->chunkStart = (mat_nnz_t *)allocateMemory((nChunks+1)*sizeof(mat_nnz_t),"thisBJDS->chunkStart");
	thisBJDS->chunkMin = (mat_idx_t *)allocateMemory((nChunks)*sizeof(mat_idx_t),"thisBJDS->chunkMin");
	thisBJDS->chunkLen = (mat_idx_t *)allocateMemory((nChunks)*sizeof(mat_idx_t),"thisBJDS->chunkMin");
	thisBJDS->rowLen = (mat_idx_t *)allocateMemory((thisBJDS->nrowsPadded)*sizeof(mat_idx_t),"thisBJDS->chunkMin");
	thisBJDS->chunkStart[0] = 0;

	mat_idx_t chunkMin = cr->ncols;
	mat_idx_t chunkLen = 0;
	mat_idx_t curChunk = 1;
	thisBJDS->nu = 0.;

	for (i=0; i<thisBJDS->nrowsPadded; i++) {
		if (i<cr->nrows) {
			if (flags & GHOST_SPM_SORTED)
				thisBJDS->rowLen[i] = rowSort[i].nEntsInRow;
			else
				thisBJDS->rowLen[i] = cr->rpt[i+1]-cr->rpt[i];
		} else {
			thisBJDS->rowLen[i] = 0;
		}


		chunkMin = thisBJDS->rowLen[i]<chunkMin?thisBJDS->rowLen[i]:chunkMin;
		chunkLen = thisBJDS->rowLen[i]>chunkLen?thisBJDS->rowLen[i]:chunkLen;

		if ((i+1)%BJDS_LEN == 0) {
			thisBJDS->nEnts += BJDS_LEN*chunkLen;
			thisBJDS->chunkStart[curChunk] = thisBJDS->nEnts;
			thisBJDS->chunkMin[curChunk-1] = chunkMin;
			thisBJDS->chunkLen[curChunk-1] = chunkLen;

			thisBJDS->nu += (double)chunkMin/chunkLen;

			chunkMin = cr->ncols;
			chunkLen = 0;
			curChunk++;
		}
	}
	thisBJDS->nu /= (double)nChunks;

	thisBJDS->val = (mat_data_t *)allocateMemory(sizeof(mat_data_t)*thisBJDS->nEnts,"thisBJDS->val");
	thisBJDS->col = (mat_idx_t *)allocateMemory(sizeof(mat_idx_t)*thisBJDS->nEnts,"thisBJDS->col");

#pragma omp parallel for schedule(runtime) private(j,i)
	for (c=0; c<thisBJDS->nrowsPadded/BJDS_LEN; c++) 
	{ // loop over chunks

		for (j=0; j<(thisBJDS->chunkStart[c+1]-thisBJDS->chunkStart[c])/BJDS_LEN; j++)
		{
			for (i=0; i<BJDS_LEN; i++)
			{
				thisBJDS->val[thisBJDS->chunkStart[c]+j*BJDS_LEN+i] = 0.;
				thisBJDS->col[thisBJDS->chunkStart[c]+j*BJDS_LEN+i] = 0;
			}
		}
	}



	for (c=0; c<nChunks; c++) {

		for (j=0; j<thisBJDS->chunkLen[c]; j++) {

			for (i=0; i<BJDS_LEN; i++) {
				mat_idx_t row = c*BJDS_LEN+i;

				if (j<thisBJDS->rowLen[row]) {
					if (flags & GHOST_SPM_SORTED) {
						thisBJDS->val[thisBJDS->chunkStart[c]+j*BJDS_LEN+i] = cr->val[cr->rpt[(invRowPerm)[row]]+j];
						if (flags & GHOST_SPM_PERMUTECOLIDX)
							thisBJDS->col[thisBJDS->chunkStart[c]+j*BJDS_LEN+i] = (rowPerm)[cr->col[cr->rpt[(invRowPerm)[row]]+j]];
						else
							thisBJDS->col[thisBJDS->chunkStart[c]+j*BJDS_LEN+i] = cr->col[cr->rpt[(invRowPerm)[row]]+j];
					} else {
						thisBJDS->val[thisBJDS->chunkStart[c]+j*BJDS_LEN+i] = cr->val[cr->rpt[row]+j];
						thisBJDS->col[thisBJDS->chunkStart[c]+j*BJDS_LEN+i] = cr->col[cr->rpt[row]+j];
					}

				} else {
					thisBJDS->val[thisBJDS->chunkStart[c]+j*BJDS_LEN+i] = 0.0;
					thisBJDS->col[thisBJDS->chunkStart[c]+j*BJDS_LEN+i] = 0;
				}
				//	printf("%f ",thisBJDS->val[thisBJDS->chunkStart[c]+j*BJDS_LEN+i]);


			}
		}
	}


	DEBUG_LOG(1,"Successfully created BJDS");



}

static void BJDS_kernel_plain (ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
	//	sse_kernel_0_intr(lhs, thisBJDS, rhs, options);	
	mat_idx_t c,j,i;
	mat_data_t tmp[BJDS_LEN]; 

#pragma omp parallel for schedule(runtime) private(j,tmp,i)
	for (c=0; c<thisBJDS->nrowsPadded/BJDS_LEN; c++) 
	{ // loop over chunks
		for (i=0; i<BJDS_LEN; i++)
		{
			tmp[i] = 0;
		}

		for (j=0; j<(thisBJDS->chunkStart[c+1]-thisBJDS->chunkStart[c])/BJDS_LEN; j++) 
		{ // loop inside chunk
			for (i=0; i<BJDS_LEN; i++)
			{
				tmp[i] += thisBJDS->val[thisBJDS->chunkStart[c]+j*BJDS_LEN+i] * rhs->val[thisBJDS->col[thisBJDS->chunkStart[c]+j*BJDS_LEN+i]];
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
static void BJDS_kernel_SSE (ghost_vec_t * lhs, ghost_vec_t * invec, int options)
{
	mat_idx_t c,j;
	mat_nnz_t offs;
	__m128d tmp;
	__m128d val;
	__m128d rhs;

#pragma omp parallel for schedule(runtime) private(j,tmp,val,rhs,offs)
	for (c=0; c<thisBJDS->nrowsPadded>>1; c++) 
	{ // loop over chunks
		tmp = _mm_setzero_pd(); // tmp = 0
		offs = thisBJDS->chunkStart[c];

		for (j=0; j<(thisBJDS->chunkStart[c+1]-thisBJDS->chunkStart[c])>>1; j++) 
		{ // loop inside chunk
			val    = _mm_load_pd(&thisBJDS->val[offs]);                      // load values
			rhs    = _mm_loadl_pd(rhs,&invec->val[(thisBJDS->col[offs++])]); // load first 128 bits of RHS
			rhs    = _mm_loadh_pd(rhs,&invec->val[(thisBJDS->col[offs++])]);
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
static void BJDS_kernel_AVX(ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions)
{
	mat_idx_t c,j;
	mat_nnz_t offs;
	__m256d tmp;
	__m256d val;
	__m256d rhs;
	__m128d rhstmp;

#pragma omp parallel for schedule(runtime) private(j,tmp,val,rhs,offs,rhstmp)
	for (c=0; c<thisBJDS->nrowsPadded>>2; c++) 
	{ // loop over chunks
		tmp = _mm256_setzero_pd(); // tmp = 0
		offs = thisBJDS->chunkStart[c];

		for (j=0; j<(thisBJDS->chunkStart[c+1]-thisBJDS->chunkStart[c])>>2; j++) 
		{ // loop inside chunk

			val    = _mm256_load_pd(&thisBJDS->val[offs]);                      // load values
			rhstmp = _mm_loadl_pd(rhstmp,&invec->val[(thisBJDS->col[offs++])]); // load first 128 bits of RHS
			rhstmp = _mm_loadh_pd(rhstmp,&invec->val[(thisBJDS->col[offs++])]);
			rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);                  // insert to RHS
			rhstmp = _mm_loadl_pd(rhstmp,&invec->val[(thisBJDS->col[offs++])]); // load second 128 bits of RHS
			rhstmp = _mm_loadh_pd(rhstmp,&invec->val[(thisBJDS->col[offs++])]);
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
static void BJDS_kernel_MIC(ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions)
{
	mat_idx_t c,j;
	mat_nnz_t offs;
	__m512d tmp;
	__m512d val;
	__m512d rhs;
	__m512i idx;

#pragma omp parallel for schedule(runtime) private(j,tmp,val,rhs,idx,offs)
	for (c=0; c<thisBJDS->nrowsPadded>>3; c++) 
	{ // loop over chunks
		tmp = _mm512_setzero_pd(); // tmp = 0
		//		int offset = thisBJDS->chunkStart[c];
		offs = thisBJDS->chunkStart[c];

		for (j=0; j<(thisBJDS->chunkStart[c+1]-thisBJDS->chunkStart[c])>>3; j+=2) 
		{ // loop inside chunk
			val = _mm512_load_pd(&thisBJDS->val[offs]);
			idx = _mm512_load_epi32(&thisBJDS->col[offs]);
			rhs = _mm512_i32logather_pd(idx,invec->val,8);
			tmp = _mm512_add_pd(tmp,_mm512_mul_pd(val,rhs));

			offs += 8;

			val = _mm512_load_pd(&thisBJDS->val[offs]);
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

static void BJDS_kernel_MIC_16(ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions)
{
	mat_idx_t c,j;
	mat_nnz_t offs;
	__m512d tmp1;
	__m512d tmp2;
	__m512d val;
	__m512d rhs;
	__m512i idx;

#pragma omp parallel for schedule(runtime) private(j,tmp1,tmp2,val,rhs,idx,offs)
	for (c=0; c<thisBJDS->nrowsPadded>>4; c++) 
	{ // loop over chunks
		tmp1 = _mm512_setzero_pd(); // tmp1 = 0
		tmp2 = _mm512_setzero_pd(); // tmp2 = 0
		offs = thisBJDS->chunkStart[c];

		for (j=0; j<(thisBJDS->chunkStart[c+1]-thisBJDS->chunkStart[c])>>4; j++) 
		{ // loop inside chunk
			val = _mm512_load_pd(&thisBJDS->val[offs]);
			idx = _mm512_load_epi32(&thisBJDS->col[offs]);
			rhs = _mm512_i32logather_pd(idx,invec->val,8);
			tmp1 = _mm512_add_pd(tmp1,_mm512_mul_pd(val,rhs));

			offs += 8;

			val = _mm512_load_pd(&thisBJDS->val[offs]);
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
