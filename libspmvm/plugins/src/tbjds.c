#include "spm_format_tbjds.h"
#include "matricks.h"
#include "ghost_util.h"
#include "kernel.h"

#include <immintrin.h>

#define TBJDS(mat) ((TBJDS_TYPE *)(mat->data))

char name[] = "TBJDS plugin for ghost";
char version[] = "0.1a";
char formatID[] = "TBJDS";

static ghost_mnnz_t TBJDS_nnz(ghost_mat_t *mat);
static ghost_midx_t TBJDS_nrows(ghost_mat_t *mat);
static ghost_midx_t TBJDS_ncols(ghost_mat_t *mat);
static void TBJDS_printInfo(ghost_mat_t *mat);
static char * TBJDS_formatName(ghost_mat_t *mat);
static ghost_midx_t TBJDS_rowLen (ghost_mat_t *mat, ghost_midx_t i);
static ghost_mdat_t TBJDS_entry (ghost_mat_t *mat, ghost_midx_t i, ghost_midx_t j);
static size_t TBJDS_byteSize (ghost_mat_t *mat);
static void TBJDS_fromCRS(ghost_mat_t *mat, CR_TYPE *cr);
static void TBJDS_fromBin(ghost_mat_t *mat, char *);
static void TBJDS_free(ghost_mat_t *mat);
static void TBJDS_kernel_plain (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options);
#ifdef SSE
static void TBJDS_kernel_SSE(ghost_mat_t *mat, ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions);
#endif
#ifdef AVX
static void TBJDS_kernel_AVX(ghost_mat_t *mat, ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions);
static void TBJDS_kernel_AVX_colwise(ghost_mat_t *mat, ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions);
#endif
#ifdef MIC
static void TBJDS_kernel_MIC_16(ghost_mat_t *mat, ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions);
#endif

ghost_mat_t * init(ghost_mtraits_t * traits)
{
	ghost_mat_t *mat = (ghost_mat_t *)allocateMemory(sizeof(ghost_mat_t),"matrix");
	mat->traits = traits;
	DEBUG_LOG(1,"Setting functions for TBJDS matrix");

	mat->fromBin = &TBJDS_fromBin;
	mat->printInfo = &TBJDS_printInfo;
	mat->formatName = &TBJDS_formatName;
	mat->rowLen   = &TBJDS_rowLen;
	mat->entry    = &TBJDS_entry;
	mat->byteSize = &TBJDS_byteSize;
	mat->kernel   = &TBJDS_kernel_plain;
#ifdef SSE
	mat->kernel   = &TBJDS_kernel_SSE;
#endif
#ifdef AVX
	mat->kernel   = &TBJDS_kernel_AVX;
	UNUSED(&TBJDS_kernel_AVX_colwise);
#endif
#ifdef MIC
	mat->kernel   = &TBJDS_kernel_MIC_16;
#endif
	mat->nnz      = &TBJDS_nnz;
	mat->nrows    = &TBJDS_nrows;
	mat->ncols    = &TBJDS_ncols;
	mat->destroy  = &TBJDS_free;

	return mat;
}

static ghost_mnnz_t TBJDS_nnz(ghost_mat_t *mat)
{
	return TBJDS(mat)->nnz;
}
static ghost_midx_t TBJDS_nrows(ghost_mat_t *mat)
{
	return TBJDS(mat)->nrows;
}
static ghost_midx_t TBJDS_ncols(ghost_mat_t *mat)
{
	UNUSED(mat);
	return 0;
}
static char * TBJDS_formatName(ghost_mat_t *mat)
{
	UNUSED(mat);
	return "TBJDS";
}

static void TBJDS_printInfo(ghost_mat_t *mat)
{
	ghost_printLine("Vector block size",NULL,"%d",BJDS_LEN);
	ghost_printLine("Row length oscillation nu",NULL,"%f",TBJDS(mat)->nu);
	if (mat->traits->flags & GHOST_SPM_SORTED) {
		ghost_printLine("Sorted",NULL,"yes");
		ghost_printLine("Sort block size",NULL,"%u",*(unsigned int *)(mat->traits->aux));
		ghost_printLine("Permuted columns",NULL,"%s",mat->traits->flags&GHOST_SPM_PERMUTECOLIDX?"yes":"no");
	} else {
		ghost_printLine("Sorted",NULL,"no");
	}



}


static ghost_midx_t TBJDS_rowLen (ghost_mat_t *mat, ghost_midx_t i)
{
	if (mat->traits->flags & GHOST_SPM_SORTED)
		i = mat->rowPerm[i];

	return TBJDS(mat)->rowLen[i];
}

static ghost_mdat_t TBJDS_entry (ghost_mat_t *mat, ghost_midx_t i, ghost_midx_t j)
{
	ghost_midx_t e;
	if (mat->traits->flags & GHOST_SPM_SORTED)
		i = mat->rowPerm[i];
	if (mat->traits->flags & GHOST_SPM_PERMUTECOLIDX)
		j = mat->rowPerm[j];


	for (e=TBJDS(mat)->chunkStart[i/BJDS_LEN]+i%BJDS_LEN; 
			e<TBJDS(mat)->chunkStart[i/BJDS_LEN+1]; 
			e+=BJDS_LEN) {
		if (TBJDS(mat)->col[e] == j)
			return TBJDS(mat)->val[e];
	}
	return 0.;
}

static size_t TBJDS_byteSize (ghost_mat_t *mat)
{
	return (size_t)(
			(TBJDS(mat)->nrowsPadded/BJDS_LEN) * (sizeof(ghost_mnnz_t)+sizeof(ghost_midx_t)) + // chunkStart + chunkMin 
			TBJDS(mat)->nrows * sizeof(ghost_midx_t) + // rowLen
			TBJDS(mat)->nEnts * (sizeof(ghost_midx_t)+sizeof(ghost_mdat_t))); // col + val
}

static void TBJDS_fromBin(ghost_mat_t *mat, char *matrixPath)
{
	// TODO
	ghost_mtraits_t crsTraits = {.format = "CRS",.flags=GHOST_SPM_HOST,NULL};
	ghost_mat_t *crsMat = ghost_initMatrix(&crsTraits);
	crsMat->fromBin(crsMat,matrixPath);

	TBJDS_fromCRS(mat,crsMat->data);
}

static void TBJDS_fromCRS(ghost_mat_t *mat, CR_TYPE *cr)
{
	ghost_midx_t i,j,c;
	JD_SORT_TYPE* rowSort;
	ghost_midx_t *rowPerm = NULL, *invRowPerm = NULL;
	unsigned int flags;

	mat->data = (TBJDS_TYPE *)allocateMemory(sizeof(TBJDS_TYPE),"mv");
	mat->data = TBJDS(mat);
	mat->rowPerm = rowPerm;
	mat->invRowPerm = invRowPerm;

	flags = mat->traits->flags;

	if (mat->traits->flags & GHOST_SPM_SORTED) {
		rowPerm = (ghost_midx_t *)allocateMemory(cr->nrows*sizeof(ghost_midx_t),"sTBJDS(mat)->rowPerm");
		invRowPerm = (ghost_midx_t *)allocateMemory(cr->nrows*sizeof(ghost_midx_t),"sTBJDS(mat)->invRowPerm");

		mat->rowPerm = rowPerm;
		mat->invRowPerm = invRowPerm;

		int sortBlock = *(int *)(mat->traits->aux);
		if (sortBlock == 0)
			sortBlock = cr->nrows;

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
		/*	i=0;
			while(i < cr->nrows) {
			int start = i;

			j = rowSort[start].nEntsInRow;
			while( i<cr->nrows && rowSort[i].nEntsInRow >= j ) 
			++i;

			DEBUG_LOG(1,"sorting over %i rows (%i): %i - %i\n",i-start,j, start, i-1);
			qsort( &rowSort[start], i-start, sizeof(JD_SORT_TYPE), compareNZEOrgPos );
			}

			for(i=1; i < cr->nrows; ++i) {
			if( rowSort[i].nEntsInRow == rowSort[i-1].nEntsInRow && rowSort[i].row < rowSort[i-1].row)
			printf("Error in row %i: descending row number\n",i);
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
	TBJDS(mat)->nrows = cr->nrows;
	TBJDS(mat)->nnz = cr->nEnts;
	TBJDS(mat)->nEnts = 0;
	TBJDS(mat)->nrowsPadded = pad(TBJDS(mat)->nrows,BJDS_LEN);
	ghost_midx_t nChunks = TBJDS(mat)->nrowsPadded/BJDS_LEN;
	TBJDS(mat)->chunkStart = (ghost_mnnz_t *)allocateMemory((nChunks+1)*sizeof(ghost_mnnz_t),"TBJDS(mat)->chunkStart");
	TBJDS(mat)->chunkMin = (ghost_midx_t *)allocateMemory((nChunks)*sizeof(ghost_midx_t),"TBJDS(mat)->chunkMin");
	TBJDS(mat)->chunkLen = (ghost_midx_t *)allocateMemory((nChunks)*sizeof(ghost_midx_t),"TBJDS(mat)->chunkMin");
	TBJDS(mat)->rowLen = (ghost_midx_t *)allocateMemory((TBJDS(mat)->nrowsPadded)*sizeof(ghost_midx_t),"TBJDS(mat)->chunkMin");
	TBJDS(mat)->chunkStart[0] = 0;

	//	for(i=0; i < cr->nrows; ++i) printf("%d\n",(*invRowPerm)[i]);

	ghost_midx_t chunkMin = cr->ncols;
	ghost_midx_t chunkLen = 0;
	ghost_midx_t curChunk = 1;
	TBJDS(mat)->nu = 0.;

	for (i=0; i<TBJDS(mat)->nrowsPadded; i++) {
		if (i<cr->nrows) {
			if (flags & GHOST_SPM_SORTED)
				TBJDS(mat)->rowLen[i] = rowSort[i].nEntsInRow;
			else
				TBJDS(mat)->rowLen[i] = cr->rpt[i+1]-cr->rpt[i];
		} else {
			TBJDS(mat)->rowLen[i] = 0;
		}

		TBJDS(mat)->nEnts += TBJDS(mat)->rowLen[i];

		chunkMin = TBJDS(mat)->rowLen[i]<chunkMin?TBJDS(mat)->rowLen[i]:chunkMin;
		chunkLen = TBJDS(mat)->rowLen[i]>chunkLen?TBJDS(mat)->rowLen[i]:chunkLen;

		if ((i+1)%BJDS_LEN == 0) {
#ifdef MIC
			TBJDS(mat)->nEnts = pad(TBJDS(mat)->nEnts,16); // MIC has to be 512-bit aligned
#endif
			TBJDS(mat)->chunkStart[curChunk] = TBJDS(mat)->nEnts;
			TBJDS(mat)->chunkMin[curChunk-1] = chunkMin;
			TBJDS(mat)->chunkLen[curChunk-1] = chunkLen;

			TBJDS(mat)->nu += (double)chunkMin/chunkLen;

			chunkMin = cr->ncols;
			chunkLen = 0;
			curChunk++;
		}
	}
	TBJDS(mat)->nu /= (double)nChunks;

	TBJDS(mat)->val = (ghost_mdat_t *)allocateMemory(sizeof(ghost_mdat_t)*TBJDS(mat)->nEnts,"TBJDS(mat)->val");
	TBJDS(mat)->col = (ghost_midx_t *)allocateMemory(sizeof(ghost_midx_t)*TBJDS(mat)->nEnts,"TBJDS(mat)->col");

	//printf("nEnts: %d\n",TBJDS(mat)->nEnts);

#pragma omp parallel for schedule(runtime) private(j,i)
	for (c=0; c<TBJDS(mat)->nrowsPadded/BJDS_LEN; c++) 
	{ // loop over chunks

		for (j=0; j<TBJDS(mat)->chunkMin[c]; j++)
		{
			for (i=0; i<BJDS_LEN; i++)
			{
				TBJDS(mat)->val[TBJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = 0.;
				TBJDS(mat)->col[TBJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = 0;
			}
		}
		ghost_mnnz_t rem = TBJDS(mat)->chunkStart[c] + TBJDS(mat)->chunkMin[c]*BJDS_LEN;
		for (i=0; i<BJDS_LEN; i++)
		{
			for (j=TBJDS(mat)->chunkMin[c]; j<TBJDS(mat)->rowLen[c*BJDS_LEN+i]; j++)
			{
				TBJDS(mat)->val[rem] = 0.;
				TBJDS(mat)->col[rem++] = 0;
			}
		}
	}
	for (c=0; c<TBJDS(mat)->nrowsPadded/BJDS_LEN; c++) 
	{ // loop over chunks

		// store block
		for (j=0; j<TBJDS(mat)->chunkMin[c]; j++)
		{
			for (i=0; i<BJDS_LEN; i++)
			{
				if (flags & GHOST_SPM_SORTED) {
					TBJDS(mat)->val[TBJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = cr->val[cr->rpt[(invRowPerm)[c*BJDS_LEN+i]]+j];
					if (flags & GHOST_SPM_PERMUTECOLIDX)
						TBJDS(mat)->col[TBJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = (rowPerm)[cr->col[cr->rpt[(invRowPerm)[c*BJDS_LEN+i]]+j]];
					else
						TBJDS(mat)->col[TBJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = cr->col[cr->rpt[(invRowPerm)[c*BJDS_LEN+i]]+j];
				} else {
					TBJDS(mat)->val[TBJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = cr->val[cr->rpt[c*BJDS_LEN+i]+j];
					TBJDS(mat)->col[TBJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = cr->col[cr->rpt[c*BJDS_LEN+i]+j];
				}
			}
		}

		// store remainder
		ghost_mnnz_t rem = TBJDS(mat)->chunkStart[c] + TBJDS(mat)->chunkMin[c]*BJDS_LEN;
		if (flags & GHOST_SPM_COLMAJOR) 
		{
			for (j=TBJDS(mat)->chunkMin[c]; j<TBJDS(mat)->chunkLen[c]; j++)
			{
				for (i=0; i<BJDS_LEN; i++)
				{
					if (j<TBJDS(mat)->rowLen[c*BJDS_LEN+i] ) {
						if (flags & GHOST_SPM_SORTED) {
							TBJDS(mat)->val[rem] = cr->val[cr->rpt[(invRowPerm)[c*BJDS_LEN+i]]+j];
							if (flags & GHOST_SPM_PERMUTECOLIDX)
								TBJDS(mat)->col[rem++] = (rowPerm)[cr->col[cr->rpt[(invRowPerm)[c*BJDS_LEN+i]]+j]];
							else
								TBJDS(mat)->col[rem++] = cr->col[cr->rpt[(invRowPerm)[c*BJDS_LEN+i]]+j];
						} else {
							TBJDS(mat)->val[rem] = cr->val[cr->rpt[c*BJDS_LEN+i]+j];
							TBJDS(mat)->col[rem++] = cr->col[cr->rpt[c*BJDS_LEN+i]+j];
						}
					}
				}
			}
		} else // row major is the default 
		{
			for (i=0; i<BJDS_LEN; i++)
			{
				for (j=TBJDS(mat)->chunkMin[c]; j<TBJDS(mat)->rowLen[c*BJDS_LEN+i]; j++)
				{
					if (flags & GHOST_SPM_SORTED) {
						TBJDS(mat)->val[rem] = cr->val[cr->rpt[(invRowPerm)[c*BJDS_LEN+i]]+j];
						if (flags & GHOST_SPM_PERMUTECOLIDX)
							TBJDS(mat)->col[rem++] = (rowPerm)[cr->col[cr->rpt[(invRowPerm)[c*BJDS_LEN+i]]+j]];
						else
							TBJDS(mat)->col[rem++] = cr->col[cr->rpt[(invRowPerm)[c*BJDS_LEN+i]]+j];
					} else {
						TBJDS(mat)->val[rem] = cr->val[cr->rpt[c*BJDS_LEN+i]+j];
						TBJDS(mat)->col[rem++] = cr->col[cr->rpt[c*BJDS_LEN+i]+j];
					}
				}
			}
		}
	}
}

static void TBJDS_free(ghost_mat_t *mat)
{
	free(TBJDS(mat)->val);
	free(TBJDS(mat)->col);
	free(TBJDS(mat)->chunkStart);
	free(TBJDS(mat)->chunkMin);
	free(TBJDS(mat)->chunkLen);
	free(TBJDS(mat)->rowLen);
	
	free(mat->data);
	free(mat->rowPerm);
	free(mat->invRowPerm);

	free(mat);

}

static void TBJDS_kernel_plain (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
	DEBUG_LOG(2,"Calling plain TBJDS kernel");
	ghost_midx_t c,j,i;
	ghost_mnnz_t offs;
	ghost_mdat_t tmp[BJDS_LEN]; 

#pragma omp parallel for schedule(runtime) private(j,tmp,i,offs)
	for (c=0; c<TBJDS(mat)->nrowsPadded/BJDS_LEN; c++) 
	{ // loop over chunks

		offs = TBJDS(mat)->chunkStart[c];
		for (i=0; i<BJDS_LEN; i++)
		{
			tmp[i] = 0;
		}

		for (j=0; j<TBJDS(mat)->chunkMin[c]; j++) 
		{ // loop inside chunk
			for (i=0; i<BJDS_LEN; i++)
			{
				tmp[i] += TBJDS(mat)->val[offs] * rhs->val[TBJDS(mat)->col[offs++]];
			}

		}
		for (i=0; i<BJDS_LEN; i++)
		{
			for (j=TBJDS(mat)->chunkMin[c]; j<TBJDS(mat)->rowLen[c*BJDS_LEN+i]; j++)
			{
				tmp[i] += TBJDS(mat)->val[offs] * rhs->val[TBJDS(mat)->col[offs++]];
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
static void TBJDS_kernel_SSE(ghost_mat_t *mat, ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions)
{
	ghost_midx_t c,j;
	ghost_mnnz_t offs;
	__m128d tmp;
	__m128d val;
	__m128d rhs;

#pragma omp parallel for schedule(runtime) private(j,tmp,val,rhs,offs)
	for (c=0; c<TBJDS(mat)->nrowsPadded>>1; c++) 
	{ // loop over chunks

		tmp = _mm_setzero_pd(); // tmp = 0
		offs = TBJDS(mat)->chunkStart[c];


		for (j=0; j<TBJDS(mat)->chunkMin[c]; j++) 
		{ // loop inside chunk
			
			val    = _mm_loadu_pd(&TBJDS(mat)->val[offs]);                     // load values
			rhs    = _mm_loadl_pd(rhs,&invec->val[(TBJDS(mat)->col[offs++])]); // load first 64 bits of RHS
			rhs    = _mm_loadh_pd(rhs,&invec->val[(TBJDS(mat)->col[offs++])]);
			tmp    = _mm_add_pd(tmp,_mm_mul_pd(val,rhs));           // accumulate
		}

		// TODO: 4 loops for single precision
		for (j=TBJDS(mat)->chunkMin[c]; j<TBJDS(mat)->rowLen[c*BJDS_LEN]; j++)
		{
			res->val[c*BJDS_LEN] += TBJDS(mat)->val[offs]*invec->val[TBJDS(mat)->col[offs++]];
		}
		for (j=TBJDS(mat)->chunkMin[c]; j<TBJDS(mat)->rowLen[c*BJDS_LEN+1]; j++)
		{
			res->val[c*BJDS_LEN+1] += TBJDS(mat)->val[offs]*invec->val[TBJDS(mat)->col[offs++]];
		}


		if (spmvmOptions & GHOST_OPTION_AXPY) {
			_mm_store_pd(&res->val[c*BJDS_LEN],_mm_add_pd(tmp,_mm_load_pd(&res->val[c*BJDS_LEN])));
		} else {
			_mm_stream_pd(&res->val[c*BJDS_LEN],tmp);
		}
	}
}
#endif

#ifdef AVX
static void TBJDS_kernel_AVX(ghost_mat_t *mat, ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions)
{
	ghost_midx_t c,j;
	ghost_mnnz_t offs;
	__m256d tmp;
	__m256d val;
	__m256d rhs;
	__m128d rhstmp;

#pragma omp parallel for schedule(runtime) private(j,tmp,val,rhs,offs,rhstmp)
	for (c=0; c<TBJDS(mat)->nrowsPadded>>2; c++) 
	{ // loop over chunks
		tmp = _mm256_setzero_pd(); // tmp = 0
		offs = TBJDS(mat)->chunkStart[c];

		for (j=0; j<TBJDS(mat)->chunkMin[c]; j++) 
		{ // loop inside chunk
			val    = _mm256_load_pd(&TBJDS(mat)->val[offs]);                      // load values
			rhstmp = _mm_loadl_pd(rhstmp,&invec->val[(TBJDS(mat)->col[offs++])]); // load first 128 bits of RHS
			rhstmp = _mm_loadh_pd(rhstmp,&invec->val[(TBJDS(mat)->col[offs++])]);
			rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);                  // insert to RHS
			rhstmp = _mm_loadl_pd(rhstmp,&invec->val[(TBJDS(mat)->col[offs++])]); // load second 128 bits of RHS
			rhstmp = _mm_loadh_pd(rhstmp,&invec->val[(TBJDS(mat)->col[offs++])]);
			rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);                  // insert to RHS
			tmp    = _mm256_add_pd(tmp,_mm256_mul_pd(val,rhs));           // accumulate
		}
			/*printf("rem 1 %d..%d\n",TBJDS(mat)->chunkMin[c],TBJDS(mat)->rowLen[c*BJDS_LEN]);
			printf("rem 2 %d..%d\n",TBJDS(mat)->chunkMin[c],TBJDS(mat)->rowLen[c*BJDS_LEN+1]);
			printf("rem 3 %d..%d\n",TBJDS(mat)->chunkMin[c],TBJDS(mat)->rowLen[c*BJDS_LEN+2]);
		printf("rem 4 %d..%d\n",TBJDS(mat)->chunkMin[c],TBJDS(mat)->rowLen[c*BJDS_LEN+3]);*/
		for (j=TBJDS(mat)->chunkMin[c]; j<TBJDS(mat)->rowLen[c*BJDS_LEN]; j++)
		{
			res->val[c*BJDS_LEN] += TBJDS(mat)->val[offs] * invec->val[TBJDS(mat)->col[offs++]];
		}
		for (j=TBJDS(mat)->chunkMin[c]; j<TBJDS(mat)->rowLen[c*BJDS_LEN+1]; j++)
		{
			res->val[c*BJDS_LEN+1] += TBJDS(mat)->val[offs] * invec->val[TBJDS(mat)->col[offs++]];
		}
		for (j=TBJDS(mat)->chunkMin[c]; j<TBJDS(mat)->rowLen[c*BJDS_LEN+2]; j++)
		{
			res->val[c*BJDS_LEN+2] += TBJDS(mat)->val[offs] * invec->val[TBJDS(mat)->col[offs++]];
		}
		for (j=TBJDS(mat)->chunkMin[c]; j<TBJDS(mat)->rowLen[c*BJDS_LEN+3]; j++)
		{
			res->val[c*BJDS_LEN+3] += TBJDS(mat)->val[offs] * invec->val[TBJDS(mat)->col[offs++]];
		}


		if (spmvmOptions & GHOST_OPTION_AXPY) {
			_mm256_store_pd(&res->val[c*BJDS_LEN],_mm256_add_pd(tmp,_mm256_load_pd(&res->val[c*BJDS_LEN])));
		} else {
			_mm256_stream_pd(&res->val[c*BJDS_LEN],tmp);
		}
	}
}

static void TBJDS_kernel_AVX_colwise(ghost_mat_t *mat, ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions)
{
	ghost_midx_t c,j;
	ghost_mnnz_t offs;
	__m256d tmp;
	__m256d val;
	__m256d rhs;
	__m128d rhstmp;

#pragma omp parallel for schedule(runtime) private(j,tmp,val,rhs,offs,rhstmp)
	for (c=0; c<TBJDS(mat)->nrowsPadded>>2; c++) 
	{ // loop over chunks
		tmp = _mm256_setzero_pd(); // tmp = 0
		offs = TBJDS(mat)->chunkStart[c];

		for (j=0; j<TBJDS(mat)->chunkMin[c]; j++) 
		{ // loop inside chunk
			val    = _mm256_load_pd(&TBJDS(mat)->val[offs]);                      // load values
			rhstmp = _mm_loadl_pd(rhstmp,&invec->val[(TBJDS(mat)->col[offs++])]); // load first 128 bits of RHS
			rhstmp = _mm_loadh_pd(rhstmp,&invec->val[(TBJDS(mat)->col[offs++])]);
			rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);                  // insert to RHS
			rhstmp = _mm_loadl_pd(rhstmp,&invec->val[(TBJDS(mat)->col[offs++])]); // load second 128 bits of RHS
			rhstmp = _mm_loadh_pd(rhstmp,&invec->val[(TBJDS(mat)->col[offs++])]);
			rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);                  // insert to RHS
			tmp    = _mm256_add_pd(tmp,_mm256_mul_pd(val,rhs));           // accumulate
		}

		for (j=TBJDS(mat)->chunkMin[c]; j<TBJDS(mat)->chunkLen[c]; j++)
		{
			if (j<TBJDS(mat)->rowLen[c*BJDS_LEN])
				res->val[c*BJDS_LEN] += TBJDS(mat)->val[offs] * invec->val[TBJDS(mat)->col[offs++]];
			if (j<TBJDS(mat)->rowLen[c*BJDS_LEN+1])
				res->val[c*BJDS_LEN+1] += TBJDS(mat)->val[offs] * invec->val[TBJDS(mat)->col[offs++]];
			if (j<TBJDS(mat)->rowLen[c*BJDS_LEN+2])
				res->val[c*BJDS_LEN+2] += TBJDS(mat)->val[offs] * invec->val[TBJDS(mat)->col[offs++]];
			if (j<TBJDS(mat)->rowLen[c*BJDS_LEN+3])
				res->val[c*BJDS_LEN+3] += TBJDS(mat)->val[offs] * invec->val[TBJDS(mat)->col[offs++]];
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
static void TBJDS_kernel_MIC_16(ghost_mat_t *mat, ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions)
{
	ghost_midx_t c,j;
	ghost_mnnz_t offs;
	unsigned short i;
	__m512d tmp1;
	__m512d tmp2;
	__m512d val;
	__m512d rhs;
	__m512i idx;

#pragma omp parallel for schedule(runtime) private(i,j,tmp1,tmp2,val,rhs,idx,offs)
	for (c=0; c<TBJDS(mat)->nrowsPadded>>4; c++) 
	{ // loop over chunks
		tmp1 = _mm512_setzero_pd(); // tmp1 = 0
		tmp2 = _mm512_setzero_pd(); // tmp2 = 0
		offs = TBJDS(mat)->chunkStart[c];

		for (j=0; j<TBJDS(mat)->chunkMin[c]; j++) 
		{ // loop inside chunk
			val = _mm512_load_pd(&TBJDS(mat)->val[offs]);
			idx = _mm512_load_epi32(&TBJDS(mat)->col[offs]);
			rhs = _mm512_i32logather_pd(idx,invec->val,8);
			tmp1 = _mm512_add_pd(tmp1,_mm512_mul_pd(val,rhs));

			offs += 8;

			val = _mm512_load_pd(&TBJDS(mat)->val[offs]);
			idx = _mm512_permute4f128_epi32(idx,_MM_PERM_BADC);
			rhs = _mm512_i32logather_pd(idx,invec->val,8);
			tmp2 = _mm512_add_pd(tmp2,_mm512_mul_pd(val,rhs));

			offs += 8;
		}

		for (i=0; i<16; i++)
		{
			for (j=TBJDS(mat)->chunkMin[c]; j<TBJDS(mat)->rowLen[c*BJDS_LEN+i]; j++)
			{
				res->val[c*BJDS_LEN+i] += TBJDS(mat)->val[offs] * invec->val[TBJDS(mat)->col[offs++]];
			}
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
