#include "spm_format_bjds.h"
#include "matricks.h"
#include "ghost_util.h"
#include "kernel.h"

static void BJDS_printInfo();
static char * BJDS_formatName();
static mat_idx_t BJDS_rowLen (mat_idx_t i);
static mat_data_t BJDS_entry (mat_idx_t i, mat_idx_t j);
static size_t BJDS_byteSize (void);
static void BJDS_fromCRS(CR_TYPE *cr, mat_trait_t traits);
static void BJDS_fromBin(char *, mat_trait_t traits);
static void BJDS_kernel_plain (ghost_vec_t * lhs, ghost_vec_t * rhs, int options);

static ghost_mat_t *thisMat;
static BJDS_TYPE *thisBJDS;

void BJDS_registerFunctions(ghost_mat_t *mat)
{
	thisMat = mat;

	thisBJDS = (BJDS_TYPE *)(thisMat->data);

	thisMat->fromBin = &BJDS_fromBin;
	thisMat->printInfo = &BJDS_printInfo;
	thisMat->formatName = &BJDS_formatName;
	thisMat->rowLen     = &BJDS_rowLen;
	thisMat->entry      = &BJDS_entry;
	thisMat->byteSize   = &BJDS_byteSize;
	thisMat->kernel     = &BJDS_kernel_plain;
}

static void BJDS_printInfo()
{
	SpMVM_printLine("Vector block size",NULL,"%d",BJDS_LEN);
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
	ghost_mat_t *crsMat = (ghost_mat_t *)allocateMemory(sizeof(ghost_mat_t),"matrix");
	mat_trait_t crsTraits = {.format = GHOST_SPMFORMAT_CRS,.flags=GHOST_SPM_DEFAULT,NULL};
	CRS_registerFunctions(crsMat);
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
	thisMat->trait = trait;
	thisMat->nrows = cr->nrows;
	thisMat->ncols = cr->ncols;
	thisMat->nnz = cr->nEnts;
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

		/* get max number of entries in one row ###########################*/
		rowSort = (JD_SORT_TYPE*) allocateMemory( cr->nrows * sizeof( JD_SORT_TYPE ),
				"rowSort" );

		for( i = 0; i < cr->nrows; i++ ) {
			rowSort[i].row = i;
			rowSort[i].nEntsInRow = cr->rpt[i+1] - cr->rpt[i];
		} 

		qsort( rowSort, cr->nrows, sizeof( JD_SORT_TYPE  ), compareNZEPerRow );

		/* sort within same rowlength with asceding row number #################### */
		i=0;
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
		}
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
	thisBJDS->rowLen = (mat_idx_t *)allocateMemory(thisBJDS->nrowsPadded*sizeof(mat_idx_t),"thisBJDS->rowLen");
	thisBJDS->chunkStart[0] = 0;

	mat_idx_t chunkMax = 0;
	mat_idx_t curChunk = 1;

	for (i=0; i<thisBJDS->nrows; i++) {

		if (flags & GHOST_SPM_SORTED)
			thisBJDS->rowLen[i] = rowSort[i].nEntsInRow;
		else
			thisBJDS->rowLen[i] = cr->rpt[i+1]-cr->rpt[i];
		chunkMax = thisBJDS->rowLen[i]>chunkMax?thisBJDS->rowLen[i]:chunkMax;
#ifdef MIC
		/* The gather instruction is only available on MIC. Therefore, the
		   access to the index vector has to be 512bit-aligned only on MIC.
		   Also, the innerloop in the BJDS-kernel has to be 2-way unrolled
		   only on this case. ==> The number of columns of one chunk does
		   not have to be a multiple of two in the other cases. */
		chunkMax = chunkMax%2==0?chunkMax:chunkMax+1;
#endif

		if ((i+1)%BJDS_LEN == 0) {
			thisBJDS->nEnts += BJDS_LEN*chunkMax;
			thisBJDS->chunkStart[curChunk] = thisBJDS->chunkStart[curChunk-1]+BJDS_LEN*chunkMax;

			chunkMax = 0;
			curChunk++;
		}
	}

	thisBJDS->val = (mat_data_t *)allocateMemory(sizeof(mat_data_t)*thisBJDS->nEnts,"thisBJDS->val");
	thisBJDS->col = (mat_idx_t *)allocateMemory(sizeof(mat_idx_t)*thisBJDS->nEnts,"thisBJDS->val");

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
		mat_idx_t chunkLen = (thisBJDS->chunkStart[c+1]-thisBJDS->chunkStart[c])/BJDS_LEN;

		for (j=0; j<chunkLen; j++) {

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
