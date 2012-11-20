#include "spm_format_tbjds.h"
#include "matricks.h"
#include "ghost_util.h"
#include "kernel.h"

static void TBJDS_printInfo();
static char * TBJDS_formatName();
static mat_idx_t TBJDS_rowLen (mat_idx_t i);
static mat_data_t TBJDS_entry (mat_idx_t i, mat_idx_t j);
static size_t TBJDS_byteSize (void);
static void TBJDS_fromCRS(CR_TYPE *cr, mat_trait_t traits);
static void TBJDS_fromBin(char *, mat_trait_t traits);
static void TBJDS_kernel_plain (ghost_vec_t * lhs, ghost_vec_t * rhs, int options);

static ghost_mat_t *thisMat;
static TBJDS_TYPE *thisTBJDS;

void TBJDS_registerFunctions(ghost_mat_t *mat)
{
	DEBUG_LOG(1,"Setting functions for TBJDS matrix");
	thisMat = mat;
	thisTBJDS = (TBJDS_TYPE *)(mat->data);

	mat->printInfo = &TBJDS_printInfo;
	mat->formatName = &TBJDS_formatName;
	mat->rowLen   = &TBJDS_rowLen;
	mat->entry    = &TBJDS_entry;
	mat->byteSize = &TBJDS_byteSize;
	mat->kernel   = &TBJDS_kernel_plain;
}

static char * TBJDS_formatName()
{
	return "TBJDS";
}

static void TBJDS_printInfo()
{
	SpMVM_printLine("Vector block size",NULL,"%d",BJDS_LEN);
	SpMVM_printLine("Row length oscillation nu",NULL,"%f",thisTBJDS->nu);
	if (thisMat->trait.flags & GHOST_SPM_SORTED) {
		SpMVM_printLine("Sort block size",NULL,"%u",*(unsigned int *)(thisMat->trait.aux));
		SpMVM_printLine("Permuted columns",NULL,"%s",thisMat->trait.flags&GHOST_SPM_PERMUTECOLIDX?"yes":"no");
	}



}


static mat_idx_t TBJDS_rowLen (mat_idx_t i)
{
	if (thisMat->trait.flags & GHOST_SPM_SORTED)
		i = thisMat->rowPerm[i];

	return thisTBJDS->rowLen[i];
}

static mat_data_t TBJDS_entry (mat_idx_t i, mat_idx_t j)
{
	mat_idx_t e;
	if (thisMat->trait.flags & GHOST_SPM_SORTED)
		i = thisMat->rowPerm[i];
	if (thisMat->trait.flags & GHOST_SPM_PERMUTECOLIDX)
		j = thisMat->rowPerm[j];


	for (e=thisTBJDS->chunkStart[i/BJDS_LEN]+i%BJDS_LEN; 
			e<thisTBJDS->chunkStart[i/BJDS_LEN+1]; 
			e+=BJDS_LEN) {
		if (thisTBJDS->col[e] == j)
			return thisTBJDS->val[e];
	}
	return 0.;
}

static size_t TBJDS_byteSize (void)
{
	return (size_t)((thisTBJDS->nrowsPadded/BJDS_LEN)*sizeof(mat_nnz_t) + 
			thisTBJDS->nEnts*(sizeof(mat_idx_t)+sizeof(mat_data_t)));
}

static void TBJDS_fromBin(char *matrixPath, mat_trait_t traits)
{
	// TODO
	ghost_mat_t *crsMat = (ghost_mat_t *)allocateMemory(sizeof(ghost_mat_t),"matrix");
	mat_trait_t crsTraits = {.format = GHOST_SPMFORMAT_CRS,.flags=GHOST_SPM_DEFAULT,NULL};
	CRS_registerFunctions(crsMat);
	crsMat->fromBin(matrixPath,crsTraits);
	
	TBJDS_fromCRS(crsMat->data,traits);
}

static void TBJDS_fromCRS(CR_TYPE *cr, mat_trait_t trait)
{
	mat_idx_t i,j,c;
	TBJDS_TYPE *thisTBJDS;
	JD_SORT_TYPE* rowSort;
	mat_idx_t *rowPerm = NULL, *invRowPerm = NULL;
	unsigned int flags;

	thisTBJDS = (TBJDS_TYPE *)allocateMemory(sizeof(TBJDS_TYPE),"mv");
	thisMat->trait = trait;
	thisMat->nrows = cr->nrows;
	thisMat->ncols = cr->ncols;
	thisMat->nnz = cr->nEnts;
	thisMat->rowPerm = rowPerm;
	thisMat->invRowPerm = invRowPerm;

	thisTBJDS->nrows = cr->nrows;
	thisTBJDS->nnz = cr->nEnts;
	thisTBJDS->nEnts = 0;
	thisTBJDS->nrowsPadded = pad(thisTBJDS->nrows,BJDS_LEN);
	mat_idx_t nChunks = thisTBJDS->nrowsPadded/BJDS_LEN;
		thisTBJDS->chunkStart = (mat_nnz_t *)allocateMemory((nChunks+1)*sizeof(mat_nnz_t),"thisTBJDS->chunkStart");
		thisTBJDS->chunkMin = (mat_idx_t *)allocateMemory((nChunks)*sizeof(mat_idx_t),"thisTBJDS->chunkMin");
		thisTBJDS->chunkLen = (mat_idx_t *)allocateMemory((nChunks)*sizeof(mat_idx_t),"thisTBJDS->chunkMin");
		thisTBJDS->rowLen = (mat_idx_t *)allocateMemory((thisTBJDS->nrowsPadded)*sizeof(mat_idx_t),"thisTBJDS->chunkMin");
		thisTBJDS->chunkStart[0] = 0;
	flags = trait.flags;
	
	if (trait.flags & GHOST_SPM_SORTED) {
		rowPerm = (mat_idx_t *)allocateMemory(cr->nrows*sizeof(mat_idx_t),"sbjds->rowPerm");
		invRowPerm = (mat_idx_t *)allocateMemory(cr->nrows*sizeof(mat_idx_t),"sbjds->invRowPerm");

		(thisMat)->rowPerm = rowPerm;
		(thisMat)->invRowPerm = invRowPerm;

		unsigned int sortBlock = *(unsigned int *)(trait.aux);

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

	//	for(i=0; i < cr->nrows; ++i) printf("%d\n",(*invRowPerm)[i]);

	mat_idx_t chunkMin = cr->ncols;
	mat_idx_t chunkLen = 0;
	mat_idx_t curChunk = 1;
	thisTBJDS->nu = 0.;

	for (i=0; i<thisTBJDS->nrowsPadded; i++) {
		if (i<cr->nrows) {
			if (flags & GHOST_SPM_SORTED)
				thisTBJDS->rowLen[i] = rowSort[i].nEntsInRow;
			else
				thisTBJDS->rowLen[i] = cr->rpt[i+1]-cr->rpt[i];
		} else {
			thisTBJDS->rowLen[i] = 0;
		}

		thisTBJDS->nEnts += thisTBJDS->rowLen[i];

		chunkMin = thisTBJDS->rowLen[i]<chunkMin?thisTBJDS->rowLen[i]:chunkMin;
		chunkLen = thisTBJDS->rowLen[i]>chunkLen?thisTBJDS->rowLen[i]:chunkLen;

		if ((i+1)%BJDS_LEN == 0) {
			thisTBJDS->chunkStart[curChunk] = thisTBJDS->nEnts;
			thisTBJDS->chunkMin[curChunk-1] = chunkMin;
			thisTBJDS->chunkLen[curChunk-1] = chunkLen;

			thisTBJDS->nu += (double)chunkMin/chunkLen;

			chunkMin = cr->ncols;
			chunkLen = 0;
			curChunk++;
		}
	}
	thisTBJDS->nu /= (double)nChunks;

	thisTBJDS->val = (mat_data_t *)allocateMemory(sizeof(mat_data_t)*thisTBJDS->nEnts,"thisTBJDS->val");
	thisTBJDS->col = (mat_idx_t *)allocateMemory(sizeof(mat_idx_t)*thisTBJDS->nEnts,"thisTBJDS->col");

	//printf("nEnts: %d\n",thisTBJDS->nEnts);

#pragma omp parallel for schedule(runtime) private(j,i)
	for (c=0; c<thisTBJDS->nrowsPadded/BJDS_LEN; c++) 
	{ // loop over chunks

		for (j=0; j<thisTBJDS->chunkMin[c]; j++)
		{
			for (i=0; i<BJDS_LEN; i++)
			{
				thisTBJDS->val[thisTBJDS->chunkStart[c]+j*BJDS_LEN+i] = 0.;
				thisTBJDS->col[thisTBJDS->chunkStart[c]+j*BJDS_LEN+i] = 0;
			}
		}
		mat_nnz_t rem = thisTBJDS->chunkStart[c] + thisTBJDS->chunkMin[c]*BJDS_LEN;
		for (i=0; i<BJDS_LEN; i++)
		{
			for (j=thisTBJDS->chunkMin[c]; j<thisTBJDS->rowLen[c*BJDS_LEN+i]; j++)
			{
				thisTBJDS->val[rem] = 0.;
				thisTBJDS->col[rem++] = 0;
			}
		}
	}
	for (c=0; c<thisTBJDS->nrowsPadded/BJDS_LEN; c++) 
	{ // loop over chunks

		// store block
		for (j=0; j<thisTBJDS->chunkMin[c]; j++)
		{
			for (i=0; i<BJDS_LEN; i++)
			{
				if (flags & GHOST_SPM_SORTED) {
					thisTBJDS->val[thisTBJDS->chunkStart[c]+j*BJDS_LEN+i] = cr->val[cr->rpt[(invRowPerm)[c*BJDS_LEN+i]]+j];
					if (flags & GHOST_SPM_PERMUTECOLIDX)
						thisTBJDS->col[thisTBJDS->chunkStart[c]+j*BJDS_LEN+i] = (rowPerm)[cr->col[cr->rpt[(invRowPerm)[c*BJDS_LEN+i]]+j]];
					else
						thisTBJDS->col[thisTBJDS->chunkStart[c]+j*BJDS_LEN+i] = cr->col[cr->rpt[(invRowPerm)[c*BJDS_LEN+i]]+j];
				} else {
					thisTBJDS->val[thisTBJDS->chunkStart[c]+j*BJDS_LEN+i] = cr->val[cr->rpt[c*BJDS_LEN+i]+j];
					thisTBJDS->col[thisTBJDS->chunkStart[c]+j*BJDS_LEN+i] = cr->col[cr->rpt[c*BJDS_LEN+i]+j];
				}
			}
		}

		// store remainder
		mat_nnz_t rem = thisTBJDS->chunkStart[c] + thisTBJDS->chunkMin[c]*BJDS_LEN;
		if (flags & GHOST_SPM_COLMAJOR) 
		{
			for (j=thisTBJDS->chunkMin[c]; j<thisTBJDS->chunkLen[c]; j++)
			{
				for (i=0; i<BJDS_LEN; i++)
				{
					if (j<thisTBJDS->rowLen[c*BJDS_LEN+i] ) {
						if (flags & GHOST_SPM_SORTED) {
							thisTBJDS->val[rem] = cr->val[cr->rpt[(invRowPerm)[c*BJDS_LEN+i]]+j];
							if (flags & GHOST_SPM_PERMUTECOLIDX)
								thisTBJDS->col[rem++] = (rowPerm)[cr->col[cr->rpt[(invRowPerm)[c*BJDS_LEN+i]]+j]];
							else
								thisTBJDS->col[rem++] = cr->col[cr->rpt[(invRowPerm)[c*BJDS_LEN+i]]+j];
						} else {
							thisTBJDS->val[rem] = cr->val[cr->rpt[c*BJDS_LEN+i]+j];
							thisTBJDS->col[rem++] = cr->col[cr->rpt[c*BJDS_LEN+i]+j];
						}
					}
				}
			}
		} else // row major is the default 
		{
			for (i=0; i<BJDS_LEN; i++)
			{
				for (j=thisTBJDS->chunkMin[c]; j<thisTBJDS->rowLen[c*BJDS_LEN+i]; j++)
				{
					if (flags & GHOST_SPM_SORTED) {
						thisTBJDS->val[rem] = cr->val[cr->rpt[(invRowPerm)[c*BJDS_LEN+i]]+j];
						if (flags & GHOST_SPM_PERMUTECOLIDX)
							thisTBJDS->col[rem++] = (rowPerm)[cr->col[cr->rpt[(invRowPerm)[c*BJDS_LEN+i]]+j]];
						else
							thisTBJDS->col[rem++] = cr->col[cr->rpt[(invRowPerm)[c*BJDS_LEN+i]]+j];
					} else {
						thisTBJDS->val[rem] = cr->val[cr->rpt[c*BJDS_LEN+i]+j];
						thisTBJDS->col[rem++] = cr->col[cr->rpt[c*BJDS_LEN+i]+j];
					}
				}
			}
		}
	}
}

static void TBJDS_kernel_plain (ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
	DEBUG_LOG(2,"Calling plain TBJDS kernel");
	mat_idx_t c,j,i;
	mat_nnz_t offs;
	mat_data_t tmp[BJDS_LEN]; 

#pragma omp parallel for schedule(runtime) private(j,tmp,i,offs)
	for (c=0; c<thisTBJDS->nrowsPadded/BJDS_LEN; c++) 
	{ // loop over chunks

		offs = thisTBJDS->chunkStart[c];
		for (i=0; i<BJDS_LEN; i++)
		{
			tmp[i] = 0;
		}

		for (j=0; j<thisTBJDS->chunkMin[c]; j++) 
		{ // loop inside chunk
			for (i=0; i<BJDS_LEN; i++)
			{
				tmp[i] += thisTBJDS->val[offs] * rhs->val[thisTBJDS->col[offs++]];
			}

		}
		for (i=0; i<BJDS_LEN; i++)
		{
			for (j=thisTBJDS->chunkMin[c]; j<thisTBJDS->rowLen[c*BJDS_LEN+i]; j++)
			{
				tmp[i] += thisTBJDS->val[offs] * rhs->val[thisTBJDS->col[offs++]];
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
