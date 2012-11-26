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
static ghost_mdat_t ELLPACK_entry (ghost_mat_t *mat, mat_idx_t i, mat_idx_t j);
static size_t ELLPACK_byteSize (ghost_mat_t *mat);
static void ELLPACK_fromCRS(ghost_mat_t *mat, CR_TYPE *cr, ghost_mtraits_t traits);
static void ELLPACK_fromBin(ghost_mat_t *mat, char *, ghost_mtraits_t traits);
static void ELLPACK_free(ghost_mat_t *mat);
static void ELLPACK_kernel_plain (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);

void init(ghost_mat_t **mat)
{
	DEBUG_LOG(1,"Setting functions for ELLPACK matrix");

	(*mat)->fromBin = &ELLPACK_fromBin;
	(*mat)->printInfo = &ELLPACK_printInfo;
	(*mat)->formatName = &ELLPACK_formatName;
	(*mat)->rowLen     = &ELLPACK_rowLen;
	(*mat)->entry      = &ELLPACK_entry;
	(*mat)->byteSize   = &ELLPACK_byteSize;
	(*mat)->kernel     = &ELLPACK_kernel_plain;
	(*mat)->nnz      = &ELLPACK_nnz;
	(*mat)->nrows    = &ELLPACK_nrows;
	(*mat)->ncols    = &ELLPACK_ncols;
	(*mat)->destroy  = &ELLPACK_free;
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
	UNUSED(mat);
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

static ghost_mdat_t ELLPACK_entry (ghost_mat_t *mat, mat_idx_t i, mat_idx_t j)
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
			ELLPACK(mat)->nrowsPadded*ELLPACK(mat)->maxRowLen*(sizeof(mat_idx_t)+sizeof(ghost_mdat_t)));
}

static void ELLPACK_fromBin(ghost_mat_t *mat, char *matrixPath, ghost_mtraits_t traits)
{
	// TODO
	ghost_mat_t *crsMat = ghost_initMatrix("CRS");
	ghost_mtraits_t crsTraits = {.format = "CRS",.flags=GHOST_SPM_DEFAULT,NULL};
	crsMat->fromBin(crsMat,matrixPath,crsTraits);

	ELLPACK_fromCRS(mat,crsMat->data,traits);
}

static void ELLPACK_fromCRS(ghost_mat_t *mat, CR_TYPE *cr, ghost_mtraits_t trait)
{
	DEBUG_LOG(1,"Creating ELLPACK matrix");
	mat_idx_t i,j,c;
	unsigned int flags = trait.flags;

	mat_idx_t *rowPerm = NULL;
	mat_idx_t *invRowPerm = NULL;

	JD_SORT_TYPE* rowSort;

	mat->data = (ELLPACK_TYPE *)allocateMemory(sizeof(ELLPACK_TYPE),"ELLPACK(mat)");
	mat->trait = trait;
	mat->rowPerm = rowPerm;
	mat->invRowPerm = invRowPerm;

	ELLPACK(mat)->maxRowLen = 0;

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

			ELLPACK(mat)->maxRowLen = MAX(ELLPACK(mat)->maxRowLen,rowSort[0].nEntsInRow);
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
	} else {

		/* get max number of entries in one row ###########################*/
		rowSort = (JD_SORT_TYPE*) allocateMemory( cr->nrows * sizeof( JD_SORT_TYPE ),
				"rowSort" );

		for( i = 0; i < cr->nrows; i++ ) {
			rowSort[i].row = i;
			rowSort[i].nEntsInRow = 0;
		} 

		/* count entries per row ################################################## */
		for( i = 0; i < cr->nrows; i++) 
			rowSort[i].nEntsInRow = cr->rpt[i+1] - cr->rpt[i];

		/* sort rows with desceding number of NZEs ################################ */
		qsort( rowSort, cr->nrows, sizeof( JD_SORT_TYPE  ), compareNZEPerRow );

		ELLPACK(mat)->maxRowLen = rowSort[0].nEntsInRow;
	}





	ELLPACK(mat)->nrows = cr->nrows;
	ELLPACK(mat)->nnz = cr->nEnts;
	ELLPACK(mat)->nEnts = ELLPACK_PAD*ELLPACK(mat)->maxRowLen;
	ELLPACK(mat)->nrowsPadded = pad(ELLPACK(mat)->nrows,ELLPACK_PAD);
	ELLPACK(mat)->rowLen = (mat_idx_t *)allocateMemory(ELLPACK(mat)->nrowsPadded*sizeof(mat_idx_t),"rowLen");
	ELLPACK(mat)->col = (mat_idx_t *)allocateMemory(ELLPACK(mat)->nEnts*sizeof(mat_idx_t),"col");
	ELLPACK(mat)->val = (ghost_mdat_t *)allocateMemory(ELLPACK(mat)->nEnts*sizeof(ghost_mdat_t),"val");

	ELLPACK(mat)->rowLen = (mat_idx_t *)allocateMemory((ELLPACK(mat)->nrowsPadded)*sizeof(mat_idx_t),"ELLPACK(mat)->rowLen");

#pragma omp parallel for private(i)	
	for( i=0; i < ELLPACK(mat)->nrowsPadded; ++i) {
		for( j=0; j < ELLPACK(mat)->maxRowLen; ++j) {
			ELLPACK(mat)->col[i+j*ELLPACK(mat)->nrowsPadded] = 0;
			ELLPACK(mat)->val[i+j*ELLPACK(mat)->nrowsPadded] = 0.0;
		}
	}

	for( i=0; i < ELLPACK(mat)->nrowsPadded; ++i) {
		if (i<cr->nrows) {
			if (flags & GHOST_SPM_SORTED)
				ELLPACK(mat)->rowLen[i] = rowSort[i].nEntsInRow;
			else
				ELLPACK(mat)->rowLen[i] = cr->rpt[i+1]-cr->rpt[i];
		} else {
			ELLPACK(mat)->rowLen[i] = 0;
		}

		for( j=0; j < ELLPACK(mat)->rowLen[i]; ++j) {
			if (flags & GHOST_SPM_SORTED) {
				ELLPACK(mat)->val[i+j*ELLPACK(mat)->nrowsPadded] = cr->val[cr->rpt[invRowPerm[i]]+j];
				if (flags & GHOST_SPM_PERMUTECOLIDX)
					ELLPACK(mat)->col[i+j*ELLPACK(mat)->nrowsPadded] = rowPerm[cr->col[cr->rpt[invRowPerm[i]]+j]];
				else 
					ELLPACK(mat)->col[i+j*ELLPACK(mat)->nrowsPadded] = cr->col[cr->rpt[invRowPerm[i]]+j];
			} else {
				ELLPACK(mat)->val[i+j*ELLPACK(mat)->nrowsPadded] = cr->val[cr->rpt[i]+j];
				ELLPACK(mat)->col[i+j*ELLPACK(mat)->nrowsPadded] = cr->col[cr->rpt[i]+j];
			}
		}
	}

	free( rowSort );
	DEBUG_LOG(1,"Successfully created ELLPACK");
}

static void ELLPACK_free(ghost_mat_t *mat)
{
	free(ELLPACK(mat)->rowLen);
	free(ELLPACK(mat)->val);
	free(ELLPACK(mat)->col);
	
	free(mat->data);
	free(mat->rowPerm);
	free(mat->invRowPerm);

	free(mat);


}

static void ELLPACK_kernel_plain (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
	mat_idx_t j,i;
	ghost_mdat_t tmp; 

#pragma omp parallel for schedule(runtime) private(j,tmp,i)
	for( i=0; i < ELLPACK(mat)->nrows; ++i) {
		tmp = 0;
		for( j=0; j < ELLPACK(mat)->maxRowLen; ++j) {
			tmp += ELLPACK(mat)->val[i+j*ELLPACK(mat)->nrowsPadded] * rhs->val[ELLPACK(mat)->col[i+j*ELLPACK(mat)->nrowsPadded]];
		}
		if (options & GHOST_OPTION_AXPY)
			lhs->val[i] += tmp;
		else
			lhs->val[i] = tmp;
	}

}

