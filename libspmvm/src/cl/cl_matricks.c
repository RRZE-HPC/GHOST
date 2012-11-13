#include <mpi.h>
#include "cl_matricks.h"
#include "matricks.h"



#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


size_t getBytesize(void *mat, int format) {
	size_t sz = 0;
	switch (format) {
		case SPM_GPUFORMAT_PJDS:
			{
				CL_PJDS_TYPE * matrix = (CL_PJDS_TYPE *)mat;
				sz = matrix->nEnts*(sizeof(mat_data_t)+sizeof(int)) + matrix->nrows*sizeof(int) + matrix->nMaxRow*sizeof(int);
				break;
			}
		case SPM_GPUFORMAT_ELR:
			{
				CL_ELR_TYPE * matrix = (CL_ELR_TYPE *)mat;
				sz = matrix->nMaxRow * matrix->padding*(sizeof(mat_data_t)+sizeof(int)) + (matrix->nrows*sizeof(int));
				break;
			}
		default:
				SpMVM_abort("Invalid device matrix format in getBytesize!");
	}

	return sz;
}


/**********************  pJDS MATRIX TYPE *********************************/

PJDS_TYPE* CRStoPJDST(  const mat_data_t* crs_val, const int* crs_col, 
		const int* crs_row_ptr, const int nrows, const int threadsPerRow) 
{
	ELR_TYPE * elrs;
	elrs = CRStoELRS(crs_val, crs_col, crs_row_ptr, nrows);
	PJDS_TYPE * pjds;
	pjds = ELRStoPJDST(elrs,threadsPerRow);

	freeELR(elrs);
	return pjds;
}

PJDS_TYPE* CRStoPJDS(  const mat_data_t* crs_val, const int* crs_col, 
		const int* crs_row_ptr, const int nrows) 
{
	return CRStoPJDST( crs_val, crs_col, crs_row_ptr, nrows, 1); 
}

PJDS_TYPE* ELRStoPJDST( const ELR_TYPE* elr, int threadsPerRow )
{

	PJDS_TYPE *pjds = NULL;
	int  i,j,st;

	if (elr->padding % PJDS_CHUNK_HEIGHT != 0) {
		printf("ELR matrix cannot be divided into chunks.\n");
		exit(EXIT_FAILURE);
	}

	pjds = (PJDS_TYPE*) allocateMemory(sizeof(PJDS_TYPE),"pjds");

	// initialize pjds entries
	pjds->rowPerm = (int *)allocateMemory(elr->nrows*sizeof(int),"pjds->rowPerm");	
	pjds->invRowPerm = (int *)allocateMemory(elr->nrows*sizeof(int),"pjds->invRowPerm");	
	pjds->padding = elr->padding;
	pjds->nrows = elr->nrows;
	pjds->nMaxRow = elr->nMaxRow;
	int * chunkLen = (int*) allocateMemory((int)sizeof(int)*elr->padding/PJDS_CHUNK_HEIGHT,"chunkLen");
	pjds->rowLen = (int*) allocateMemory((int)sizeof(int)*elr->nrows,"pjds->rowLen");
	pjds->nEnts = 0;
	pjds->T = threadsPerRow;

	memcpy(pjds->rowPerm,elr->rowPerm,elr->nrows*sizeof(int));
	memcpy(pjds->invRowPerm,elr->invRowPerm,elr->nrows*sizeof(int));

	for (i=0; i<pjds->nrows; i++) {
		pjds->rowLen[i] = elr->rowLen[i];
		if (pjds->rowLen[i]%threadsPerRow != 0)
			pjds->rowLen[i] += (threadsPerRow-pjds->rowLen[i]%threadsPerRow);
	}


	int *colHeight = (int *)allocateMemory(sizeof(int)*pjds->rowLen[0],"colHeight");
	int curCol = pjds->rowLen[0]-1;
	int curChunk = 0;

	for (i=0; i<pjds->padding; i++) 
	{

		// save chunkLen and update nEnts
		if (i%PJDS_CHUNK_HEIGHT == 0) 
		{
			chunkLen[curChunk] = pjds->rowLen[i>=pjds->nrows?pjds->nrows-1:i];
			pjds->nEnts += PJDS_CHUNK_HEIGHT*chunkLen[curChunk];

			// if a step occurs save the column heights
			if (curChunk != 0 && chunkLen[curChunk] != chunkLen[curChunk-1]) 
			{
				for (st=0; st<chunkLen[curChunk-1]-chunkLen[curChunk]; st++) // count all cols
				{ 
					colHeight[curCol] = i;
					curCol--;
				}
			}
			curChunk++;
		}

	}

	// collect all columns with maximal height
	while(curCol >= 0) {
		colHeight[curCol] = pjds->padding;
		curCol--;
	}


	pjds->val = (mat_data_t*) allocateMemory(sizeof(mat_data_t)*pjds->nEnts,"pjds->val"); 
	pjds->colStart = (int*) allocateMemory(sizeof(int)*(chunkLen[0]),"pjds->colStart");
	pjds->col = (int*) allocateMemory(sizeof(int)*pjds->nEnts,"pjds->col");

	for( i=0; i < pjds->nrows; ++i) {
		pjds->rowLen[i] /= threadsPerRow; 
		if (i%PJDS_CHUNK_HEIGHT == 0)
			chunkLen[i/PJDS_CHUNK_HEIGHT] /= threadsPerRow;
	}

	pjds->colStart[0] = 0; // initial colStart is zero
	//printf("colStart[0]: %d\n",0);
	for (j=1; j<chunkLen[0]; j++) // save all other colStart
	{
		pjds->colStart[j] = pjds->colStart[j-1]+threadsPerRow*colHeight[j*threadsPerRow-1];
		//printf("colStart[%d]: %d\n",j,pjds->colStart[j]);
	}

	// check for sanity
	//assert(pjds->colStart[chunkLen[0]-1] + threadsPerRow*colHeight[(chunkLen[0]-1)] == pjds->nEnts);

	for (i=0; i<pjds->nEnts; i++) pjds->val[i] = 0.0;
	for (i=0; i<pjds->nEnts; i++) pjds->col[i] = 0;


	int idb, idx;
	// copy col and val from elr-s to pjds
	for (j=0; j<elr->rowLen[0]; j++) 
	{
		for (i=0; i<colHeight[j]; i++) 
		{
			idb = j%threadsPerRow;
			idx = pjds->colStart[j/threadsPerRow]+i*threadsPerRow+idb;
			pjds->val[idx] = elr->val[j*elr->padding+i];
			pjds->col[idx] = elr->col[j*elr->padding+i];
		}	
	}

	free(colHeight);
	return pjds;
}

ELR_TYPE* CRStoELRT(const mat_data_t* crs_val, const int* crs_col, 
		const int* crs_row_ptr, const int nrows, int threadsPerRow) {

	int i, j;
	int idb,stack;
	ELR_TYPE* elr = NULL;

	elr = (ELR_TYPE *) allocateMemory(sizeof(ELR_TYPE ),"elr");
	elr->nrows       = nrows;
	elr->padding = pad(nrows,ELR_PADDING);

	elr->nMaxRow   = 0;
	for (i=0; i<nrows; ++i) { 
		elr->nMaxRow = (elr->nMaxRow > crs_row_ptr[i+1]-crs_row_ptr[i])?
			elr->nMaxRow:crs_row_ptr[i+1]-crs_row_ptr[i];
	}

	if (elr->nMaxRow%threadsPerRow != 0)
		elr->nMaxRow += threadsPerRow-elr->nMaxRow%threadsPerRow;

	elr->rowLen = (int*) allocateMemory(sizeof(int)*elr->nrows,"elr->rowLen"); 
	elr->col = (int*) allocateMemory(sizeof(int)*elr->padding*elr->nMaxRow,
			"elr->col"); 
	elr->val = (mat_data_t*)allocateMemory(sizeof(mat_data_t)*elr->padding*elr->nMaxRow,
			"elr->val"); 
	elr->T = threadsPerRow;


	// initialize
	for( j=0; j < elr->nMaxRow; ++j) {
		for( i=0; i < elr->padding; ++i) {
			elr->col[i+j*elr->padding] = 0;
			elr->val[i+j*elr->padding] = 0.0;
		}
	}

	// copy row lenghts
	for( i=0; i < elr->nrows; ++i) {
		elr->rowLen[i] = crs_row_ptr[i+1]-crs_row_ptr[i];
	}

	// copy values and column indices
	for( i = 0; i < elr->nrows; ++i) {
		for( j = 0; j < elr->rowLen[i]; ++j) {
			idb = j%threadsPerRow;
			stack = j/threadsPerRow;
			elr->col[stack*threadsPerRow*elr->padding + threadsPerRow*i + idb] =
			   	crs_col[ crs_row_ptr[i]+j ];
			elr->val[stack*threadsPerRow*elr->padding + threadsPerRow*i + idb] =
			   	crs_val[ crs_row_ptr[i]+j ];
		}
	}


	// pad row lenghts
	for( i=0; i < elr->nrows; ++i) {
		if (elr->rowLen[i]%threadsPerRow != 0)
			elr->rowLen[i] += threadsPerRow-elr->rowLen[i]%threadsPerRow;
		elr->rowLen[i] /= threadsPerRow; 
	}

	return elr;
}

ELR_TYPE* CRStoELRTP(const mat_data_t* crs_val, const int* crs_col, 
		const int* crs_row_ptr, const int nrows,  
		const int* invRowPerm, int threadsPerRow) 
{

	int i, j;
	ELR_TYPE* elr = NULL;

	elr = (ELR_TYPE *) allocateMemory(sizeof(ELR_TYPE ),"elr");
	elr->nrows       = nrows;
	elr->padding = pad(nrows,ELR_PADDING);

	elr->nMaxRow   = 0;
	for (i=0; i<nrows; ++i) 
		elr->nMaxRow = (elr->nMaxRow > crs_row_ptr[i+1]-crs_row_ptr[i])?elr->nMaxRow:crs_row_ptr[i+1]-crs_row_ptr[i];

	if (elr->nMaxRow%threadsPerRow != 0)
		elr->nMaxRow += threadsPerRow-elr->nMaxRow%threadsPerRow;

	elr->rowLen      = (int*) allocateMemory(sizeof(int)*elr->nrows,"elr->rowLen"); 
	elr->col         = (int*) allocateMemory(sizeof(int)*elr->padding*elr->nMaxRow,"elr->col"); 
	elr->val         = (mat_data_t*)allocateMemory(sizeof(mat_data_t)*elr->padding*elr->nMaxRow,"elr->val"); 
	elr->T			 = threadsPerRow;


	for( j=0; j < elr->nMaxRow; ++j) {
		for( i=0; i < elr->padding; ++i) {
			elr->col[i+j*elr->padding] = 0;
			elr->val[i+j*elr->padding] = 0.0;
		}
	}


	int idb,stack,idx;

	for( i = 0; i < elr->nrows; ++i) {
		elr->rowLen[i] = crs_row_ptr[invRowPerm[i]+1]-crs_row_ptr[invRowPerm[i]];
		for( j = 0; j < elr->rowLen[i]; ++j) {

			idb = j%threadsPerRow;
			stack = j/threadsPerRow;
			idx = stack*threadsPerRow*elr->padding + threadsPerRow*i + idb;
//			if (SPMVM_OPTIONS & SPMVM_OPTION_PERMCOLS)
//				elr->col[idx] = rowPerm[crs_col[crs_row_ptr[invRowPerm[i]]+j]];
//			else
				elr->col[idx] = crs_col[ crs_row_ptr[invRowPerm[i]]+j ];
			elr->val[idx] = crs_val[ crs_row_ptr[invRowPerm[i]]+j ];
		}
	}

	for( i=0; i < elr->nrows; ++i) {
		if (elr->rowLen[i]%threadsPerRow != 0)
			elr->rowLen[i] += threadsPerRow-elr->rowLen[i]%threadsPerRow;
		elr->rowLen[i] /= threadsPerRow; 
	}

	return elr;
}

ELR_TYPE* CRStoELRP(  const mat_data_t* crs_val, const int* crs_col, 
		const int* crs_row_ptr, const int nrows, const int* invRowPerm) {


	JD_SORT_TYPE* rowSort;
	int i, j, rowMaxEnt, padRows;
	size_t size_val, size_col, size_rowlen;
	int *rowLen, *col;
	mat_data_t* val;
	ELR_TYPE* elr = NULL;

	/* get max number of entries in one row ###########################*/
	rowSort = (JD_SORT_TYPE*) allocateMemory( nrows * sizeof( JD_SORT_TYPE ),
			"rowSort" );

	for( i = 0; i < nrows; i++ ) {
		rowSort[i].row = i;
		rowSort[i].nEntsInRow = 0;
	} 

	/* count entries per row ################################################## */
	for( i = 0; i < nrows; i++) {
		rowSort[i].nEntsInRow = crs_row_ptr[i+1] - crs_row_ptr[i];
		//IF_DEBUG(1) printf("row: %d, nEnts: %d\n",i,rowSort[i].nEntsInRow);
	}

	IF_DEBUG(2) {
		i=0;
		while(i < nrows) {
			int start = i;

			j = rowSort[start].nEntsInRow;
			while( i<nrows && rowSort[i].nEntsInRow == j ) ++i;

			if( (i-start)%5 != 0 || j%5 != 0 )
				printf("%i rows (%i): %i - %i\n",i-start,j, start, i-1);

		}
	}
	/* sort rows with desceding number of NZEs ################################ */
	qsort( rowSort, nrows, sizeof( JD_SORT_TYPE  ), compareNZEPerRow );
	rowMaxEnt = rowSort[0].nEntsInRow;

	/* allocate memory ################################################*/
	elr = (ELR_TYPE*) allocateMemory( sizeof( ELR_TYPE ), "elr_sorted");
	padRows = nrows;
	//#ifdef PADDING
	padRows = pad(nrows, ELR_PADDING);
	IF_DEBUG(1)  printf("convertCRS to ELR: padding: \t nrows=%i to %i\n", nrows, padRows);
	//#endif

	size_val    = (size_t) sizeof(mat_data_t) * padRows * rowMaxEnt;
	size_col    = (size_t) sizeof(int) * padRows * rowMaxEnt;
	size_rowlen = (size_t) sizeof(int) * nrows;

	rowLen = (int*)   allocateMemory( size_rowlen, "elrp->rowLen" ); 
	col   = (int*)    allocateMemory( size_col, "elrp->vol" ); 
	val   = (mat_data_t*)   allocateMemory( size_val,"elrp->val" ); 

	/* initialize values ########################################### */
	elr->rowLen = rowLen;
	elr->col = col;
	elr->val = val;
	elr->nrows  = nrows;
	elr->nMaxRow = rowMaxEnt;
	elr->padding = padRows;

	for( i = 0; i < nrows; ++i) {
		/* i runs in the permuted index, access to crs needs to be original index */
		/* RHS is also permuted to sorted system */
		elr->rowLen[i] = crs_row_ptr[invRowPerm[i]+1] - crs_row_ptr[invRowPerm[i]];
		for( j = 0; j < elr->rowLen[i]; ++j) {

			if( j*padRows+i >= elr->nMaxRow*padRows ) 
				printf("error: in i=%i, j=%i\n",i,j);

			//elr->col[ j*padRows+i ]   = rowPerm[crs_col[ crs_row_ptr[invRowPerm[i]]+j ]]; //PERMCOLS
	//		if (SPMVM_OPTIONS & SPMVM_OPTION_PERMCOLS)
	//			elr->col[ j*padRows+i ]   = rowPerm[crs_col[ crs_row_ptr[invRowPerm[i]]+j ]];
	//		else
				elr->col[ j*padRows+i ]   = crs_col[ crs_row_ptr[invRowPerm[i]]+j ];
			elr->val[ j*padRows+i ]   = crs_val[ crs_row_ptr[invRowPerm[i]]+j ];
		}
	}

	return elr;

}

ELR_TYPE* CRStoELRS(  const mat_data_t* crs_val, const int* crs_col, 
		const int* crs_row_ptr, const int nrows) {

	JD_SORT_TYPE* rowSort;
	int i, j, rowMaxEnt, padRows;
	size_t size_val, size_col, size_rowlen, size_rowperm;
	int *rowLen, *col;
	mat_data_t* val;
	ELR_TYPE* elr = NULL;

	/* get max number of entries in one row ###########################*/
	rowSort = (JD_SORT_TYPE*) allocateMemory( nrows * sizeof( JD_SORT_TYPE ),
			"rowSort" );

	for( i = 0; i < nrows; i++ ) {
		rowSort[i].row = i;
		rowSort[i].nEntsInRow = 0;
	} 

	/* count entries per row ################################################## */
	for( i = 0; i < nrows; i++) {
		rowSort[i].nEntsInRow = crs_row_ptr[i+1] - crs_row_ptr[i];
		//IF_DEBUG(1) printf("row: %d, nEnts: %d\n",i,rowSort[i].nEntsInRow);
	}

	IF_DEBUG(2) {
		i=0;
		while(i < nrows) {
			int start = i;

			j = rowSort[start].nEntsInRow;
			while( i<nrows && rowSort[i].nEntsInRow == j ) ++i;

			if( (i-start)%5 != 0 || j%5 != 0 )
				printf("%i rows (%i): %i - %i\n",i-start,j, start, i-1);

		}
	}
	/* sort rows with desceding number of NZEs ################################ */
	qsort( rowSort, nrows, sizeof( JD_SORT_TYPE  ), compareNZEPerRow );
	rowMaxEnt = rowSort[0].nEntsInRow;

	/* sort within same rowlength with asceding row number #################### */
	i=0;
	while(i < nrows) {
		int start = i;

		j = rowSort[start].nEntsInRow;
		while( i<nrows && rowSort[i].nEntsInRow >= j ) 
			++i;

		IF_DEBUG(1) printf("sorting over %i rows (%i): %i - %i\n",i-start,j, start, i-1);
		qsort( &rowSort[start], i-start, sizeof(JD_SORT_TYPE), compareNZEOrgPos );
	}

	for(i=1; i < nrows; ++i) {
		if( rowSort[i].nEntsInRow == rowSort[i-1].nEntsInRow && rowSort[i].row < rowSort[i-1].row)
			printf("Error in row %i: descending row number\n",i);
	}


	/* allocate memory ################################################*/
	elr = (ELR_TYPE*) allocateMemory( sizeof( ELR_TYPE ), "elr_sorted");
	padRows = nrows;
	//#ifdef PADDING
	padRows = pad(nrows, ELR_PADDING);
	IF_DEBUG(1)  printf("convertCRS to ELR: padding: \t nrows=%i to %i\n", nrows, padRows);
	//#endif

	size_val    = (size_t) sizeof(mat_data_t) * padRows * rowMaxEnt;
	size_col    = (size_t) sizeof(int) * padRows * rowMaxEnt;
	size_rowlen = (size_t) sizeof(int) * nrows;
	size_rowperm= (size_t) sizeof(int) * nrows;
	elr->rowPerm = (int *) allocateMemory(size_rowperm,"elr->rowPerm");
	elr->invRowPerm = (int *)allocateMemory(size_rowperm,"elr->invRowPerm");

	/* get the permutation indices ############################################ */
	for(i=0; i < nrows; ++i) {
		/* invRowPerm maps an index in the permuted system to the original index,
		 * rowPerm gets the original index and returns the corresponding permuted position.
		 */
		if( rowSort[i].row >= nrows ) printf("error: invalid row number %i in %i\n",rowSort[i].row, i); fflush(stdout);

		elr->invRowPerm[i] = rowSort[i].row;
		elr->rowPerm[rowSort[i].row] = i;
	}

	rowLen = (int*)   allocateMemory( size_rowlen,"elr->rowLen" ); 
	col   = (int*)    allocateMemory( size_col,"elr->col" ); 
	val   = (mat_data_t*)   allocateMemory( size_val,"elr->val" ); 

	/* initialize values ########################################### */
	elr->rowLen = rowLen;
	elr->col = col;
	elr->val = val;
	elr->nrows  = nrows;
	elr->nMaxRow = rowMaxEnt;
	elr->padding = padRows;

	/* fill with zeros ############################################ */
	for( j=0; j < elr->nMaxRow; ++j) {
		for( i=0; i < padRows; ++i) {
			elr->col[i+j*padRows] = 0;
			elr->val[i+j*padRows] = 0.0;
		}
	}

	for( i=0; i < elr->nrows; ++i) {
		elr->rowLen[i] = rowSort[i].nEntsInRow;
		/* should be equivalent to: */
		//elr->rowLen[rowPerm[i]] = crs_row_ptr[i+1] - crs_row_ptr[i];
	}

	/* copy values ################################################ */
	for( i = 0; i < nrows; ++i) {
		/* i runs in the permuted index, access to crs needs to be original index */
		/* RHS is also permuted to sorted system */
		for( j = 0; j < elr->rowLen[i]; ++j) {

			if( j*padRows+i >= elr->nMaxRow*padRows ) 
				printf("error: in i=%i, j=%i\n",i,j);

			//elr->col[ j*padRows+i ]   = rowPerm[crs_col[ crs_row_ptr[invRowPerm[i]]+j ]];
			// XXX: columns are NOT being permuted!

		//	if (SPMVM_OPTIONS & SPMVM_OPTION_PERMCOLS)
		//		elr->col[ j*padRows+i ]   = elr->rowPerm[crs_col[ crs_row_ptr[elr->invRowPerm[i]]+j ]];
		//	else
				elr->col[ j*padRows+i ]   = crs_col[ crs_row_ptr[elr->invRowPerm[i]]+j ];
			elr->val[ j*padRows+i ]   = crs_val[ crs_row_ptr[elr->invRowPerm[i]]+j ];
		}
	}

	/* access to both RHS and Res need to be changed, but Result also needs
	 * to be reverted back after MVM.
	 */
	j=0;

	for(i=1; i<nrows; ++i)
		if(rowSort[i].row == rowSort[i-1].row+1) 
			j++;

	IF_DEBUG(1)
	{
		printf("coherency: %2f\n", 100.0*j/nrows);

		for(i=0; i <MIN(10,nrows); ++i) {
			printf("row %i (len: %d): ",i,rowLen[i]);
			for(j=0; j<MIN(10,rowMaxEnt) && j< elr->rowLen[i]; ++j) {
				printf("%d: %.2f+%.2fi ",elr->col[j*padRows+i],REAL(elr->val[j*padRows+i]),IMAG(elr->val[j*padRows+i]));
			}
			printf("\n");
		}
	}

	free(rowSort);

	return elr;
}


/*void checkCRStToPJDS(const mat_data_t* crs_val, const int* crs_col, 
		const int* crs_row_ptr, const int nrows,
		const PJDS_TYPE* pjds) {


}*/

ELR_TYPE* CRStoELR(  const mat_data_t* crs_val, const int* crs_col, 
		const int* crs_row_ptr, const int nrows) {

	/* allocate and fill ELR-format matrix from CRS format data;
	 * elements in row retain CRS order (usually sorted by column);*/

	JD_SORT_TYPE* rowSort;
	int i, j, rowMaxEnt, padRows;
	size_t size_val, size_col, size_rowlen;
	int *rowLen, *col;
	mat_data_t* val;
	ELR_TYPE* elr = NULL;

	/* get max number of entries in one row ###########################*/
	rowSort = (JD_SORT_TYPE*) allocateMemory( nrows * sizeof( JD_SORT_TYPE ),
			"rowSort" );

	for( i = 0; i < nrows; i++ ) {
		rowSort[i].row = i;
		rowSort[i].nEntsInRow = 0;
	} 

	/* count entries per row ################################################## */
	for( i = 0; i < nrows; i++) 
		rowSort[i].nEntsInRow = crs_row_ptr[i+1] - crs_row_ptr[i];

	/* sort rows with desceding number of NZEs ################################ */
	qsort( rowSort, nrows, sizeof( JD_SORT_TYPE  ), compareNZEPerRow );

	rowMaxEnt = rowSort[0].nEntsInRow;
	free( rowSort );

	/* allocate memory ################################################*/
	elr = (ELR_TYPE*) allocateMemory( sizeof( ELR_TYPE ), "elr");
	padRows = nrows;
	//#ifdef PADDING
	padRows = pad(nrows, ELR_PADDING);
	IF_DEBUG(1)  printf("convertCRS to ELR: padding: \t nrows=%i to %i\n", nrows, padRows);
	//#endif

	size_val    = (size_t) sizeof(mat_data_t) * padRows * rowMaxEnt;
	size_col    = (size_t) sizeof(int) * padRows * rowMaxEnt;
	size_rowlen = (size_t) sizeof(int) * nrows;

	rowLen = (int*)   allocateMemory( size_rowlen,"elr->rowLen" ); 
	col   = (int*)    allocateMemory( size_col,"elr->col" ); 
	val   = (mat_data_t*) allocateMemory( size_val,"elr->val" ); 

	/* initialize values ########################################### */
	elr->rowLen = rowLen;
	elr->col = col;
	elr->val = val;
	elr->nrows 	= nrows;
	elr->nMaxRow = rowMaxEnt;
	elr->padding = padRows;

	/* fill with zeros ############################################ */
	for( j=0; j < elr->nMaxRow; ++j) {
		for( i=0; i < padRows; ++i) {
			elr->col[i+j*padRows] = 0;
			elr->val[i+j*padRows] = 0.0;
		}
	}

	for( i=0; i < elr->nrows; ++i) {
		elr->rowLen[i] = crs_row_ptr[i+1] - crs_row_ptr[i];
	}

	/* copy values ################################################ */
	for( i = 0; i < nrows; ++i) {

		for( j = 0; j < elr->rowLen[i]; ++j) {
			elr->col[ j*padRows+i ]   = crs_col[ crs_row_ptr[i]+j ];
			elr->val[ j*padRows+i ]   = crs_val[ crs_row_ptr[i]+j ];
		}
	}


	return elr;
}


void checkCRSToELR(	const mat_data_t* crs_val, const int* crs_col, 
		const int* crs_row_ptr, const int nrows,
		const ELR_TYPE* elr) {
	/* check if matrix in elr is consistent with CRS;
	 * assume FORTRAN numbering in crs, C numbering in ELR */

	int i,j, hlpi;
	int me;

	MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD, &me));

	printf("PE%i: -- ELRcopy sanity check:\n", me);
	for (i=0; i<nrows; i++){
		if( (crs_row_ptr[i+1] - crs_row_ptr[i]) != elr->rowLen[i]) 
			printf("PE%i: wrong number of entries in row %i:\t %i | %i\n", me, i, 
					crs_row_ptr[i+1] - crs_row_ptr[i], elr->rowLen[i]);

		hlpi = 0;
		for (j=crs_row_ptr[i]; j<crs_row_ptr[i+1]; j++){
			if( ABS(crs_val[j]-elr->val[i+hlpi*elr->padding])>EPSILON) 
				printf("PE%i: value mismatch [%i,%i]:\t%e+%ei | %e+%ei\n",
						me, i,hlpi, REAL(crs_val[j]), IMAG(crs_val[j]), REAL(elr->val[i+hlpi*elr->padding]),IMAG(elr->val[i+hlpi*elr->padding]));
			if( crs_col[j] != elr->col[i+hlpi*elr->padding]) 
				printf("PE%i: index mismatch [%i,%i]:\t%i | %i\n",
						me, i,hlpi, crs_col[j], elr->col[i+hlpi*elr->padding]);
			hlpi++;
		}
	}

	printf("PE%i: -- finished sanity check.\n", me);
}


/* ########################################################################## */

void resetPJDS( PJDS_TYPE* pjds ) {

	/* set col, val and rowLen in pjds to 0 */

	int i;

	for(i = 0; i < pjds->nEnts; ++i) {
		pjds->col[i] = 0;
		pjds->val[i] = 0.0;
	}
	for(i = 0; i < pjds->nrows; ++i) pjds->rowLen[i] = 0;
	for(i = 0; i < pjds->nMaxRow+1; ++i) pjds->colStart[i] = 0;
}

void resetELR( ELR_TYPE* elr ) {

	/* set col, val and rowLen in elr to 0 */

	int i,j;

	for(i = 0; i < elr->nrows; ++i) {
		for(j = 0; j < elr->nMaxRow; ++j) {
			elr->col[i+j*elr->padding] = 0;
			elr->val[i+j*elr->padding] = 0.0;
		}
	}
	for(i = 0; i < elr->nrows; ++i) elr->rowLen[i] = 0;
}


/* ########################################################################## */


void elrColIdToFortran( ELR_TYPE* elr ) {
	int i,j;

	for(i = 0; i < elr->nrows; ++i) {
		for(j = 0; j < elr->nMaxRow; ++j) {
			elr->col[i+j*elr->padding] += 1;
			if( elr->col[i+j*elr->padding] < 1 || elr->col[i+j*elr->padding] > elr->nrows ) {
				fprintf(stderr, "error in elrColIdToFortran: index out of bounds\n");
				exit(1);
			}
		}
	}
}

/* ########################################################################## */


void elrColIdToC( ELR_TYPE* elr ) {
	int i,j;

	for(i = 0; i < elr->nrows; ++i) {
		for(j = 0; j < elr->nMaxRow; ++j) {
			elr->col[i+j*elr->padding] -= 1;
			if( elr->col[i+j*elr->padding] < 0 || elr->col[i+j*elr->padding] > elr->nrows-1 ) {
				fprintf(stderr, "error in elrColIdToC: index out of bounds: elr->col[%i][%i]=%i\n",
						i,j,elr->col[i+j*elr->padding]);fflush(stderr);
				exit(1);
			}
		}
	}
}

CL_PJDS_TYPE* CL_initPJDS( const PJDS_TYPE* pjds) {

	/* allocate (but do not fill) memory for elr matrix on device */

	cl_mem col, rowLen, colStart, val;

	int me;

	size_t colMemSize = (size_t) pjds->nEnts * sizeof( int );
	size_t colStartMemSize = (size_t) (pjds->nMaxRow+1) * sizeof( int );
	size_t valMemSize = (size_t) pjds->nEnts * sizeof( mat_data_t );
	size_t rowMemSize = (size_t) pjds->nrows * sizeof( int );

	/* allocate */

	IF_DEBUG(1) { 
		MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD, &me));
		printf("PE%i: CPJDSinitAlloc: in columns\t %lu MB\n", me, colMemSize/(1024*1024));	
	}

	col = CL_allocDeviceMemory(colMemSize);

	IF_DEBUG(1) { 
		MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD, &me));
		printf("PE%i: CPJDSinitAlloc: in columns\t %lu MB\n", me, colStartMemSize/(1024*1024));	
	}
	colStart = CL_allocDeviceMemory(colStartMemSize);

	IF_DEBUG(1) printf("PE%i: CPJDSinitAlloc: in rows\t %lu MB\n",
			me, valMemSize/(1024*1024));
	val = CL_allocDeviceMemory(valMemSize);

	IF_DEBUG(1) printf("PE%i: CPJDSinitAlloc: in rLeng\t %lu MB\n",
			me, rowMemSize/(1024*1024));
	rowLen = CL_allocDeviceMemory(rowMemSize);

	/* create host handle */
	CL_PJDS_TYPE* cupjds = (CL_PJDS_TYPE*) allocateMemory( sizeof( CL_PJDS_TYPE ), "cuda_pjds");
	cupjds->nEnts   	= pjds->nEnts;
	cupjds->nrows   	= pjds->nrows;
	cupjds->padding  = pjds->padding;
	cupjds->nMaxRow 	= pjds->nMaxRow;
	cupjds->rowLen 	= rowLen;
	cupjds->col		= col;
	cupjds->colStart= colStart;
	cupjds->val		= val;
	IF_DEBUG(1) {
		printf("PE%i: created CL_PJDS type from PJDS with:\n nrows:\tcpjds=%i\t(pjds=%i)\n padding:\tcpjds=%i\t(pjds=%i)\nnMaxRow:\tcpjds=%i\t(pjds=%i)\n",
				me, cupjds->nrows, pjds->nrows, cupjds->padding, pjds->padding,
				cupjds->nMaxRow, pjds->nMaxRow);
	}
	return cupjds;
}


CL_ELR_TYPE* CL_initELR( const ELR_TYPE* elr) {

	/* allocate (but do not fill) memory for elr matrix on device */

	cl_mem col, rowLen, val;

	int me;

	size_t colMemSize = (size_t) elr->padding * elr->nMaxRow * sizeof( int );
	size_t valMemSize = (size_t) elr->padding * elr->nMaxRow * sizeof( mat_data_t );
	size_t rowMemSize = (size_t) elr->nrows * sizeof( int );


	/* allocate */

	IF_DEBUG(1) { 
		MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD, &me));
		printf("PE%i: CELRinitAlloc: in columns\t %lu MB\n", me, colMemSize/(1024*1024));	
	}
	col = CL_allocDeviceMemory(colMemSize);


	IF_DEBUG(1) printf("PE%i: CELRinitAlloc: in rows\t %lu MB\n",
			me, valMemSize/(1024*1024));
	val = CL_allocDeviceMemory(valMemSize);

	IF_DEBUG(1) printf("PE%i: CELRinitAlloc: in rLeng\t %lu MB\n",
			me, rowMemSize/(1024*1024));
	rowLen = CL_allocDeviceMemory(rowMemSize);

	/* create host handle */
	CL_ELR_TYPE* cuelr = (CL_ELR_TYPE*) allocateMemory( sizeof( CL_ELR_TYPE ), "cuda_elr");
	cuelr->nrows   	= elr->nrows;
	cuelr->padding  = elr->padding;
	cuelr->nMaxRow 	= elr->nMaxRow;
	cuelr->rowLen 	= rowLen;
	cuelr->col		= col;
	cuelr->val		= val;
	IF_DEBUG(1) {
		printf("PE%i: created CL_ELR type from ELR with:\n nrows:\tcelr=%i\t(elr=%i)\n padding:\tcelr=%i\t(elr=%i)\nnMaxRow:\tcelr=%i\t(elr=%i)\n",
				me, cuelr->nrows, elr->nrows, cuelr->padding, elr->padding,
				cuelr->nMaxRow, elr->nMaxRow);
	}
	return cuelr;
}

/* ########################################################################## */

void CL_uploadPJDS( CL_PJDS_TYPE* cpjds,  const PJDS_TYPE* pjds ) {

	/* copy col, val and rowLen from CPU elr format to device;
	 * celr must be allocated in advance (using cudaELRInit) */

	assert( cpjds->nEnts == pjds->nEnts );
	assert( cpjds->nrows == pjds->nrows );
	assert( cpjds->padding == pjds->padding );
	assert( cpjds->nMaxRow == pjds->nMaxRow );

	int me;
	MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD, &me));

	size_t colMemSize = (size_t) pjds->nEnts * sizeof( int );
	size_t colStartMemSize = (size_t) (pjds->nMaxRow+1) * sizeof( int );
	size_t valMemSize = (size_t) pjds->nEnts * sizeof( mat_data_t );
	size_t rowMemSize = (size_t) pjds->nrows * sizeof( int );

	IF_DEBUG(1) printf("PE%i: PJDStoDevice: col %lu \t(%lu MB)\n", me,
			colMemSize, colMemSize/(1024*1024));
	CL_copyHostToDevice( cpjds->col, pjds->col, colMemSize );

	IF_DEBUG(1) printf("PE%i: PJDStoDevice: colStart %lu \t(%lu MB)\n", me,
			colStartMemSize, colStartMemSize/(1024*1024));
	CL_copyHostToDevice( cpjds->colStart, pjds->colStart, colStartMemSize );

	IF_DEBUG(1) printf("PE%i: PJDStoDevice: val %lu \t(%lu MB)\n", me,
			valMemSize, valMemSize/(1024*1024));
	CL_copyHostToDevice( cpjds->val, pjds->val, valMemSize );

	IF_DEBUG(1) printf("PE%i: PJDStoDevice: row %lu \t(%lu MB)\n", me,
			rowMemSize, rowMemSize/(1024*1024));
	CL_copyHostToDevice( cpjds->rowLen, pjds->rowLen, rowMemSize );
}

void CL_uploadELR( CL_ELR_TYPE* celr,  const ELR_TYPE* elr ) {

	/* copy col, val and rowLen from CPU elr format to device;
	 * celr must be allocated in advance (using cudaELRInit) */

	assert( celr->nrows == elr->nrows );
	assert( celr->padding == elr->padding );
	assert( celr->nMaxRow == elr->nMaxRow );


	int me;
	MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD, &me));

	size_t colMemSize = (size_t) elr->padding * elr->nMaxRow * sizeof( int );
	size_t valMemSize = (size_t) elr->padding * elr->nMaxRow * sizeof( mat_data_t );
	size_t rowMemSize = (size_t) elr->nrows * sizeof( int );

	IF_DEBUG(1) printf("PE%i: ELRtoDevice: col %lu \t(%lu MB)\n", me,
			colMemSize, colMemSize/(1024*1024));
	CL_copyHostToDevice( celr->col, elr->col, colMemSize );

	IF_DEBUG(1) printf("PE%i: ELRtoDevice: val %lu \t(%lu MB)\n", me,
			valMemSize, valMemSize/(1024*1024));
	CL_copyHostToDevice( celr->val, elr->val, valMemSize );

	IF_DEBUG(1) printf("PE%i: ELRtoDevice: row %lu \t(%lu MB)\n", me,
			rowMemSize, rowMemSize/(1024*1024));
	CL_copyHostToDevice( celr->rowLen, elr->rowLen, rowMemSize );

}


void CL_downloadPJDS( PJDS_TYPE* pjds, const CL_PJDS_TYPE* cpjds ) {

	/* copy col, val and rowLen from CPU elr format to device;
	 * celr must be allocated in advance (using cudaELRInit) */

	assert( cpjds->nEnts == pjds->nEnts );
	assert( cpjds->nrows == pjds->nrows );
	assert( cpjds->padding == pjds->padding );
	assert( cpjds->nMaxRow == pjds->nMaxRow );


	int me;
	MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD, &me));

	size_t colMemSize = (size_t) pjds->nEnts * sizeof( int );
	size_t colStartMemSize = (size_t) pjds->nMaxRow * sizeof( int );
	size_t valMemSize = (size_t) pjds->nEnts * sizeof( mat_data_t );
	size_t rowMemSize = (size_t) pjds->nrows * sizeof( int );

	IF_DEBUG(1) printf("PE%i: PJDStoDevice: col %lu \t(%lu MB)\n", me,
			colMemSize, colMemSize/(1024*1024));
	CL_copyHostToDevice( cpjds->col, pjds->col, colMemSize );

	IF_DEBUG(1) printf("PE%i: PJDStoDevice: col %lu \t(%lu MB)\n", me,
			colStartMemSize, colStartMemSize/(1024*1024));
	CL_copyHostToDevice( cpjds->colStart, pjds->colStart, colStartMemSize );

	IF_DEBUG(1) printf("PE%i: PJDStoDevice: val %lu \t(%lu MB)\n", me,
			valMemSize, valMemSize/(1024*1024));
	CL_copyHostToDevice( cpjds->val, pjds->val, valMemSize );

	IF_DEBUG(1) printf("PE%i: PJDStoDevice: row %lu \t(%lu MB)\n", me,
			rowMemSize, rowMemSize/(1024*1024));
	CL_copyHostToDevice( cpjds->rowLen, pjds->rowLen, rowMemSize );
}

void CL_downloadELR( ELR_TYPE* elr, const CL_ELR_TYPE* celr ) {

	/* copy col, val and rowLen from device celr to CPU;
	 * elr must be allocated in advance */

	assert( celr->nrows == elr->nrows );
	assert( celr->padding == elr->padding );
	assert( celr->nMaxRow == elr->nMaxRow );

	int me;
	MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD, &me));

	size_t colMemSize = (size_t) elr->padding * elr->nMaxRow * sizeof( int );
	size_t valMemSize = (size_t) elr->padding * elr->nMaxRow * sizeof( mat_data_t );
	size_t rowMemSize = (size_t) elr->nrows * sizeof( int );

	IF_DEBUG(1) printf("PE%i: ELRtoHost: col %lu \t(%lu MB)\n", me,
			colMemSize, colMemSize/(1024*1024));
	CL_copyDeviceToHost( elr->col, celr->col, colMemSize );

	IF_DEBUG(1) printf("PE%i: ELRtoHost: val %lu \t(%lu MB)\n", me,
			valMemSize, valMemSize/(1024*1024));
	CL_copyDeviceToHost( elr->val, celr->val, valMemSize );

	IF_DEBUG(1) printf("PE%i: ELRtoHost: row %lu \t(%lu MB)\n", me,
			rowMemSize, rowMemSize/(1024*1024));
	CL_copyDeviceToHost( elr->rowLen, celr->rowLen, rowMemSize );

}

/* ########################################################################## */

void freePJDS( PJDS_TYPE* const pjds ) {
	if( pjds ) {
		free( pjds->rowLen );
		free( pjds->col );
		free( pjds->colStart );
		free( pjds->val );
		free( pjds );
	}
}

void freeELR( ELR_TYPE* const elr ) {
	if( elr ) {
		free( elr->rowLen );
		free( elr->col );
		free( elr->val );
		free( elr );
	}
}


void CL_freePJDS( CL_PJDS_TYPE* const cpjds ) {
	if( cpjds ) {
		CL_freeDeviceMemory( cpjds->rowLen );
		CL_freeDeviceMemory( cpjds->col );
		CL_freeDeviceMemory( cpjds->colStart );
		CL_freeDeviceMemory( cpjds->val );
		free( cpjds );
	}
}

void CL_freeELR( CL_ELR_TYPE* const celr ) {
	if( celr ) {
		CL_freeDeviceMemory( celr->rowLen );
		CL_freeDeviceMemory( celr->col );
		CL_freeDeviceMemory( celr->val );
		free( celr );
	}
}

void CL_freeMatrix(void *matrix, int format) {
	if (matrix) {
		if (format == SPM_GPUFORMAT_ELR) {
			CL_freeELR((CL_ELR_TYPE *)matrix);
		} else if (format == SPM_GPUFORMAT_PJDS) {
			CL_freePJDS((CL_PJDS_TYPE *)matrix);
		}
	}
}

