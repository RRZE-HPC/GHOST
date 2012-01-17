#include <mpi.h>
#include "my_ellpack.h"
#include "matricks.h"
#include "mymacros.h"
#include "cudafun.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

void getPadding(int nRows, int* paddedRows) {

	/* determine padding of rowlength in ELR format to achieve half-warp alignment */

	int padBlock = 16*sizeof(double);

	if(  nRows % padBlock != 0) {
		*paddedRows = nRows + padBlock - nRows % padBlock;
	} else {
		*paddedRows = nRows;
	}
}

/**********************  pJDS MATRIX TYPE *********************************/

PJDS_TYPE* convertCRSToPJDSMatrix(  const double* crs_val, const int* crs_col, 
		const int* crs_row_ptr, const int nRows, int* rowPerm, int* invRowPerm) 
{
	ELR_TYPE * elrs;
	elrs = convertCRSToELRSortedMatrix(crs_val, crs_col, crs_row_ptr, nRows, rowPerm, invRowPerm);
	PJDS_TYPE * pjds;
	pjds = convertELRSortedToPJDSMatrix(elrs);
	return pjds;
}



PJDS_TYPE* convertELRSortedToPJDSMatrix( const ELR_TYPE* elr ) {
	PJDS_TYPE *pjds = NULL;
	int *rowLen;
	int  i;
	size_t size_val, size_col, size_rowlen, size_colStart;

	if (elr->padding % PJDS_CHUNK_HEIGHT != 0) {
		printf("ELR matrix cannot be divided into chunks.\n");
		exit(EXIT_FAILURE);
	}

	size_rowlen   = (int)sizeof(int)*elr->padding;


	rowLen = (int*) allocHostMemory( size_rowlen ); 

	pjds = (PJDS_TYPE*) allocateMemory( sizeof( PJDS_TYPE ), "pjds");

	// initialize pjds entries	
	pjds->padding = elr->padding;
	pjds->nRows = elr->nRows;
	pjds->nMaxRow = elr->nMaxRow;
	pjds->rowLen = rowLen;
	pjds->nEnts = 0;
	memcpy(pjds->rowLen,elr->rowLen,size_rowlen);


	size_t *colHeight = (size_t *)malloc(sizeof(size_t)*elr->rowLen[0]);
	int curCol = elr->rowLen[0]-1;
	int curChunk = 0, st=0;

	for (i=0; i<elr->padding; i+=PJDS_CHUNK_HEIGHT) 
	{
		if (i>=elr->nRows)
			pjds->rowLen[i] = pjds->rowLen[pjds->nRows-1];

		pjds->nEnts += PJDS_CHUNK_HEIGHT*pjds->rowLen[i];

		// if a step occurs save the column heights
		if (curChunk != 0 && pjds->rowLen[i] != pjds->rowLen[i-PJDS_CHUNK_HEIGHT]) 
		{
			IF_DEBUG(1) printf("step at chunk %d from %d to %d\n",curChunk,pjds->rowLen[i-PJDS_CHUNK_HEIGHT],pjds->rowLen[i]);
			for (st=0; st<pjds->rowLen[i-PJDS_CHUNK_HEIGHT]-pjds->rowLen[i]; st++) // count all cols
			{ 
				colHeight[curCol] = i;
				IF_DEBUG(2)	printf("col: %d | height: %ld\n",curCol,colHeight[curCol]);
				curCol--;
			}
		}
		curChunk++;

	}

	// collect all columns with maximal height
	while(curCol >= 0) {
		colHeight[curCol] = elr->padding;
		IF_DEBUG(2)	printf("col: %d | height: %ld\n",curCol,colHeight[curCol]);
		curCol--;
	}

	size_val = (size_t)sizeof(double)*pjds->nEnts;
	size_col = (size_t)sizeof(int)*pjds->nEnts;
	size_colStart = sizeof(int)*(pjds->rowLen[0]+1);


	pjds->val = (double*) allocHostMemory( size_val ); 
	pjds->colStart = (int*) allocHostMemory( size_colStart );
	pjds->col = (int*) allocHostMemory( size_col);

	int j;
	pjds->colStart[0] = 0; // initial colStart is zero
	for (j=1; j<pjds->rowLen[0]; j++) // save all other colStart
	{
		pjds->colStart[j] = pjds->colStart[j-1]+colHeight[j-1];
	}
	pjds->colStart[pjds->rowLen[0]]=pjds->nEnts;

	// check for sanity
	assert(pjds->colStart[pjds->rowLen[0]-1] + colHeight[(pjds->rowLen[0]-1)] == pjds->nEnts);

	// copy col and val from elr-s to pjds

	for (j=0; j<pjds->rowLen[0]; j++) 
	{
		for (i=0; i<pjds->colStart[j+1]-pjds->colStart[j]; i++) 
		{
			pjds->val[pjds->colStart[j]+i] = elr->val[j*elr->padding+i];
			pjds->col[pjds->colStart[j]+i] = elr->col[j*elr->padding+i];

		}	
	}
	

	return pjds;

}


/**********************  sorted ELR MATRIX TYPE *********************************/

ELR_TYPE* convertCRSToELRSortedMatrix(  const double* crs_val, const int* crs_col, 
		const int* crs_row_ptr, const int nRows,
		int* rowPerm, int* invRowPerm) {

	JD_SORT_TYPE* rowSort;
	int i, j, rowMaxEnt, padRows;
	size_t size_val, size_col, size_rowlen;
	int *rowLen, *col;
	double* val;
	ELR_TYPE* elr = NULL;

	/* get max number of entries in one row ###########################*/
	rowSort = (JD_SORT_TYPE*) allocateMemory( nRows * sizeof( JD_SORT_TYPE ),
			"rowSort" );

	for( i = 0; i < nRows; i++ ) {
		rowSort[i].row = i;
		rowSort[i].nEntsInRow = 0;
	} 

	/* count entries per row ################################################## */
	for( i = 0; i < nRows; i++) {
		rowSort[i].nEntsInRow = crs_row_ptr[i+1] - crs_row_ptr[i];
		//IF_DEBUG(1) printf("row: %d, nEnts: %d\n",i,rowSort[i].nEntsInRow);
	}

	IF_DEBUG(2) {
		i=0;
		while(i < nRows) {
			int start = i;

			j = rowSort[start].nEntsInRow;
			while( i<nRows && rowSort[i].nEntsInRow == j ) ++i;

			if( (i-start)%5 != 0 || j%5 != 0 )
				printf("%i rows (%i): %i - %i\n",i-start,j, start, i-1);

		}
	}
	/* sort rows with desceding number of NZEs ################################ */
	qsort( rowSort, nRows, sizeof( JD_SORT_TYPE  ), compareNZEPerRow );
	rowMaxEnt = rowSort[0].nEntsInRow;

	/* sort within same rowlength with asceding row number #################### */
	i=0;
	while(i < nRows) {
		int start = i;

		j = rowSort[start].nEntsInRow;
		while( i<nRows && rowSort[i].nEntsInRow >= j ) 
			++i;

		printf("sorting over %i rows (%i): %i - %i\n",i-start,j, start, i-1);
		qsort( &rowSort[start], i-start, sizeof(JD_SORT_TYPE), compareNZEOrgPos );
	}

	for(i=1; i < nRows; ++i) {
		if( rowSort[i].nEntsInRow == rowSort[i-1].nEntsInRow && rowSort[i].row < rowSort[i-1].row)
			printf("Error in row %i: descending row number\n",i);
	}


	/* allocate memory ################################################*/
	elr = (ELR_TYPE*) allocateMemory( sizeof( ELR_TYPE ), "elr_sorted");
	padRows = nRows;
	//#ifdef PADDING
	getPadding(nRows, &padRows);
	IF_DEBUG(1)  printf("convertCRS to ELR: padding: \t nRows=%i to %i\n", nRows, padRows);
	//#endif

	size_val    = (size_t) sizeof(double) * padRows * rowMaxEnt;
	size_col    = (size_t) sizeof(int) * padRows * rowMaxEnt;
	size_rowlen = (size_t) sizeof(int) * nRows;

	/* get the permutation indices ############################################ */
	printf("permuted ELR format\n");
	for(i=0; i < nRows; ++i) {
		/* invRowPerm maps an index in the permuted system to the original index,
		 * rowPerm gets the original index and returns the corresponding permuted position.
		 */
		if( rowSort[i].row >= nRows ) printf("error: invalid row number %i in %i\n",rowSort[i].row, i); fflush(stdout);

		invRowPerm[i] = rowSort[i].row;
		rowPerm[rowSort[i].row] = i;
	}

	rowLen = (int*)   allocHostMemory( size_rowlen ); 
	col   = (int*)    allocHostMemory( size_col ); 
	val   = (double*)   allocHostMemory( size_val ); 

	/* initialize values ########################################### */
	elr->rowLen = rowLen;
	elr->col = col;
	elr->val = val;
	elr->nRows  = nRows;
	elr->nMaxRow = rowMaxEnt;
	elr->padding = padRows;

	/* fill with zeros ############################################ */
	for( j=0; j < elr->nMaxRow; ++j) {
		for( i=0; i < padRows; ++i) {
			elr->col[i+j*padRows] = 0;
			elr->val[i+j*padRows] = 0.0;
		}
	}

	for( i=0; i < elr->nRows; ++i) {
		elr->rowLen[i] = rowSort[i].nEntsInRow;
		/* should be equivalent to: */
		//elr->rowLen[rowPerm[i]] = crs_row_ptr[i+1] - crs_row_ptr[i];
	}

	/* copy values ################################################ */
	for( i = 0; i < nRows; ++i) {
		/* i runs in the permuted index, access to crs needs to be original index */
		/* RHS is also permuted to sorted system */
		for( j = 0; j < elr->rowLen[i]; ++j) {

			if( j*padRows+i >= elr->nMaxRow*padRows ) 
				printf("error: in i=%i, j=%i\n",i,j);

			//elr->col[ j*padRows+i ]   = rowPerm[crs_col[ crs_row_ptr[invRowPerm[i]]+j ]];
			// XXX: columns are NOT being permuted!
			elr->col[ j*padRows+i ]   = crs_col[ crs_row_ptr[invRowPerm[i]]+j ];
			elr->val[ j*padRows+i ]   = crs_val[ crs_row_ptr[invRowPerm[i]]+j ];
		}
	}

	/* access to both RHS and Res need to be changed, but Result also needs
	 * to be reverted back after MVM.
	 */
	j=0;

	for(i=1; i<nRows; ++i)
		if(rowSort[i].row == rowSort[i-1].row+1) 
			j++;

	printf("coherency: %2f\n", 100.0*j/nRows);

	for(i=0; i <10; ++i) {
		printf("row %i (len: %d): ",i,rowLen[i]);
		for(j=0; j<10 && j< elr->rowLen[i]; ++j) {
			printf("%d: %f ",elr->col[j*padRows+i],elr->val[j*padRows+i]);
		}
		printf("\n");
	}

	free(rowSort);

	return elr;
}


void checkCRSToPJDSsanity(const double* crs_val, const int* crs_col, 
		const int* crs_row_ptr, const int nRows,
		const PJDS_TYPE* pjds, const int* invRowPerm) {

	
}

/**********************  ELR MATRIX TYPE *********************************/

ELR_TYPE* convertCRSToELRMatrix(  const double* crs_val, const int* crs_col, 
		const int* crs_row_ptr, const int nRows) {

	/* allocate and fill ELR-format matrix from CRS format data;
	 * elements in row retain CRS order (usually sorted by column);*/

	JD_SORT_TYPE* rowSort;
	int i, j, e, pos, rowMaxEnt, padRows;
	size_t size_val, size_col, size_rowlen;
	int *rowLen, *col;
	double* val;
	ELR_TYPE* elr = NULL;

	/* get max number of entries in one row ###########################*/
	rowSort = (JD_SORT_TYPE*) allocateMemory( nRows * sizeof( JD_SORT_TYPE ),
			"rowSort" );

	for( i = 0; i < nRows; i++ ) {
		rowSort[i].row = i;
		rowSort[i].nEntsInRow = 0;
	} 

	/* count entries per row ################################################## */
	for( i = 0; i < nRows; i++) 
		rowSort[i].nEntsInRow = crs_row_ptr[i+1] - crs_row_ptr[i];

	/* sort rows with desceding number of NZEs ################################ */
	qsort( rowSort, nRows, sizeof( JD_SORT_TYPE  ), compareNZEPerRow );

	rowMaxEnt = rowSort[0].nEntsInRow;
	free( rowSort );

	/* allocate memory ################################################*/
	elr = (ELR_TYPE*) allocateMemory( sizeof( ELR_TYPE ), "elr");
	padRows = nRows;
	//#ifdef PADDING
	getPadding(nRows, &padRows);
	IF_DEBUG(1)  printf("convertCRS to ELR: padding: \t nRows=%i to %i\n", nRows, padRows);
	//#endif

	size_val    = (size_t) sizeof(double) * padRows * rowMaxEnt;
	size_col    = (size_t) sizeof(int) * padRows * rowMaxEnt;
	size_rowlen = (size_t) sizeof(int) * nRows;

	rowLen = (int*)   allocHostMemory( size_rowlen ); 
	col   = (int*)    allocHostMemory( size_col ); 
	val   = (double*) allocHostMemory( size_val ); 

	/* initialize values ########################################### */
	elr->rowLen = rowLen;
	elr->col = col;
	elr->val = val;
	elr->nRows 	= nRows;
	elr->nMaxRow = rowMaxEnt;
	elr->padding = padRows;

	/* fill with zeros ############################################ */
	for( j=0; j < elr->nMaxRow; ++j) {
		for( i=0; i < padRows; ++i) {
			elr->col[i+j*padRows] = 0;
			elr->val[i+j*padRows] = 0.0;
		}
	}

	for( i=0; i < elr->nRows; ++i) {
		elr->rowLen[i] = crs_row_ptr[i+1] - crs_row_ptr[i];
	}

	/* copy values ################################################ */
	for( i = 0; i < nRows; ++i) {

		for( j = 0; j < elr->rowLen[i]; ++j) {
			elr->col[ j*padRows+i ]   = crs_col[ crs_row_ptr[i]+j ];
			elr->val[ j*padRows+i ]   = crs_val[ crs_row_ptr[i]+j ];
		}
	}


	return elr;
}


/* ########################################################################## */


void checkCRSToELRsanity(	const double* crs_val, const int* crs_col, 
		const int* crs_row_ptr, const int nRows,
		const ELR_TYPE* elr) {
	/* check if matrix in elr is consistent with CRS;
	 * assume FORTRAN numbering in crs, C numbering in ELR */

	int i,j, hlpi;
	int me, ierr;

	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &me);

	printf("PE%i: -- ELRcopy sanity check:\n", me);
	for (i=0; i<nRows; i++){
		if( (crs_row_ptr[i+1] - crs_row_ptr[i]) != elr->rowLen[i]) 
			printf("PE%i: wrong number of entries in row %i:\t %i | %i\n", me, i, 
					crs_row_ptr[i+1] - crs_row_ptr[i], elr->rowLen[i]);

		hlpi = 0;
		for (j=crs_row_ptr[i]; j<crs_row_ptr[i+1]; j++){
			if( crs_val[j] != elr->val[i+hlpi*elr->padding]) 
				printf("PE%i: value mismatch [%i,%i]:\t%e | %e\n",
						me, i,hlpi, crs_val[j], elr->val[i+hlpi*elr->padding]);
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
	for(i = 0; i < pjds->nRows; ++i) pjds->rowLen[i] = 0;
	for(i = 0; i < pjds->nMaxRow+1; ++i) pjds->colStart[i] = 0;
}

void resetELR( ELR_TYPE* elr ) {

	/* set col, val and rowLen in elr to 0 */

	int i,j;

	for(i = 0; i < elr->nRows; ++i) {
		for(j = 0; j < elr->nMaxRow; ++j) {
			elr->col[i+j*elr->padding] = 0;
			elr->val[i+j*elr->padding] = 0.0;
		}
	}
	for(i = 0; i < elr->nRows; ++i) elr->rowLen[i] = 0;
}


/* ########################################################################## */


void elrColIdToFortran( ELR_TYPE* elr ) {
	int i,j;

	for(i = 0; i < elr->nRows; ++i) {
		for(j = 0; j < elr->nMaxRow; ++j) {
			elr->col[i+j*elr->padding] += 1;
			if( elr->col[i+j*elr->padding] < 1 || elr->col[i+j*elr->padding] > elr->nRows ) {
				fprintf(stderr, "error in elrColIdToFortran: index out of bounds\n");
				exit(1);
			}
		}
	}
}

/* ########################################################################## */


void elrColIdToC( ELR_TYPE* elr ) {
	int i,j;

	for(i = 0; i < elr->nRows; ++i) {
		for(j = 0; j < elr->nMaxRow; ++j) {
			elr->col[i+j*elr->padding] -= 1;
			if( elr->col[i+j*elr->padding] < 0 || elr->col[i+j*elr->padding] > elr->nRows-1 ) {
				fprintf(stderr, "error in elrColIdToC: index out of bounds: elr->col[%i][%i]=%i\n",
						i,j,elr->col[i+j*elr->padding]);fflush(stderr);
				exit(1);
			}
		}
	}
}

/* ########################################################################## */

CUDA_PJDS_TYPE* cudaPJDSInit( const PJDS_TYPE* pjds) {

	/* allocate (but do not fill) memory for elr matrix on device */

	int *col, *rowLen, *colStart;
	double *val;

	int me, ierr;

	size_t colMemSize = (size_t) pjds->nEnts * sizeof( int );
	size_t colStartMemSize = (size_t) (pjds->nMaxRow+1) * sizeof( int );
	size_t valMemSize = (size_t) pjds->nEnts * sizeof( double );
	size_t rowMemSize = (size_t) pjds->padding * sizeof( int );

	/* allocate */

	IF_DEBUG(1) { 
		ierr = MPI_Comm_rank(MPI_COMM_WORLD, &me);
		printf("PE%i: CPJDSinitAlloc: in columns\t %lu MB\n", me, colMemSize/(1024*1024));	
	}
	col     = allocDeviceMemory( colMemSize );

	IF_DEBUG(1) { 
		ierr = MPI_Comm_rank(MPI_COMM_WORLD, &me);
		printf("PE%i: CPJDSinitAlloc: in columns\t %lu MB\n", me, colStartMemSize/(1024*1024));	
	}
	colStart     = allocDeviceMemory( colStartMemSize );

	IF_DEBUG(1) printf("PE%i: CPJDSinitAlloc: in rows\t %lu MB\n",
			me, valMemSize/(1024*1024));
	val     = allocDeviceMemory( valMemSize );

	IF_DEBUG(1) printf("PE%i: CPJDSinitAlloc: in rLeng\t %lu MB\n",
			me, rowMemSize/(1024*1024));
	rowLen  = allocDeviceMemory( rowMemSize );

	/* create host handle */
	CUDA_PJDS_TYPE* cupjds = (CUDA_PJDS_TYPE*) allocateMemory( sizeof( CUDA_PJDS_TYPE ), "cuda_pjds");
	cupjds->nEnts   	= pjds->nEnts;
	cupjds->nRows   	= pjds->nRows;
	cupjds->padding  = pjds->padding;
	cupjds->nMaxRow 	= pjds->nMaxRow;
	cupjds->rowLen 	= rowLen;
	cupjds->col		= col;
	cupjds->colStart= colStart;
	cupjds->val		= val;
	IF_DEBUG(1) {
		printf("PE%i: created cudaPJDS type from PJDS with:\n nRows:\tcpjds=%i\t(pjds=%i)\n padding:\tcpjds=%i\t(pjds=%i)\nnMaxRow:\tcpjds=%i\t(pjds=%i)\n",
				me, cupjds->nRows, pjds->nRows, cupjds->padding, pjds->padding,
				cupjds->nMaxRow, pjds->nMaxRow);
	}
	return cupjds;
}

CUDA_ELR_TYPE* cudaELRInit( const ELR_TYPE* elr) {

	/* allocate (but do not fill) memory for elr matrix on device */

	int *col, *rowLen;
	double *val;

	int me, ierr;

	size_t colMemSize = (size_t) elr->padding * elr->nMaxRow * sizeof( int );
	size_t valMemSize = (size_t) elr->padding * elr->nMaxRow * sizeof( double );
	size_t rowMemSize = (size_t) elr->nRows * sizeof( int );

	/* allocate */

	IF_DEBUG(1) { 
		ierr = MPI_Comm_rank(MPI_COMM_WORLD, &me);
		printf("PE%i: CELRinitAlloc: in columns\t %lu MB\n", me, colMemSize/(1024*1024));	
	}
	col     = allocDeviceMemory( colMemSize );

	IF_DEBUG(1) printf("PE%i: CELRinitAlloc: in rows\t %lu MB\n",
			me, valMemSize/(1024*1024));
	val     = allocDeviceMemory( valMemSize );

	IF_DEBUG(1) printf("PE%i: CELRinitAlloc: in rLeng\t %lu MB\n",
			me, rowMemSize/(1024*1024));
	rowLen  = allocDeviceMemory( rowMemSize );

	/* create host handle */
	CUDA_ELR_TYPE* cuelr = (CUDA_ELR_TYPE*) allocateMemory( sizeof( CUDA_ELR_TYPE ), "cuda_elr");
	cuelr->nRows   	= elr->nRows;
	cuelr->padding  = elr->padding;
	cuelr->nMaxRow 	= elr->nMaxRow;
	cuelr->rowLen 	= rowLen;
	cuelr->col		= col;
	cuelr->val		= val;
	IF_DEBUG(1) {
		printf("PE%i: created cudaELR type from ELR with:\n nRows:\tcelr=%i\t(elr=%i)\n padding:\tcelr=%i\t(elr=%i)\nnMaxRow:\tcelr=%i\t(elr=%i)\n",
				me, cuelr->nRows, elr->nRows, cuelr->padding, elr->padding,
				cuelr->nMaxRow, elr->nMaxRow);
	}
	return cuelr;
}


/* ########################################################################## */

void cudaCopyPJDSToDevice( CUDA_PJDS_TYPE* cpjds,  const PJDS_TYPE* pjds ) {

	/* copy col, val and rowLen from CPU elr format to device;
	 * celr must be allocated in advance (using cudaELRInit) */

	assert( cpjds->nEnts == pjds->nEnts );
	assert( cpjds->nRows == pjds->nRows );
	assert( cpjds->padding == pjds->padding );
	assert( cpjds->nMaxRow == pjds->nMaxRow );

	int me, ierr;
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &me);

	size_t colMemSize = (size_t) pjds->nEnts * sizeof( int );
	size_t colStartMemSize = (size_t) (pjds->nMaxRow+1) * sizeof( int );
	size_t valMemSize = (size_t) pjds->nEnts * sizeof( double );
	size_t rowMemSize = (size_t) pjds->padding * sizeof( int );

	IF_DEBUG(1) printf("PE%i: PJDStoDevice: col %lu \t(%lu MB)\n", me,
			colMemSize, colMemSize/(1024*1024));
	copyHostToDevice( cpjds->col, pjds->col, colMemSize );

	IF_DEBUG(1) printf("PE%i: PJDStoDevice: col %lu \t(%lu MB)\n", me,
			colMemSize, colStartMemSize/(1024*1024));
	copyHostToDevice( cpjds->colStart, pjds->colStart, colStartMemSize );

	IF_DEBUG(1) printf("PE%i: PJDStoDevice: val %lu \t(%lu MB)\n", me,
			valMemSize, valMemSize/(1024*1024));
	copyHostToDevice( cpjds->val, pjds->val, valMemSize );

	IF_DEBUG(1) printf("PE%i: PJDStoDevice: row %lu \t(%lu MB)\n", me,
			rowMemSize, rowMemSize/(1024*1024));
	copyHostToDevice( cpjds->rowLen, pjds->rowLen, rowMemSize );
}

void cudaCopyELRToDevice( CUDA_ELR_TYPE* celr,  const ELR_TYPE* elr ) {

	/* copy col, val and rowLen from CPU elr format to device;
	 * celr must be allocated in advance (using cudaELRInit) */

	assert( celr->nRows == elr->nRows );
	assert( celr->padding == elr->padding );
	assert( celr->nMaxRow == elr->nMaxRow );


	int me, ierr;
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &me);

	size_t colMemSize = (size_t) elr->padding * elr->nMaxRow * sizeof( int );
	size_t valMemSize = (size_t) elr->padding * elr->nMaxRow * sizeof( double );
	size_t rowMemSize = (size_t) elr->nRows * sizeof( int );

	IF_DEBUG(1) printf("PE%i: ELRtoDevice: col %lu \t(%lu MB)\n", me,
			colMemSize, colMemSize/(1024*1024));
	copyHostToDevice( celr->col, elr->col, colMemSize );

	IF_DEBUG(1) printf("PE%i: ELRtoDevice: val %lu \t(%lu MB)\n", me,
			valMemSize, valMemSize/(1024*1024));
	copyHostToDevice( celr->val, elr->val, valMemSize );

	IF_DEBUG(1) printf("PE%i: ELRtoDevice: row %lu \t(%lu MB)\n", me,
			rowMemSize, rowMemSize/(1024*1024));
	copyHostToDevice( celr->rowLen, elr->rowLen, rowMemSize );

}


/* ########################################################################## */

void cudaCopyPJDSBackToHost( PJDS_TYPE* pjds, const CUDA_PJDS_TYPE* cpjds ) {

	/* copy col, val and rowLen from CPU elr format to device;
	 * celr must be allocated in advance (using cudaELRInit) */

	assert( cpjds->nEnts == pjds->nEnts );
	assert( cpjds->nRows == pjds->nRows );
	assert( cpjds->padding == pjds->padding );
	assert( cpjds->nMaxRow == pjds->nMaxRow );


	int me, ierr;
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &me);

	size_t colMemSize = (size_t) pjds->nEnts * sizeof( int );
	size_t colStartMemSize = (size_t) (pjds->nMaxRow+1) * sizeof( int );
	size_t valMemSize = (size_t) pjds->nEnts * sizeof( double );
	size_t rowMemSize = (size_t) pjds->padding * sizeof( int );

	IF_DEBUG(1) printf("PE%i: PJDStoDevice: col %lu \t(%lu MB)\n", me,
			colMemSize, colMemSize/(1024*1024));
	copyHostToDevice( cpjds->col, pjds->col, colMemSize );

	IF_DEBUG(1) printf("PE%i: PJDStoDevice: col %lu \t(%lu MB)\n", me,
			colStartMemSize, colStartMemSize/(1024*1024));
	copyHostToDevice( cpjds->colStart, pjds->colStart, colStartMemSize );

	IF_DEBUG(1) printf("PE%i: PJDStoDevice: val %lu \t(%lu MB)\n", me,
			valMemSize, valMemSize/(1024*1024));
	copyHostToDevice( cpjds->val, pjds->val, valMemSize );

	IF_DEBUG(1) printf("PE%i: PJDStoDevice: row %lu \t(%lu MB)\n", me,
			rowMemSize, rowMemSize/(1024*1024));
	copyHostToDevice( cpjds->rowLen, pjds->rowLen, rowMemSize );
}

void cudaCopyELRBackToHost( ELR_TYPE* elr, const CUDA_ELR_TYPE* celr ) {

	/* copy col, val and rowLen from device celr to CPU;
	 * elr must be allocated in advance */

	assert( celr->nRows == elr->nRows );
	assert( celr->padding == elr->padding );
	assert( celr->nMaxRow == elr->nMaxRow );

	int me, ierr;
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &me);

	size_t colMemSize = (size_t) elr->padding * elr->nMaxRow * sizeof( int );
	size_t valMemSize = (size_t) elr->padding * elr->nMaxRow * sizeof( double );
	size_t rowMemSize = (size_t) elr->nRows * sizeof( int );

	IF_DEBUG(1) printf("PE%i: ELRtoHost: col %lu \t(%lu MB)\n", me,
			colMemSize, colMemSize/(1024*1024));
	copyDeviceToHost( elr->col, celr->col, colMemSize );

	IF_DEBUG(1) printf("PE%i: ELRtoHost: val %lu \t(%lu MB)\n", me,
			valMemSize, valMemSize/(1024*1024));
	copyDeviceToHost( elr->val, celr->val, valMemSize );

	IF_DEBUG(1) printf("PE%i: ELRtoHost: row %lu \t(%lu MB)\n", me,
			rowMemSize, rowMemSize/(1024*1024));
	copyDeviceToHost( elr->rowLen, celr->rowLen, rowMemSize );

}


/* ########################################################################## */

void freePJDSMatrix( PJDS_TYPE* const pjds ) {
	if( pjds ) {
		printf("freeing host mem %p\n",pjds->rowLen);
		freeHostMemory( pjds->rowLen );
		freeHostMemory( pjds->col );
		freeHostMemory( pjds->colStart );
		freeHostMemory( pjds->val );
		free( pjds );
	}
}

void freeELRMatrix( ELR_TYPE* const elr ) {
	if( elr ) {
		freeHostMemory( elr->rowLen );
		freeHostMemory( elr->col );
		freeHostMemory( elr->val );
		free( elr );
	}
}


/* ########################################################################## */

void freeCUDAPJDSMatrix( CUDA_PJDS_TYPE* const cpjds ) {
	if( cpjds ) {
		freeDeviceMemory( cpjds->rowLen );
		freeDeviceMemory( cpjds->col );
		freeDeviceMemory( cpjds->colStart );
		freeDeviceMemory( cpjds->val );
		free( cpjds );
	}
}

void freeCUDAELRMatrix( CUDA_ELR_TYPE* const celr ) {
	if( celr ) {
		freeDeviceMemory( celr->rowLen );
		freeDeviceMemory( celr->col );
		freeDeviceMemory( celr->val );
		free( celr );
	}
}

