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
    for( i = 0; i < nRows; i++) rowSort[i].nEntsInRow = crs_row_ptr[i+1] - crs_row_ptr[i];

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


void freeELRMatrix( ELR_TYPE* const elr ) {
	if( elr ) {
		freeHostMemory( elr->rowLen );
    freeHostMemory( elr->col );
    freeHostMemory( elr->val );
		free( elr );
	}
}


/* ########################################################################## */


void freeCUDAELRMatrix( CUDA_ELR_TYPE* const celr ) {
	if( celr ) {
		freeDeviceMemory( celr->rowLen );
		freeDeviceMemory( celr->col );
		freeDeviceMemory( celr->val );
		free( celr );
	}
}

