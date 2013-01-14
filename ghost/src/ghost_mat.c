#define _XOPEN_SOURCE 600
#include "ghost_mat.h"
#include "ghost.h"
#include "ghost_util.h"
#include "ghost_mmio.h"

#include <string.h>
#include <libgen.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>


int compareNZEPos( const void* a, const void* b ) 
{

	/* comparison function for sorting of matrix entries;
	 * sort lesser row id first, then lesser column id first;
	 * if MAIN_DIAGONAL_FIRST is defined sort diagonal 
	 * before lesser column id */

	int aRow = ((NZE_TYPE*)a)->row,
		bRow = ((NZE_TYPE*)b)->row,
		aCol = ((NZE_TYPE*)a)->col,
		bCol = ((NZE_TYPE*)b)->col;

	if( aRow == bRow ) {
#ifdef MAIN_DIAGONAL_FIRST
		if( aRow == aCol ) aCol = -1;
		if( bRow == bCol ) bCol = -1;
#endif /* MAIN_DIAGONAL_FIRST */
		return aCol - bCol;
	}
	else return aRow - bRow;
}

int compareNZEOrgPos( const void* a, const void* b ) 
{
	return  ((JD_SORT_TYPE*)a)->row - ((JD_SORT_TYPE*)b)->row;
}

int compareNZEPerRow( const void* a, const void* b ) 
{
	/* comparison function for JD_SORT_TYPE; 
	 * sorts rows with higher number of non-zero elements first */

	return  ((JD_SORT_TYPE*)b)->nEntsInRow - ((JD_SORT_TYPE*)a)->nEntsInRow;
}

int isMMfile(const char *filename) 
{

	FILE *file = fopen( filename, "r" );

	if( ! file ) {
		ABORT("Could not open file in isMMfile: %s",filename);
	}

	const char *keyword="%%MatrixMarket";
	char *readkw = (char *)allocateMemory((strlen(keyword)+1)*sizeof(char),"readkw");
	if (NULL == fgets(readkw,strlen(keyword)+1,file))
		return 0;

	int cmp = strcmp(readkw,keyword);

	free(readkw);
	return cmp==0?1:0;
}

int pad(int nrows, int padding) 
{
	int nrowsPadded;

	if(  nrows % padding != 0) {
		nrowsPadded = nrows + padding - nrows % padding;
	} else {
		nrowsPadded = nrows;
	}
	return nrowsPadded;
}
