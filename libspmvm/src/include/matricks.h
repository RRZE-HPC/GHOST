#ifndef _MATRICKS_H_
#define _MATRICKS_H_

#include "spmvm.h"

#include <stdio.h>
#include <stdlib.h>


typedef struct 
{
	int row, col, nThEntryInRow;
	mat_data_t val;
} 
NZE_TYPE;

typedef struct 
{
	int nRows, nCols, nEnts;
	NZE_TYPE* nze;
} 
MM_TYPE;

typedef struct 
{
	int row, nEntsInRow;
} 
JD_SORT_TYPE;

typedef unsigned long long uint64;

int isMMfile(const char *filename);

void* allocateMemory( const size_t size, const char* desc );
void freeMemory(size_t, const char*, void*);


MM_TYPE* readMMFile( const char* filename );

CR_TYPE* convertMMToCRMatrix( const MM_TYPE* mm );


void crColIdToFortran( CR_TYPE* cr );
void crColIdToC( CR_TYPE* cr );

void freeMMMatrix( MM_TYPE* const mm );

void readCRbinFile(CR_TYPE*, const char* );
void readCRrowsBinFile(CR_TYPE* cr, const char* path);

int compareNZEPerRow( const void*, const void*);
BJDS_TYPE * CRStoBJDS(CR_TYPE *cr);
BJDS_TYPE * CRStoSBJDS(CR_TYPE *cr, int **rowPerm, int **invRowPerm); 
BJDS_TYPE * CRStoTBJDS(CR_TYPE *cr); 
int pad(int nRows, int padding);

#endif /* _MATRICKS_H_ */
