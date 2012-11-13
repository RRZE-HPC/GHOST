#ifndef _MATRICKS_H_
#define _MATRICKS_H_

#include "spmvm.h"

#include <stdio.h>
#include <stdlib.h>


typedef struct 
{
	mat_idx_t row, col, nThEntryInRow;
	mat_data_t val;
} 
NZE_TYPE;

typedef struct 
{
	mat_idx_t nrows, ncols;
	mat_nnz_t nEnts;
	NZE_TYPE* nze;
} 
MM_TYPE;

typedef struct 
{
	mat_idx_t row, nEntsInRow;
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

CR_TYPE * readCRbinFile(const char*, int, int);

int compareNZEPerRow( const void*, const void*);
void CRStoBJDS(CR_TYPE *cr, mat_trait_t, MATRIX_TYPE **matrix);
void CRStoSBJDS(CR_TYPE *cr, mat_trait_t trait, MATRIX_TYPE **matrix);
void CRStoTBJDS(CR_TYPE *cr, mat_trait_t trait, MATRIX_TYPE **matrix);
void CRStoSTBJDS(CR_TYPE *cr, mat_trait_t trait, MATRIX_TYPE **matrix);
void CRStoCRS(CR_TYPE *cr, mat_trait_t trait, MATRIX_TYPE **matrix);
int pad(int nrows, int padding);

#endif /* _MATRICKS_H_ */
