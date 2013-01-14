#ifndef __GHOST_MAT_H__
#define __GHOST_MAT_H__

#include "ghost.h"
#include "ghost_spmformats.h"

#include <stdio.h>
#include <stdlib.h>

#define DIAG_NEW (char)0
#define DIAG_OK (char)1
#define DIAG_INVALID (char)2

typedef struct 
{
	ghost_midx_t row, col, nThEntryInRow;
	ghost_mdat_t val;
} 
NZE_TYPE;

typedef struct 
{
	ghost_midx_t nrows, ncols;
	ghost_mnnz_t nEnts;
	NZE_TYPE* nze;
} 
MM_TYPE;

typedef struct 
{
	ghost_midx_t row, nEntsInRow;
} 
JD_SORT_TYPE;

typedef unsigned long long uint64;

int isMMfile(const char *filename);



MM_TYPE* readMMFile( const char* filename );

CR_TYPE* convertMMToCRMatrix( const MM_TYPE* mm );
void ghost_freeCRS( CR_TYPE* const cr );

void freeMMMatrix( MM_TYPE* const mm );

CR_TYPE * readCRbinFile(const char*, int, int);
ghost_mat_t * ghost_createMatrixFromCRS(CR_TYPE *cr, ghost_mtraits_t trait);

int compareNZEPos( const void* a, const void* b ); 
int compareNZEOrgPos( const void* a, const void* b ); 
int compareNZEPerRow( const void*, const void*);
void CRStoBJDS(CR_TYPE *cr, ghost_mtraits_t, ghost_mat_t **matrix);
void CRStoTBJDS(CR_TYPE *cr, ghost_mtraits_t trait, ghost_mat_t **matrix);
void CRStoCRS(CR_TYPE *cr, ghost_mtraits_t trait, ghost_mat_t **matrix);
int pad(int nrows, int padding);

#endif
