#ifndef _MATRICKS_H_
#define _MATRICKS_H_

#include "spmvm.h"
#include "spm_format_crs.h"
#include "spm_format_bjds.h"
#include "spm_format_tbjds.h"

#include <stdio.h>
#include <stdlib.h>

typedef struct
{
	mat_idx_t len;
	mat_idx_t idx;
	mat_data_t val;
	mat_idx_t minRow;
	mat_idx_t maxRow;
}
CONST_DIAG;


typedef struct 
{
	mat_idx_t nrows, ncols;
	mat_nnz_t nEnts;
	mat_idx_t*        rpt;
	mat_idx_t*        col;
	mat_data_t* val;

	mat_idx_t nConstDiags;
	CONST_DIAG *constDiags;
} 
CR_TYPE;

typedef struct 
{
	mat_data_t *val;
	mat_idx_t *col;
	mat_nnz_t *chunkStart;
	mat_idx_t *chunkMin; // for version with remainder loop
	mat_idx_t *chunkLen; // for version with remainder loop
	mat_idx_t *rowLen;   // for version with remainder loop
	mat_idx_t nrows;
	mat_idx_t nrowsPadded;
	mat_nnz_t nnz;
	mat_nnz_t nEnts;
	double nu;
} 
BJDS_TYPE;

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



MM_TYPE* readMMFile( const char* filename );

CR_TYPE* convertMMToCRMatrix( const MM_TYPE* mm );
void SpMVM_freeCRS( CR_TYPE* const cr );


void crColIdToFortran( CR_TYPE* cr );
void crColIdToC( CR_TYPE* cr );

void freeMMMatrix( MM_TYPE* const mm );

CR_TYPE * readCRbinFile(const char*, int, int);
ghost_mat_t * SpMVM_createMatrixFromCRS(CR_TYPE *cr, mat_trait_t trait);

int compareNZEPerRow( const void*, const void*);
void CRStoBJDS(CR_TYPE *cr, mat_trait_t, ghost_mat_t **matrix);
void CRStoTBJDS(CR_TYPE *cr, mat_trait_t trait, ghost_mat_t **matrix);
void CRStoCRS(CR_TYPE *cr, mat_trait_t trait, ghost_mat_t **matrix);
int pad(int nrows, int padding);

#endif /* _MATRICKS_H_ */
