#ifndef __GHOST_SPMFORMAT_TBJDS__
#define __GHOST_SPMFORMAT_TBJDS__

#include "ghost.h"

typedef struct 
{
	ghost_mdat_t *val;
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
TBJDS_TYPE;

void init(ghost_mat_t **);

#endif
