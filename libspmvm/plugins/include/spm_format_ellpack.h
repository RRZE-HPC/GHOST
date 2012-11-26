#ifndef __GHOST_SPMFORMAT_ELLPACK__
#define __GHOST_SPMFORMAT_ELLPACK__

#include "ghost.h"

#define ELLPACK_PAD 512

typedef struct 
{
	ghost_mdat_t *val;
	mat_idx_t *col;
	mat_idx_t nrows;
	mat_idx_t nrowsPadded;
	mat_nnz_t nnz;
	mat_nnz_t nEnts;
	mat_idx_t *rowLen;
	mat_idx_t maxRowLen;
} 
ELLPACK_TYPE;

void init(ghost_mat_t **);

#endif
