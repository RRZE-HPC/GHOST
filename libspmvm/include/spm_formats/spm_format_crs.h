#ifndef __GHOST_SPMFORMAT_CRS__
#define __GHOST_SPMFORMAT_CRS__

#include "spmvm.h"

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
	mat_idx_t  nrows, ncols;
	mat_nnz_t  nEnts;
	mat_idx_t  *rpt;
	mat_idx_t  *col;
	mat_data_t *val;

	mat_idx_t nConstDiags;
	CONST_DIAG *constDiags;
} 
CR_TYPE;

void CRS_init(ghost_mat_t *mat);

#endif

