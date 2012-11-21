#ifndef __GHOST_SPMFORMAT_CRS__
#define __GHOST_SPMFORMAT_CRS__

#include "ghost.h"

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
#ifdef OPENCL
	ghost_cl_midx_t  nrows, ncols;
	ghost_cl_mnnz_t  nEnts;
	cl_mem rpt;
	cl_mem col;
	cl_mem val;
#endif
} 
CL_CR_TYPE;

typedef struct 
{
	mat_idx_t  nrows, ncols;
	mat_nnz_t  nEnts;
	mat_idx_t  *rpt;
	mat_idx_t  *col;
	mat_data_t *val;

	CL_CR_TYPE *clmat;
	mat_idx_t nConstDiags;
	CONST_DIAG *constDiags;
} 
CR_TYPE;


void init(ghost_mat_t *);

#endif

