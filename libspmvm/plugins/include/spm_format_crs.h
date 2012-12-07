#ifndef __GHOST_SPMFORMAT_CRS__
#define __GHOST_SPMFORMAT_CRS__

#include "ghost.h"

typedef struct
{
	mat_idx_t len;
	mat_idx_t idx;
	ghost_mdat_t val;
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
	ghost_mdat_t *val;

	CL_CR_TYPE *clmat;
	mat_idx_t nConstDiags;
	CONST_DIAG *constDiags;
} 
CR_TYPE;

typedef struct{
	ghost_mat_t *mat;
	char *matrixPath;
} CRS_readRpt_args_t;

typedef struct{
	ghost_mat_t *mat;
	char *matrixPath;
	size_t offsetEnts;
	size_t offsetRows;
	size_t nRows;
	size_t nEnts;
	int IOtype;
} CRS_readColValOffset_args_t;

ghost_mat_t * init(ghost_mtraits_t *);

#endif

