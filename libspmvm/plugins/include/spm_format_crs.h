#ifndef __GHOST_SPMFORMAT_CRS__
#define __GHOST_SPMFORMAT_CRS__

#include "ghost.h"

typedef struct
{
	ghost_midx_t len;
	ghost_midx_t idx;
	ghost_mdat_t val;
	ghost_midx_t minRow;
	ghost_midx_t maxRow;
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
	ghost_midx_t  nrows, ncols;
	ghost_mnnz_t  nEnts;
	ghost_midx_t  *rpt;
	ghost_midx_t  *col;
	ghost_mdat_t *val;

	CL_CR_TYPE *clmat;
	ghost_midx_t nConstDiags;
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

