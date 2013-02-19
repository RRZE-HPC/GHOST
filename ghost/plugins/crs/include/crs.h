#ifndef __GHOST_SPMFORMAT_CRS__
#define __GHOST_SPMFORMAT_CRS__

#include "ghost.h"

typedef struct
{
	ghost_midx_t len;
	ghost_midx_t idx;
	ghost_dt val;
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
	ghost_dt *val;

	CL_CR_TYPE *clmat;
	ghost_midx_t nConstDiags;
	CONST_DIAG *constDiags;
} 
CR_TYPE;

typedef struct 
{
	ghost_midx_t row, col, nThEntryInRow;
	ghost_dt val;
} 
NZE_TYPE;



ghost_mat_t * init(ghost_mtraits_t *);


#endif

