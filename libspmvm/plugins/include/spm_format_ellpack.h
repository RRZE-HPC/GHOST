#ifndef __GHOST_SPMFORMAT_ELLPACK__
#define __GHOST_SPMFORMAT_ELLPACK__

#include "ghost.h"

#define ELLPACK_PAD 32 // TODO

typedef struct 
{
#ifdef OPENCL
	cl_mem val;
	cl_mem col;
	cl_mem rowLen;
	ghost_cl_midx_t nrows;
	ghost_cl_midx_t nrowsPadded;
	ghost_cl_midx_t maxRowLen;
#endif
} 
CL_ELLPACK_TYPE;

typedef struct 
{
	ghost_mdat_t *val;
	mat_idx_t *col;
	mat_idx_t *rowLen;
	mat_idx_t nrows;
	mat_idx_t nrowsPadded;
	mat_nnz_t nnz;
	mat_nnz_t nEnts;
	mat_idx_t maxRowLen;

	CL_ELLPACK_TYPE *clmat;
} 
ELLPACK_TYPE;

ghost_mat_t * init(ghost_mtraits_t *);

#endif
