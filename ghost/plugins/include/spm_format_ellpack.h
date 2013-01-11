#ifndef __GHOST_SPMFORMAT_ELLPACK__
#define __GHOST_SPMFORMAT_ELLPACK__

#include "ghost.h"

#define ELLPACK_PAD 1024
#define ELLPACK_WGXSIZE 256

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
	ghost_midx_t *col;
	ghost_midx_t *rowLen;
	ghost_midx_t nrows;
	ghost_midx_t nrowsPadded;
	ghost_mnnz_t nnz;
	ghost_mnnz_t nEnts;
	ghost_midx_t maxRowLen;
	ghost_midx_t T;

	CL_ELLPACK_TYPE *clmat;
} 
ELLPACK_TYPE;

ghost_mat_t * init(ghost_mtraits_t *);

#endif
