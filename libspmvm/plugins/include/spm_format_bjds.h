
#ifndef __GHOST_SPMFORMAT_BJDS__
#define __GHOST_SPMFORMAT_BJDS__

#include "ghost.h"

#ifdef MIC
//#define BJDS_LEN 8
#define BJDS_LEN 16
#elif defined (AVX)
#define BJDS_LEN 4 // TODO single/double precision
#elif defined (SSE)
#define BJDS_LEN 2
#elif defined (OPENCL)
#define BJDS_LEN 256
#else
#define BJDS_LEN 1
#endif

typedef struct 
{
#ifdef OPENCL
	cl_mem val;
	cl_mem col;
	cl_mem rowLen;
	cl_mem chunkStart;
	cl_mem chunkLen;
	ghost_cl_midx_t nrows;
	ghost_cl_midx_t nrowsPadded;
#endif
} 
CL_BJDS_TYPE;

typedef struct 
{
	ghost_mdat_t *val;
	mat_idx_t *col;
	mat_nnz_t *chunkStart;
	mat_idx_t nrows;
	mat_idx_t nrowsPadded;
	mat_nnz_t nnz;
	mat_nnz_t nEnts;
	double nu;
	mat_idx_t *chunkMin; // for version with remainder loop
	mat_idx_t *chunkLen; // for version with remainder loop
	mat_idx_t *rowLen;   // for version with remainder loop
	
	CL_BJDS_TYPE *clmat;
} 
BJDS_TYPE;

ghost_mat_t * init(ghost_mtraits_t *);

#endif
