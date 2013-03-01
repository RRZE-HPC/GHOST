
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
#elif defined (OPENCL) || defined (CUDA)
#define BJDS_LEN 256
#elif defined (VSX)
#define BJDS_LEN 2
#else
#define BJDS_LEN 4
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
#ifdef CUDA
	ghost_cu_dt * val;
	ghost_midx_t * col;
	ghost_midx_t * rowLen;
	ghost_mnnz_t * chunkStart;
	ghost_midx_t * chunkLen;
	ghost_midx_t nrows;
	ghost_midx_t nrowsPadded;
#endif
} 
CU_BJDS_TYPE;

typedef struct 
{
	ghost_dt *val;
	ghost_midx_t *col;
	ghost_mnnz_t *chunkStart;
	ghost_midx_t nrows;
	ghost_midx_t nrowsPadded;
	ghost_mnnz_t nnz;
	ghost_mnnz_t nEnts;
	double nu;
	double mu;
	ghost_midx_t *chunkMin; // for version with remainder loop
	ghost_midx_t *chunkLen; // for version with remainder loop
	ghost_midx_t *rowLen;   // for version with remainder loop
	
	CL_BJDS_TYPE *clmat;
	CU_BJDS_TYPE *cumat;
} 
BJDS_TYPE;

ghost_mat_t * init(ghost_mtraits_t *);

#endif
