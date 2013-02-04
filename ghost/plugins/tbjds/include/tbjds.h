#ifndef __GHOST_SPMFORMAT_TBJDS__
#define __GHOST_SPMFORMAT_TBJDS__

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
#define BJDS_LEN 1
#endif

typedef struct 
{
	ghost_mdat_t *val;
	ghost_midx_t *col;
	ghost_mnnz_t *chunkStart;
	ghost_midx_t *chunkMin; // for version with remainder loop
	ghost_midx_t *chunkLen; // for version with remainder loop
	ghost_midx_t *rowLen;   // for version with remainder loop
	ghost_midx_t nrows;
	ghost_midx_t nrowsPadded;
	ghost_mnnz_t nnz;
	ghost_mnnz_t nEnts;
	double nu;
} 
TBJDS_TYPE;

ghost_mat_t * init(ghost_mtraits_t *);

#endif
