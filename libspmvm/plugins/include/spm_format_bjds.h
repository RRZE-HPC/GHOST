
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
#else
#define BJDS_LEN 1
#endif

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
} 
BJDS_TYPE;

void init(ghost_mat_t **);

#endif
