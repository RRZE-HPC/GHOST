#ifndef __GHOST_SPMFORMAT_TBJDS__
#define __GHOST_SPMFORMAT_TBJDS__

#include "ghost.h"

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
