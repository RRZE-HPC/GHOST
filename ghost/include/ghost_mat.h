#ifndef __GHOST_MAT_H__
#define __GHOST_MAT_H__

#include "ghost.h"
//#include "crs.h"

#include <stdio.h>
#include <stdlib.h>

#define DIAG_NEW (char)0
#define DIAG_OK (char)1
#define DIAG_INVALID (char)2

/*typedef struct 
{
	ghost_midx_t nrows, ncols;
	ghost_mnnz_t nEnts;
	NZE_TYPE* nze;
} 
ghost_mm_t;*/

typedef struct 
{
	ghost_midx_t row, nEntsInRow;
} 
ghost_sorting_t;

#ifdef __cplusplus
extern "C" {
#endif

int isMMfile(const char *filename);
//ghost_mm_t * readMMFile(const char* filename );

int compareNZEOrgPos( const void* a, const void* b ); 
int compareNZEPerRow( const void*, const void*);

#ifdef __cplusplus
}
#endif

#endif
