#ifndef INIT_H
#define INIT_H

#include "ghost/sparsemat.h"

#ifdef __cplusplus
extern "C" {
#endif
void initChunkPtr(ghost_sparsemat *mat);
void initRowLen(ghost_sparsemat *mat);
void initVal(ghost_sparsemat *val);
void initCol(ghost_sparsemat *col);
void reInit(ghost_sparsemat *mat);
#ifdef __cplusplus
}
#endif

#endif
