/**
 * @file matrixmarket.h
 * @brief Functionality for Market Market file read-in.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_MATRIXMARKET_H
#define GHOST_MATRIXMARKET_H

#include "types.h"

#define GHOST_SPARSEMAT_ROWFUNC_MM_ROW_INIT -1
#define GHOST_SPARSEMAT_ROWFUNC_MM_ROW_FINALIZE -2
#define GHOST_SPARSEMAT_ROWFUNC_MM_ROW_GETRPT -3
#define GHOST_SPARSEMAT_ROWFUNC_MM_ROW_GETDIM -4

typedef struct 
{
    char *filename;
    ghost_datatype dt;
} ghost_sparsemat_rowfunc_mm_initargs;

/**
 * @brief 
 *
 * @param row
 * @param rowlen
 * @param col
 * @param val
 *
 * @return 
 *
 * If called with row #GHOST_SPARSEMAT_ROWFUNC_MM_ROW_INIT, the parameter
 * \p val has to be a ghost_sparsemat_rowfunc_mm_initargs * with the according
 * information filled in. The parameter \p col has to be a ghost_gidx[2] in 
 * which the number of rows and columns will be stored.
 */
int ghost_sparsemat_rowfunc_mm(ghost_gidx row, ghost_lidx *rowlen, ghost_gidx *col, void *val, void *arg);

int ghost_sparsemat_rowfunc_mm_transpose(ghost_gidx row, ghost_lidx *rowlen, ghost_gidx *col, void *val, void *arg);

#endif
