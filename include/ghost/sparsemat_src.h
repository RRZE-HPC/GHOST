/**
 * @file sparsemat_src.h
 * @brief GHOST sparsemat sources.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_SPARSEMAT_SRC_H
#define GHOST_SPARSEMAT_SRC_H

/**
 * @brief Callback function to construct a ghost_sparsemat
 *
 * @param[in] row The global row index.
 * @param[out] nnz The number of values in this row.
 * @param[out] val The values in the specified row.
 * @param[out] col The column indices of the given values.
 * @param[inout] arg Additional arguments.
 *
 * @return  
 */
typedef int (*ghost_sparsemat_rowfunc)(ghost_gidx row, ghost_lidx *nnz, ghost_gidx *col, void *val, void *arg);

/**
 * @brief Flags to be passed to a row-wise matrix assembly function.
 */
typedef enum {
    /**
     * @brief Default behaviour.
     */
    GHOST_SPARSEMAT_ROWFUNC_DEFAULT = 0
} ghost_sparsemat_rowfunc_flags;

/**
 * @brief Defines a rowfunc-based sparsemat source.
 */
typedef struct {
    /**
     * @brief The callback function which assembled the matrix row-wise.
     * @note The function func may be called several times for each row concurrently by multiple threads.
     */
    ghost_sparsemat_rowfunc func;
    /**
     * @brief Maximum row length of the matrix.
     */
    ghost_lidx maxrowlen;
    /**
     * @brief 0 for C, 1 for Fortran-like indexing.
     */
    int base;
    /**
     * @brief Flags to the row function.
     */
    ghost_sparsemat_rowfunc_flags flags;
    ghost_gidx gnrows;
    ghost_gidx gncols;
    void *arg;
} ghost_sparsemat_src_rowfunc;

extern const ghost_sparsemat_src_rowfunc GHOST_SPARSEMAT_SRC_ROWFUNC_INITIALIZER;



#endif
