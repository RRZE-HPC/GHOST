/**
 * @file sparsemat.h
 * @brief Types and functions related to sparse matrices.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_SPARSEMAT_H
#define GHOST_SPARSEMAT_H

#include "config.h"
#include "types.h"
#include "spmv.h"
#include "context.h"
#include "densemat.h"

#include <stdarg.h>

#define GHOST_SPARSEMAT_SORT_GLOBAL -1
#define GHOST_SPARSEMAT_SORT_LOCAL -2

/**
 * @brief Available sparse matrix storage formats.
 */
typedef enum {
    /**
     * @brief The CRS data format.
     */
    GHOST_SPARSEMAT_CRS,
    /**
     * @brief The SELL (Sliced ELLPACK) data format.
     */
    GHOST_SPARSEMAT_SELL
} ghost_sparsemat_format_t;


typedef struct 
{
    ghost_idx_t row, nEntsInRow;
} 
ghost_sorting_t;


/**
 * @brief Possible sparse matrix symmetries.
 */
typedef enum {
    /**
     * @brief Non-symmetric (general) matrix.
     */
    GHOST_SPARSEMAT_SYMM_GENERAL = 1,
    /**
     * @brief Symmetric matrix.
     */
    GHOST_SPARSEMAT_SYMM_SYMMETRIC = 2,
    /**
     * @brief Skew-symmetric matrix.
     */
    GHOST_SPARSEMAT_SYMM_SKEW_SYMMETRIC = 4,
    /**
     * @brief Hermitian matrix.
     */
    GHOST_SPARSEMAT_SYMM_HERMITIAN = 8
} ghost_sparsemat_symmetry_t;

    

typedef int (*ghost_sparsemat_fromRowFunc_t)(ghost_idx_t, ghost_idx_t *, ghost_idx_t *, void *);
typedef struct ghost_sparsemat_traits_t ghost_sparsemat_traits_t;
typedef struct ghost_sparsemat_t ghost_sparsemat_t;

/**
 * @brief Flags to be passed to a row-wise matrix assembly function.
 */
typedef enum {
    /**
     * @brief Default behaviour.
     */
    GHOST_SPARSEMAT_FROMROWFUNC_DEFAULT = 0
} ghost_sparsemat_fromRowFunc_flags_t;

typedef struct 
{
    ghost_sparsemat_fromRowFunc_t func;
    ghost_idx_t maxrowlen;
    int base;
    ghost_sparsemat_fromRowFunc_flags_t flags;
} ghost_sparsemat_src_rowfunc_t;

#define GHOST_SPARSEMAT_SRC_ROWFUNC_INITIALIZER {\
    .func = NULL,\
    .maxrowlen = 0,\
    .base = 0,\
    .flags = GHOST_SPARSEMAT_FROMROWFUNC_DEFAULT\
}

/**
 * @brief Flags to a sparse matrix.
 */
typedef enum {
    /**
     * @brief A default sparse matrix.
     */
    GHOST_SPARSEMAT_DEFAULT       = 0,
    /**
     * @brief Matrix is stored on host.
     */
    GHOST_SPARSEMAT_HOST          = 1,
    /**
     * @brief Matrix is store on device.
     */
    GHOST_SPARSEMAT_DEVICE        = 2,
    /**
     * @brief If the matrix rows have been re-ordered, do _NOT_ permute the column indices accordingly.
     */
    GHOST_SPARSEMAT_NOT_PERMUTE_COLS = 4,
    /**
     * @brief The matrix rows should be re-ordered in a certain way (defined in the traits). 
     */
    GHOST_SPARSEMAT_PERMUTE       = 8,
    /**
     * @brief The permutation is global, i.e., across processes. 
     */
    //GHOST_SPARSEMAT_PERMUTE_GLOBAL  = 256,
    /**
     * @brief If the matrix columns have been re-ordered, do _NOT_ care for ascending column indices wrt. memory location. 
     */
    GHOST_SPARSEMAT_NOT_SORT_COLS    = 16,
    /**
     * @brief Do _NOT_ store the local and remote part of the matrix.
     */
    GHOST_SPARSEMAT_NOT_STORE_SPLIT = 32,
    /**
     * @brief Do _NOT_ store the full matrix (local and remote combined).
     */
    GHOST_SPARSEMAT_NOT_STORE_FULL = 64,
    /**
     * @brief Reduce the matrix bandwidth with PT-Scotch
     */
    GHOST_SPARSEMAT_SCOTCHIFY = 128
} ghost_sparsemat_flags_t;


/**
 * @brief Sparse matrix traits.
 */
struct ghost_sparsemat_traits_t
{
    ghost_sparsemat_format_t format;
    ghost_sparsemat_flags_t flags;
    ghost_sparsemat_symmetry_t symmetry;
    /**
     * @brief Auxiliary matrix traits (to be interpreted by the concrete format implementation).
     */
    void * aux;
    char * scotchStrat;
    ghost_idx_t sortScope;
    ghost_datatype_t datatype;
};

#ifdef GHOST_HAVE_MPI
#define GHOST_SCOTCH_STRAT_DEFAULT "n{ole=q{strat=g},ose=q{strat=g},osq=g}"
#else
#define GHOST_SCOTCH_STRAT_DEFAULT "g"
#endif

/**
 * @brief Initialize sparse matrix traits with default values.
 */
#define GHOST_SPARSEMAT_TRAITS_INITIALIZER {\
    .flags = GHOST_SPARSEMAT_DEFAULT,\
    .aux = NULL,\
    .datatype = (ghost_datatype_t) (GHOST_DT_DOUBLE|GHOST_DT_REAL),\
    .sortScope = 1,\
    .format = GHOST_SPARSEMAT_CRS,\
    .symmetry = GHOST_SPARSEMAT_SYMM_GENERAL,\
    .scotchStrat = GHOST_SCOTCH_STRAT_DEFAULT\
};

/**
 * @ingroup types
 *
 * @brief A sparse matrix.
 * 
 * The according functions act locally and are accessed via function pointers. The first argument of
 * each member function always has to be a pointer to the vector itself.
 */
struct ghost_sparsemat_t
{
    ghost_sparsemat_traits_t *traits;
    ghost_sparsemat_t *localPart;
    ghost_sparsemat_t *remotePart;
    ghost_context_t *context;
    char *name;
    void *data;
    
    /**
     * @brief Size (in bytes) of one matrix element.
     */
    size_t elSize;

    ghost_permutation_t *permutation;
    //ghost_idx_t *rowPerm;
    //ghost_idx_t *invRowPerm;
    
    ghost_idx_t nrows;
    ghost_idx_t ncols;
    ghost_idx_t nrowsPadded;
    ghost_nnz_t nnz;
    ghost_idx_t lowerBandwidth;
    ghost_idx_t upperBandwidth;
    ghost_idx_t bandwidth;
    ghost_idx_t maxRowLen;
    ghost_idx_t nMaxRows;
    double variance; // row length variance
    double deviation; // row lenght standard deviation
    double cv; // row lenght coefficient of variation
    /**
     * @brief Array of length nrows with nzDist[i] = number nonzeros with distance i from diagonal
     */
    ghost_nnz_t *nzDist;
    ghost_nnz_t nEnts;

    /**
     * @brief Permute the matrix rows and column indices (if set in mat->traits->flags) with the given permutation.
     *
     * @param mat The matrix.
     * @param perm The permutation vector.
     * @param invPerm The inverse permutation vector.
     */
    ghost_error_t (*permute) (ghost_sparsemat_t *mat, ghost_idx_t *perm, ghost_idx_t *invPerm);
    /**
     * @brief Calculate y = gamma * (A - I*alpha) * x + beta * y.
     *
     * @param mat The matrix A. 
     * @param res The vector y.
     * @param rhs The vector x.
     * @param flags A number of flags to control the operation's behaviour.
     *
     * For detailed information on the flags check the documentation of ghost_spmv_flags_t.
     */
    ghost_error_t (*spmv) (ghost_sparsemat_t *mat, ghost_densemat_t *res, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp);
    /**
     * @brief Destroy the matrix, i.e., free all of its data.
     *
     * @param mat The matrix.
     *
     * Returns if the matrix is NULL.
     */
    void       (*destroy) (ghost_sparsemat_t *mat);
    /**
     * @brief Prints specific information on the matrix.
     *
     * @param mat The matrix.
     * @param str Where to store the string.
     *
     * This function is called in ghost_printMatrixInfo() to print format-specific information alongside with
     * general matrix information.
     */
    void       (*auxString) (ghost_sparsemat_t *mat, char **str);
    /**
     * @brief Turns the matrix into a string.
     *
     * @param mat The matrix.
     * @param str Where to store the string.
     * @param dense If 0, only the elements stored in the sparse matrix will be included.
     * If 1, the matrix will be interpreted as a dense matrix.
     *
     * @return The stringified matrix.
     */
    ghost_error_t (*string) (ghost_sparsemat_t *mat, char **str, int dense);
    /**
     * @brief Get the length of the given row.
     *
     * @param mat The matrix.
     * @param row The row.
     *
     * @return The length of the row or zero if the row index is out of bounds. 
     */
    ghost_idx_t  (*rowLen) (ghost_sparsemat_t *mat, ghost_idx_t row);
    /**
     * @brief Return the name of the storage format.
     *
     * @param mat The matrix.
     *
     * @return A string containing the storage format name. 
     */
    const char *  (*formatName) (ghost_sparsemat_t *mat);
    /**
     * @brief Create the matrix from a matrix file in GHOST's binary CRS format.
     *
     * @param mat The matrix. 
     * @param path Path to the file.
     */
    ghost_error_t (*fromFile)(ghost_sparsemat_t *mat, char *path);
    /**
     * @brief Create the matrix from a function which defined the matrix row by row.
     *
     * @param mat The matrix.
     * @param src The source.
     *
     * The function func may be called several times for each row concurrently by multiple threads.
     */
    ghost_error_t (*fromRowFunc)(ghost_sparsemat_t *, ghost_sparsemat_src_rowfunc_t *src);
    //ghost_idx_t maxrowlen, int base, ghost_sparsemat_fromRowFunc_t func, ghost_sparsemat_fromRowFunc_flags_t flags);
    /**
     * @brief Write a matrix to a binary CRS file.
     *
     * @param mat The matrix. 
     * @param path Path of the file.
     */
    ghost_error_t (*toFile)(ghost_sparsemat_t *mat, char *path);
    /**
     * @brief Upload the matrix to the CUDA device.
     *
     * @param mat The matrix.
     */
    ghost_error_t (*upload)(ghost_sparsemat_t * mat);
    /**
     * @brief Get the entire memory footprint of the matrix.
     *
     * @param mat The matrix.
     *
     * @return The memory footprint of the matrix in bytes or zero if the matrix is not valid.
     */
    size_t     (*byteSize)(ghost_sparsemat_t *mat);
    /**
     * @brief Create a matrix from a CRS matrix.
     *
     * @param mat The matrix. 
     * @param crsMat The CRS matrix.
     */
    ghost_error_t     (*fromCRS)(ghost_sparsemat_t *mat, ghost_sparsemat_t *crsMat);
    /**
     * @brief Split the matrix into a local and a remote part.
     *
     * @param mat The matrix.
     */
    ghost_error_t       (*split)(ghost_sparsemat_t *mat);
};


#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @ingroup types
     *
     * @brief Create a sparse matrix. 
     *
     * @param mat Where to store the matrix
     * @param ctx The context the matrix lives in.
     * @param traits The matrix traits. They can be specified for the full matrix, the local and the remote part.
     * @param nTraits The number of traits. 
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_sparsemat_create(ghost_sparsemat_t **mat, ghost_context_t *ctx, ghost_sparsemat_traits_t *traits, int nTraits);
    ghost_error_t ghost_sparsemat_string(char **str, ghost_sparsemat_t *matrix);
    ghost_error_t ghost_sparsemat_nnz(ghost_nnz_t *nnz, ghost_sparsemat_t *mat);
    ghost_error_t ghost_sparsemat_nrows(ghost_idx_t *nrows, ghost_sparsemat_t *mat);
    char * ghost_sparsemat_symmetry_string(int symmetry);
    bool ghost_sparsemat_symmetry_valid(int symmetry);
    ghost_error_t ghost_sparsemat_perm_scotch(ghost_sparsemat_t *mat, void *matrixSource, ghost_sparsemat_src_t srcType);
    ghost_error_t ghost_sparsemat_perm_sort(ghost_sparsemat_t *mat, void *matrixSource, ghost_sparsemat_src_t srcType, ghost_idx_t scope);
    ghost_error_t ghost_sparsemat_sortrow(ghost_idx_t *col, char *val, size_t valSize, ghost_idx_t rowlen, ghost_idx_t stride);
    /**
     * @brief Common function for matrix creation from a file.
     *
     * @param mat
     * @param matrixPath
     *
     * @return 
     */
    ghost_error_t ghost_sparsemat_fromfile_common(ghost_sparsemat_t *mat, char *matrixPath, ghost_idx_t **rpt);
    ghost_error_t ghost_sparsemat_fromfunc_common(ghost_sparsemat_t *mat, ghost_sparsemat_src_rowfunc_t *src);
    ghost_error_t ghost_sparsemat_registerrow(ghost_sparsemat_t *mat, ghost_idx_t row, ghost_idx_t *col, ghost_idx_t rowlen, ghost_idx_t stride);
    ghost_error_t ghost_sparsemat_registerrow_finalize(ghost_sparsemat_t *mat);

#ifdef __cplusplus
} extern "C"
#endif

#endif
