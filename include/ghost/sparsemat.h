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
    ghost_lidx_t row, nEntsInRow;
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

    

typedef int (*ghost_sparsemat_fromRowFunc_t)(ghost_gidx_t, ghost_lidx_t *, ghost_gidx_t *, void *);
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
    ghost_lidx_t maxrowlen;
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
    GHOST_SPARSEMAT_SCOTCHIFY = 128,
    GHOST_SPARSEMAT_SAVE_ORIG_COLS = 256
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
    ghost_gidx_t sortScope;
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
    .format = GHOST_SPARSEMAT_CRS,\
    .flags = GHOST_SPARSEMAT_DEFAULT,\
    .symmetry = GHOST_SPARSEMAT_SYMM_GENERAL,\
    .aux = NULL,\
    .scotchStrat = (char*)GHOST_SCOTCH_STRAT_DEFAULT,\
    .sortScope = 1,\
    .datatype = (ghost_datatype_t) (GHOST_DT_DOUBLE|GHOST_DT_REAL)\
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

    ghost_gidx_t *col_orig;

    ghost_lidx_t nrows;
    ghost_gidx_t ncols;
    ghost_lidx_t nrowsPadded;
    ghost_lidx_t nnz;
    ghost_gidx_t lowerBandwidth;
    ghost_gidx_t upperBandwidth;
    ghost_gidx_t bandwidth;
    ghost_gidx_t maxRowLen;
    ghost_gidx_t nMaxRows;
    double variance; // row length variance
    double deviation; // row lenght standard deviation
    double cv; // row lenght coefficient of variation
    /**
     * @brief Array of length (2*nrows-1) with nzDist[i] = number nonzeros with distance i from diagonal
     */
    ghost_gidx_t *nzDist;
    ghost_lidx_t nEnts;

    /**
     * @brief Permute the matrix rows and column indices (if set in mat->traits->flags) with the given permutation.
     *
     * @param mat The matrix.
     * @param perm The permutation vector.
     * @param invPerm The inverse permutation vector.
     */
    ghost_error_t (*permute) (ghost_sparsemat_t *mat, ghost_lidx_t *perm, ghost_lidx_t *invPerm);
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
    ghost_lidx_t  (*rowLen) (ghost_sparsemat_t *mat, ghost_lidx_t row);
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
     * @param[out] mat Where to store the matrix
     * @param[in] ctx The context the matrix lives in.
     * @param[in] traits The matrix traits. They can be specified for the full matrix, the local and the remote part.
     * @param[in] nTraits The number of traits. 
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_sparsemat_create(ghost_sparsemat_t **mat, ghost_context_t *ctx, ghost_sparsemat_traits_t *traits, int nTraits);
    /**
     * @ingroup stringification
     *
     * @brief Create a string holding information about the sparsemat
     *
     * @param[out] str Where to store the string.
     * @param[in] matrix The sparse matrix.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_sparsemat_string(char **str, ghost_sparsemat_t *matrix);
    /**
     * @brief Obtain the global number of nonzero elements of a sparse matrix.
     *
     * @param[out] nnz Where to store the result.
     * @param[in] mat The sparse matrix.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_sparsemat_nnz(ghost_gidx_t *nnz, ghost_sparsemat_t *mat);
    /**
     * @brief Obtain the global number of rows of a sparse matrix.
     
     * @param[out] nnz Where to store the result.
     * @param[in] mat The sparse matrix.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_sparsemat_nrows(ghost_gidx_t *nrows, ghost_sparsemat_t *mat);
    /**
     * @ingroup stringification
     *
     * @brief Convert the matrix' symmetry information to a string.
     *
     * @param[in] symmetry The symmetry information.
     *
     * @return A string holding the symmetry information.
     */
    char * ghost_sparsemat_symmetry_string(ghost_sparsemat_symmetry_t symmetry);
    /**
     * @brief Check if the symmetry information of a sparse matrix is valid.
     *
     * @param[in] symmetry The symmetry information.
     *
     * @return True if it is valid, false otherwise.
     */
    bool ghost_sparsemat_symmetry_valid(ghost_sparsemat_symmetry_t symmetry);
    /**
     * @brief Create a matrix permutation based on (PT-)SCOTCH
     *
     * @param[inout] mat The sparse matrix.
     * @param[in] matrixSource The matrix source. This will be casted depending on \p srcType.
     * @param[in] srcType Type of the matrix source.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_sparsemat_perm_scotch(ghost_sparsemat_t *mat, void *matrixSource, ghost_sparsemat_src_t srcType);
    ghost_error_t ghost_sparsemat_perm_sort(ghost_sparsemat_t *mat, void *matrixSource, ghost_sparsemat_src_t srcType, ghost_gidx_t scope);
    ghost_error_t ghost_sparsemat_sortrow(ghost_gidx_t *col, char *val, size_t valSize, ghost_lidx_t rowlen, ghost_lidx_t stride);
    /**
     * @brief Common function for matrix creation from a file.
     *
     * This function does work which is independent of the storage format and it should be called 
     * at the beginning of a sparse matrix' fromFile function.
     * This function also reads the row pointer from the file and stores it into the parameter rpt.
     *
     * @param[in] mat The sparse matrix.
     * @param[in] matrixPath The matrix file path.
     * @param[out] rpt Where to store the row pointer information which may be needed afterwards.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_sparsemat_fromfile_common(ghost_sparsemat_t *mat, char *matrixPath, ghost_lidx_t **rpt) ;
    /**
     * @brief Common function for matrix creation from a function.
     *
     * This function does work which is independent of the storage format and it should be called 
     * at the beginning of a sparse matrix' fromFunc function.
     *
     * @param[in] mat The sparse matrix.
     * @param[in] src The matrix source function.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_sparsemat_fromfunc_common(ghost_sparsemat_t *mat, ghost_sparsemat_src_rowfunc_t *src);
    /**
     * @ingroup io
     *
     * @brief Write the sparse matrix header to a binary CRS file.
     *
     * @param[in] mat The sparse matrix.
     * @param[in] path Path of the matrix file.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_sparsemat_tofile_header(ghost_sparsemat_t *mat, char *path);
    /**
     * @brief Store matrix information like bandwidth and nonzero distribution for a given matrix row.
     *
     * This function should be called at matrix creation for each row.
     *
     * @param[out] mat The matrix.
     * @param[in] row The row index.
     * @param[in] col The column indices of the row.
     * @param[in] rowlen The length of the row.
     * @param[in] stride The stride for the parameter col.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_sparsemat_registerrow(ghost_sparsemat_t *mat, ghost_gidx_t row, ghost_gidx_t *col, ghost_lidx_t rowlen, ghost_lidx_t stride);
    /**
     * @brief Finalize the storing of matrix information like bandwidth and nonzero distribution.
     *
     * This function should be after matrix creation.
     * It finalizes the processing of information obtained in ghost_sparsemat_registerrow.
     *
     * @param[out] mat The matrix.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_sparsemat_registerrow_finalize(ghost_sparsemat_t *mat);

    /**
     * @brief Common (independent of the storage formats) destruction of matrix data.
     *
     * @param[inout] mat The matrix.
     */
    void ghost_sparsemat_destroy_common(ghost_sparsemat_t *mat);

#ifdef __cplusplus
} extern "C"
#endif

#endif
