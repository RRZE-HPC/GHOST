/**
 * @file mat.h
 * @brief Types and functions related to GHOST sparse matrices.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_MAT_H
#define GHOST_MAT_H

#include "config.h"
#include "types.h"
#include "context.h"
#include "densemat.h"

#define GHOST_SPM_FORMAT_CRS 0
#define GHOST_SPM_FORMAT_SELL 1

typedef enum {
    GHOST_SPMVM_DEFAULT = 0,
    GHOST_SPMVM_AXPY = 1,
    GHOST_SPMVM_MODE_NOMPI = 2,
    GHOST_SPMVM_MODE_VECTORMODE = 4,
    GHOST_SPMVM_MODE_GOODFAITH = 8,
    GHOST_SPMVM_MODE_TASKMODE = 16,
    GHOST_SPMVM_APPLY_SHIFT = 32,
    GHOST_SPMVM_APPLY_SCALE = 64,
    GHOST_SPMVM_AXPBY = 128,
    GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT = 256
} ghost_spmv_flags_t;

typedef enum {
    GHOST_SPM_SYMM_GENERAL = 1,
    GHOST_SPM_SYMM_SYMMETRIC = 2,
    GHOST_SPM_SYMM_SKEW_SYMMETRIC = 4,
    GHOST_SPM_SYMM_HERMITIAN = 8
} ghost_spm_symmetry_t;


typedef void (*ghost_spmFromRowFunc_t)(ghost_midx_t, ghost_midx_t *, ghost_midx_t *, void *);
typedef struct ghost_sparsemat_traits_t ghost_sparsemat_traits_t;
typedef struct ghost_sparsemat_t ghost_sparsemat_t;

typedef enum {
    GHOST_SPMFROMROWFUNC_DEFAULT = 0
} ghost_spmFromRowFunc_flags_t;

/**
 * @brief Flags to a sparse matrix.
 */
typedef enum {
    /**
     * @brief A default sparse matrix.
     */
    GHOST_SPM_DEFAULT       = 0,
    /**
     * @brief Matrix is stored on host.
     */
    GHOST_SPM_HOST          = 1,
    /**
     * @brief Matrix is store on device.
     */
    GHOST_SPM_DEVICE        = 2,
    /**
     * @brief If the matrix rows have been re-ordered, also permute the column indices accordingly.
     */
    GHOST_SPM_PERMUTECOLIDX = 4,
    /**
     * @brief The matrix rows should be re-ordered in a certain way (defined in the traits). 
     */
    GHOST_SPM_SORTED        = 32,
    /**
     * @brief If the matrix columns have been re-ordered, care for ascending column indices in wrt. memory location. 
     */
    GHOST_SPM_ASC_COLIDX    = 64,
    /**
     * @brief Store the local and remote part of the matrix.
     */
    GHOST_SPM_STORE_SPLIT = 128,
    /**
     * @brief Store the full matrix (local and remote combined).
     */
    GHOST_SPM_STORE_FULL = 256
} ghost_spm_flags_t;

#define GHOST_SPMVM_MODES_FULL     (GHOST_SPMVM_MODE_NOMPI | GHOST_SPMVM_MODE_VECTORMODE)
#define GHOST_SPMVM_MODES_SPLIT    (GHOST_SPMVM_MODE_GOODFAITH | GHOST_SPMVM_MODE_TASKMODE)
#define GHOST_SPMVM_MODES_ALL      (GHOST_SPMVM_MODES_FULL | GHOST_SPMVM_MODES_SPLIT)

struct ghost_sparsemat_traits_t
{
    int format;
    ghost_spm_flags_t flags;
    ghost_spm_symmetry_t symmetry;
    void * aux;
    int nAux;
    int datatype;
    size_t elSize;
    void * shift;
    void * scale;
    void * beta; // scale factor for AXPBY
};
/**
 * @brief Initialize sparse matrix traits with default values as specified in mat.c
 */
extern const ghost_sparsemat_traits_t GHOST_MTRAITS_INITIALIZER;
#define GHOST_MTRAITS_INIT(...) {.flags = GHOST_SPM_DEFAULT, .aux = NULL, .nAux = 0, .datatype = GHOST_DT_DOUBLE|GHOST_DT_REAL, .format = GHOST_SPM_FORMAT_CRS, .shift = NULL, .scale = NULL, ## __VA_ARGS__ }

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

    ghost_midx_t nrows;
    ghost_midx_t ncols;
    ghost_midx_t nrowsPadded;
    ghost_mnnz_t nnz;
    ghost_midx_t lowerBandwidth;
    ghost_midx_t upperBandwidth;
    ghost_midx_t bandwidth;
    /**
     * @brief Array of length nrows with nzDist[i] = number nonzeros with distance i from diagonal
     */
    ghost_mnnz_t *nzDist;
    ghost_mnnz_t nEnts;

    /**
     * @brief Permute the matrix rows and column indices (if set in mat->traits->flags) with the given permutation.
     *
     * @param mat The matrix.
     * @param perm The permutation vector.
     * @param invPerm The inverse permutation vector.
     */
    ghost_error_t (*permute) (ghost_sparsemat_t *mat, ghost_midx_t *perm, ghost_midx_t *invPerm);
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
    ghost_error_t (*spmv) (ghost_sparsemat_t *mat, ghost_densemat_t *res, ghost_densemat_t *rhs, ghost_spmv_flags_t flags);
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
     *
     * This function is called in ghost_printMatrixInfo() to print format-specific information alongside with
     * general matrix information.
     */
    void       (*printInfo) (ghost_sparsemat_t *mat);
    /**
     * @brief Turns the matrix into a string.
     *
     * @param mat The matrix.
     * @param dense If 0, only the elements stored in the sparse matrix will be included.
     * If 1, the matrix will be interpreted as a dense matrix.
     *
     * @return The stringified matrix.
     */
    const char * (*stringify) (ghost_sparsemat_t *mat, int dense);
    /**
     * @brief Get the length of the given row.
     *
     * @param mat The matrix.
     * @param row The row.
     *
     * @return The length of the row or zero if the row index is out of bounds. 
     */
    ghost_midx_t  (*rowLen) (ghost_sparsemat_t *mat, ghost_midx_t row);
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
     * @param maxrowlen The maximum row length of the matrix.
     * @param base The base of indices (e.g., 0 for C, 1 for Fortran).
     * @param func The function defining the matrix.
     * @param flags Flags to control the behaviour of the function.
     *
     * The function func may be called several times for each row concurrently by multiple threads.
     */
    ghost_error_t (*fromRowFunc)(ghost_sparsemat_t *, ghost_midx_t maxrowlen, int base, ghost_spmFromRowFunc_t func, ghost_spmFromRowFunc_flags_t flags);
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
    ghost_error_t ghost_createMatrix(ghost_sparsemat_t **mat, ghost_context_t *ctx, ghost_sparsemat_traits_t *traits, int nTraits);
    ghost_error_t ghost_printMatrixInfo(ghost_sparsemat_t *matrix);
    ghost_error_t ghost_getMatNnz(ghost_mnnz_t *nnz, ghost_sparsemat_t *mat);
    ghost_error_t ghost_getMatNrows(ghost_midx_t *nrows, ghost_sparsemat_t *mat);

#ifdef __cplusplus
} extern "C"
#endif

#endif
