#ifndef GHOST_MAT_H
#define GHOST_MAT_H

#include "config.h"
#include "types.h"
#include "context.h"
#include "vec.h"

#define GHOST_SPM_FORMAT_CRS 0
#define GHOST_SPM_FORMAT_SELL 1

#define GHOST_SPM_SYMM_GENERAL        (1)
#define GHOST_SPM_SYMM_SYMMETRIC      (2)
#define GHOST_SPM_SYMM_SKEW_SYMMETRIC (4)
#define GHOST_SPM_SYMM_HERMITIAN      (8)

typedef void (*ghost_spmFromRowFunc_t)(ghost_midx_t, ghost_midx_t *, ghost_midx_t *, void *);
typedef struct ghost_mtraits_t ghost_mtraits_t;
typedef struct ghost_mat_t ghost_mat_t;

typedef enum {
    GHOST_SPMFROMROWFUNC_DEFAULT = 0
} ghost_spmFromRowFunc_flags_t;

typedef enum {
    GHOST_SPM_DEFAULT       = 0,
    GHOST_SPM_HOST          = 1,
    GHOST_SPM_DEVICE        = 2,
    GHOST_SPM_PERMUTECOLIDX = 4,
    GHOST_SPM_COLMAJOR      = 8,
    GHOST_SPM_ROWMAJOR      = 16,
    GHOST_SPM_SORTED        = 32,
    GHOST_SPM_ASC_COLIDX    = 64,
    GHOST_SPM_STORE_SPLIT = 128,
    GHOST_SPM_STORE_FULL = 256
} ghost_spm_flags_t;

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

#define GHOST_SPMVM_MODES_FULL     (GHOST_SPMVM_MODE_NOMPI | GHOST_SPMVM_MODE_VECTORMODE)
#define GHOST_SPMVM_MODES_SPLIT    (GHOST_SPMVM_MODE_GOODFAITH | GHOST_SPMVM_MODE_TASKMODE)
#define GHOST_SPMVM_MODES_ALL      (GHOST_SPMVM_MODES_FULL | GHOST_SPMVM_MODES_SPLIT)

struct ghost_mtraits_t
{
    int format;
    ghost_spm_flags_t flags;
    int symmetry;
    void * aux;
    int nAux;
    int datatype;
    void * shift;
    void * scale;
    void * beta; // scale factor for AXPBY
};
extern const ghost_mtraits_t GHOST_MTRAITS_INITIALIZER;

struct ghost_mat_t
{
    ghost_mtraits_t *traits;
    ghost_mat_t *localPart;
    ghost_mat_t *remotePart;
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
    ghost_error_t (*permute) (ghost_mat_t *mat, ghost_midx_t *perm, ghost_midx_t *invPerm);
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
    ghost_error_t (*spmv) (ghost_mat_t *mat, ghost_vec_t *res, ghost_vec_t *rhs, ghost_spmv_flags_t flags);
    /**
     * @brief Destroy the matrix, i.e., free all of its data.
     *
     * @param mat The matrix.
     *
     * Returns if the matrix is NULL.
     */
    void       (*destroy) (ghost_mat_t *mat);
    /**
     * @brief Prints specific information on the matrix.
     *
     * @param mat The matrix.
     *
     * This function is called in ghost_printMatrixInfo() to print format-specific information alongside with
     * general matrix information.
     */
    void       (*printInfo) (ghost_mat_t *mat);
    /**
     * @brief Turns the matrix into a string.
     *
     * @param mat The matrix.
     * @param dense If 0, only the elements stored in the sparse matrix will be included.
     * If 1, the matrix will be interpreted as a dense matrix.
     *
     * @return The stringified matrix.
     */
    const char * (*stringify) (ghost_mat_t *mat, int dense);
    /**
     * @brief Get the length of the given row.
     *
     * @param mat The matrix.
     * @param row The row.
     *
     * @return The length of the row or zero if the row index is out of bounds. 
     */
    ghost_midx_t  (*rowLen) (ghost_mat_t *mat, ghost_midx_t row);
    /**
     * @brief Return the name of the storage format.
     *
     * @param mat The matrix.
     *
     * @return A string containing the storage format name. 
     */
    const char *  (*formatName) (ghost_mat_t *mat);
    /**
     * @brief Create the matrix from a matrix file in GHOST's binary CRS format.
     *
     * @param mat The matrix. 
     * @param path Path to the file.
     */
    ghost_error_t (*fromFile)(ghost_mat_t *mat, char *path);
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
    ghost_error_t (*fromRowFunc)(ghost_mat_t *, ghost_midx_t maxrowlen, int base, ghost_spmFromRowFunc_t func, ghost_spmFromRowFunc_flags_t flags);
    /**
     * @brief Write a matrix to a binary CRS file.
     *
     * @param mat The matrix. 
     * @param path Path of the file.
     */
    ghost_error_t (*toFile)(ghost_mat_t *mat, char *path);
    /**
     * @brief Upload the matrix to the CUDA device.
     *
     * @param mat The matrix.
     */
    ghost_error_t (*upload)(ghost_mat_t * mat);
    /**
     * @brief Get the entire memory footprint of the matrix.
     *
     * @param mat The matrix.
     *
     * @return The memory footprint of the matrix in bytes or zero if the matrix is not valid.
     */
    size_t     (*byteSize)(ghost_mat_t *mat);
    /**
     * @deprecated
     * @brief Create a matrix from a CRS matrix.
     *
     * @param mat The matrix. 
     * @param crsMat The CRS matrix.
     */
    ghost_error_t     (*fromCRS)(ghost_mat_t *mat, ghost_mat_t *crsMat);
    /**
     * @brief Split the matrix into a local and a remote part.
     *
     * @param mat The matrix.
     */
    ghost_error_t       (*split)(ghost_mat_t *mat);
};


#ifdef __cplusplus
extern "C" {
#endif

    ghost_error_t ghost_createMatrix(ghost_context_t *, ghost_mtraits_t *, int, ghost_mat_t **);
    ghost_error_t ghost_printMatrixInfo(ghost_mat_t *matrix);
    ghost_error_t ghost_getMatNnz(ghost_mnnz_t *nnz, ghost_mat_t *mat);
    ghost_error_t ghost_getMatNrows(ghost_midx_t *nrows, ghost_mat_t *mat);

#ifdef __cplusplus
} extern "C"
#endif

#endif
