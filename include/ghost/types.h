#ifndef GHOST_TYPES_H
#define GHOST_TYPES_H

#include "error.h"

#ifdef GHOST_HAVE_MPI
#ifdef __INTEL_COMPILER
#pragma warning (disable : 869)
#pragma warning (disable : 424)
#endif
#include <mpi.h>
typedef MPI_Comm ghost_mpi_comm_t;
#ifdef __INTEL_COMPILER
#pragma warning (enable : 424)
#pragma warning (enable : 869)
#endif
#else
typedef int ghost_mpi_comm_t;
#define MPI_COMM_WORLD 0
//typedef int MPI_Comm;
//#define MPI_COMM_WORLD 0 // TODO unschoen
#endif

#include <inttypes.h>
#include <stdint.h>
#include <sys/types.h>

#if GHOST_HAVE_LONGIDX
typedef int64_t ghost_midx_t; // type for the index of the matrix
typedef int64_t ghost_mnnz_t; // type for the number of nonzeros in the matrix
typedef int64_t ghost_vidx_t; // type for the index of the vector
typedef long long int ghost_blas_idx_t;

#define ghost_mpi_dt_midx MPI_LONG_LONG
#define ghost_mpi_dt_mnnz MPI_LONG_LONG

#define PRmatNNZ PRId64
#define PRmatIDX PRId64
#define PRvecIDX PRId64

#else

typedef int32_t ghost_midx_t; // type for the index of the matrix
typedef int32_t ghost_mnnz_t; // type for the number of nonzeros in the matrix
typedef int32_t ghost_vidx_t; // type for the index of the vector
typedef int ghost_blas_idx_t;


#define ghost_mpi_dt_midx MPI_INT
#define ghost_mpi_dt_mnnz MPI_INT

#define PRmatNNZ PRId32
#define PRmatIDX PRId32
#define PRvecIDX PRId32

#endif


typedef enum {
    GHOST_HYBRIDMODE_INVALID, 
    GHOST_HYBRIDMODE_ONEPERNODE, 
    GHOST_HYBRIDMODE_ONEPERNUMA, 
    GHOST_HYBRIDMODE_ONEPERCORE
} ghost_hybridmode_t;

typedef enum {
    GHOST_TYPE_INVALID, 
    GHOST_TYPE_COMPUTE, 
    GHOST_TYPE_CUDAMGMT
} ghost_type_t;

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
    GHOST_SPMFROMROWFUNC_DEFAULT = 0
} ghost_spmFromRowFunc_flags_t;

typedef struct ghost_vec_t ghost_vec_t;
typedef struct ghost_mat_t ghost_mat_t;
typedef struct ghost_context_t ghost_context_t;
//typedef struct ghost_comm_t ghost_comm_t;
typedef struct ghost_mtraits_t ghost_mtraits_t;
typedef struct ghost_vtraits_t ghost_vtraits_t;
typedef struct ghost_acc_info_t ghost_acc_info_t;
typedef struct ghost_matfile_header_t ghost_matfile_header_t;
typedef struct ghost_mpi_c ghost_mpi_c;
typedef struct ghost_mpi_z ghost_mpi_z;
typedef void (*ghost_spmvsolver_t)(ghost_context_t *, ghost_vec_t*, ghost_mat_t *, ghost_vec_t*, int);
typedef void (*ghost_spmFromRowFunc_t)(ghost_midx_t, ghost_midx_t *, ghost_midx_t *, void *);

/**
 * @brief This struct represents a vector (dense matrix) datatype.  The
 * according functions are accessed via function pointers. The first argument of
 * each member function always has to be a pointer to the vector itself.
 */
struct ghost_vec_t
{
    /**
     * @brief The vector's traits.
     */
    ghost_vtraits_t *traits;
    /**
     * @brief The context in which the vector is living.
     */
    ghost_context_t *context;
    /**
     * @brief The values of the vector.
     */
    char** val;
    /**
     * @brief Indicates whether the vector is a view. A view is a vector whose
     * #val pointer points to some data which is not allocated withing the
     * vector.
     */
//    int isView;

    /**
     * @brief Performs <em>y := a*x + y</em> with scalar a
     *
     * @param y The in-/output vector
     * @param x The input vector
     * @param a Points to the scale factor.
     */
    void          (*axpy) (ghost_vec_t *y, ghost_vec_t *x, void *a);
    /**
     * @brief Performs <em>y := a*x + b*y</em> with scalar a and b
     *
     * @param y The in-/output vector.
     * @param x The input vector
     * @param a Points to the scale factor a.
     * @param b Points to the scale factor b.
     */
    void          (*axpby) (ghost_vec_t *y, ghost_vec_t *x, void *a, void *b);
    /**
     * @brief Clones a given number of columns of a source vector at a given
     * column offset.
     *
     * @param vec The source vector.
     * @param ncols The number of columns to clone.
     * @param coloffset The first column to clone.
     *
     * @return A clone of the source vector.
     */
    ghost_vec_t * (*clone) (ghost_vec_t *vec, ghost_vidx_t ncols, ghost_vidx_t
            coloffset);
    /**
     * @brief Compresses a vector, i.e., make it non-scattered.
     * If the vector is a view, it will no longer be one afterwards.
     *
     * @param vec The vector.
     */
    void          (*compress) (ghost_vec_t *vec);
    /**
     * @brief Collects vec from all MPI ranks and combines them into globalVec.
     * The row permutation (if present) if vec's context is used.
     *
     * @param vec The distributed vector.
     * @param globalVec The global vector.
     */
    void          (*collect) (ghost_vec_t *vec, ghost_vec_t *globalVec);
    /**
     * @brief \deprecated
     */
    void          (*CUdownload) (ghost_vec_t *);
    /**
     * @brief \deprecated
     */
    void          (*CUupload) (ghost_vec_t *);
    /**
     * @brief Destroys a vector, i.e., frees all its data structures.
     *
     * @param vec The vector
     */
    void          (*destroy) (ghost_vec_t *vec);
    /**
     * @brief Distributes a global vector into node-local vetors.
     *
     * @param vec The global vector.
     * @param localVec The local vector.
     */
    void          (*distribute) (ghost_vec_t *vec, ghost_vec_t *localVec);
    /**
     * @brief Computes the dot product of two vectors and stores the result in
     * res.
     *
     * @param a The first vector.
     * @param b The second vector.
     * @param res A pointer to where the result should be stored.
     */
    void          (*dotProduct) (ghost_vec_t *a, ghost_vec_t *b, void *res);
    /**
     * @brief Downloads an entire vector from a compute device. Does nothing if
     * the vector is not present on the device.
     *
     * @param vec The vector.
     */
    void          (*download) (ghost_vec_t *vec);
    /**
     * @brief Downloads only a vector's halo elements from a compute device.
     * Does nothing if the vector is not present on the device.
     *
     * @param vec The vector.
     */
    void          (*downloadHalo) (ghost_vec_t *vec);
    /**
     * @brief Downloads only a vector's local elements (i.e., without halo
     * elements) from a compute device. Does nothing if the vector is not
     * present on the device.
     *
     * @param vec The vector.
     */
    void          (*downloadNonHalo) (ghost_vec_t *vec);
    /**
     * @brief Stores the entry of the vector at a given index (row i, column j)
     * into entry.
     *
     * @param vec The vector.
     * @param ghost_vidx_t i The row.
     * @param ghost_vidx_t j The column.
     * @param entry Where to store the entry.
     */
    void          (*entry) (ghost_vec_t *vec, ghost_vidx_t i, ghost_vidx_t j,
            void *entry);
    /**
     * @brief Initializes a vector from a given function.
     * Malloc's memory for the vector's values if this hasn't happened before.
     *
     * @param vec The vector.
     * @param fp The function pointer. The function takes three arguments: The row index, the column index and a pointer to where to store the value at this position.
     */
    void          (*fromFunc) (ghost_vec_t *vec, void (*fp)(int,int,void *)); // TODO ghost_vidx_t
    /**
     * @brief Initializes a vector from another vector at a given column offset.
     * Malloc's memory for the vector's values if this hasn't happened before.
     *
     * @param vec The vector.
     * @param src The source vector.
     * @param ghost_vidx_t The column offset in the source vector.
     */
    void          (*fromVec) (ghost_vec_t *vec, ghost_vec_t *src, ghost_vidx_t offset);
    /**
     * @brief Initializes a vector from a file.
     * Malloc's memory for the vector's values if this hasn't happened before.
     *
     * @param vec The vector.
     * @param filename Path to the file.
     */
    void          (*fromFile) (ghost_vec_t *vec, char *filename);
    /**
     * @brief Initiliazes a vector from random values.
     *
     * @param vec The vector.
     */
    void          (*fromRand) (ghost_vec_t *vec);
    /**
     * @brief Initializes a vector from a given scalar value.
     *
     * @param vec The vector.
     * @param val A pointer to the value.
     */
    void          (*fromScalar) (ghost_vec_t *vec, void *val);
    /**
     * @brief Normalize a vector, i.e., scale it such that its 2-norm is one.
     *
     * @param vec The vector.
     */
    void          (*normalize) (ghost_vec_t *vec);
    /**
     * @brief Permute a vector with a given permutation.
     *
     * @param vec The vector.
     * @param perm The permutation.
     */
    void          (*permute) (ghost_vec_t *vec, ghost_vidx_t *perm);
    /**
     * @brief Print a vector.
     *
     * @param vec The vector.
     */
    void          (*print) (ghost_vec_t *vec);
    /**
     * @brief Scale a vector with a given scalar.
     *
     * @param vec The vector.
     * @param scale The scale factor.
     */
    void          (*scale) (ghost_vec_t *vec, void *scale);
    /**
     * @brief Swap two vectors.
     *
     * @param vec1 The first vector.
     * @param vec2 The second vector.
     */
    void          (*swap) (ghost_vec_t *vec1, ghost_vec_t *vec2);
    /**
     * @brief Write a vector to a file.
     *
     * @param vec The vector.
     * @param filename The path to the file.
     */
    void          (*toFile) (ghost_vec_t *vec, char *filename);
    /**
     * @brief Uploads an entire vector to a compute device. Does nothing if
     * the vector is not present on the device.
     *
     * @param vec The vector.
     */
    void          (*upload) (ghost_vec_t *vec);
    /**
     * @brief Uploads only a vector's halo elements to a compute device.
     * Does nothing if the vector is not present on the device.
     *
     * @param vec The vector.
     */
    void          (*uploadHalo) (ghost_vec_t *vec);
    /**
     * @brief Uploads only a vector's local elements (i.e., without halo
     * elements) to a compute device. Does nothing if the vector is not
     * present on the device.
     *
     * @param vec The vector.
     */
    void          (*uploadNonHalo) (ghost_vec_t *vec);
    /**
     * @brief View plain data in the vector.
     * That means that the vector has no memory malloc'd but its data pointer only points to the memory provided.
     *
     * @param vec The vector.
     * @param data The plain data.
     * @param ghost_vidx_t nr The number of rows.
     * @param ghost_vidx_t nc The number of columns.
     * @param ghost_vidx_t roffs The row offset.
     * @param ghost_vidx_t coffs The column offset.
     * @param ghost_vidx_t lda The number of rows per column.
     */
    void          (*viewPlain) (ghost_vec_t *vec, void *data, ghost_vidx_t nr, ghost_vidx_t nc, ghost_vidx_t roffs, ghost_vidx_t coffs, ghost_vidx_t lda);

    ghost_vec_t * (*viewScatteredVec) (ghost_vec_t *src, ghost_vidx_t nc, ghost_vidx_t *coffs);


    /**
     * @brief Create a vector as a view of another vector.
     *
     * @param src The source vector.
     * @param nc The nunber of columns to view.
     * @param coffs The column offset.
     *
     * @return The new vector.
     */
    ghost_vec_t * (*viewVec) (ghost_vec_t *src, ghost_vidx_t nc, ghost_vidx_t coffs);
    /**
     * @brief Scale each column of a vector with a given scale factor.
     *
     * @param vec The vector.
     * @param scale The scale factors.
     */
    void          (*vscale) (ghost_vec_t *, void *);
    void          (*vaxpy) (ghost_vec_t *, ghost_vec_t *, void *);
    void          (*vaxpby) (ghost_vec_t *, ghost_vec_t *, void *, void *);
    void          (*zero) (ghost_vec_t *);

#ifdef GHOST_HAVE_CUDA
    char * CU_val;
#endif
};

struct ghost_vtraits_t
{
    ghost_midx_t nrows;
    ghost_midx_t nrowshalo;
    ghost_midx_t nrowspadded;
    ghost_midx_t nvecs;
    int flags;
    int datatype;
    void * aux;
    void * localdot;
};
extern const ghost_vtraits_t GHOST_VTRAITS_INITIALIZER;

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
    ghost_error_t (*CUupload)(ghost_mat_t * mat);
    /**
     * @brief Get the entire memory footprint of the matrix.
     *
     * @param mat The matrix.
     *
     * @return The memory footprint of the matrix in bytes (zero if the matrix is NULL).
     */
    size_t     (*byteSize)(ghost_mat_t *mat);
    /**
     * @deprecated
     * @brief Create a matrix from a CRS matrix.
     *
     * @param mat The matrix. 
     * @param crsMat The CRS matrix.
     */
    void       (*fromCRS)(ghost_mat_t *mat, ghost_mat_t *crsMat);
    /**
     * @brief Split the matrix into a local and a remote part.
     *
     * @param mat The matrix.
     */
    ghost_error_t       (*split)(ghost_mat_t *mat);
};

struct ghost_mtraits_t
{
    int format;
    int flags;
    int symmetry;
    void * aux;
    int nAux;
    int datatype;
    void * shift;
    void * scale;
    void * beta; // scale factor for AXPBY
};
extern const ghost_mtraits_t GHOST_MTRAITS_INITIALIZER;

struct ghost_context_t
{
    ghost_spmvsolver_t *spmvsolvers;

    // if the context is distributed by nnz, the row pointers are being read
    // at context creation in order to create the distribution. once the matrix
    // is being created, the row pointers are distributed
    ghost_midx_t *rpt;

   // ghost_comm_t *communicator;
    ghost_midx_t gnrows;
    ghost_midx_t gncols;
    int flags;
    double weight;

    ghost_midx_t *rowPerm;    // may be NULL
    ghost_midx_t *invRowPerm; // may be NULL

    ghost_mpi_comm_t mpicomm;
    
    /**
     * @brief Number of remote elements with unique colidx
     */
    ghost_midx_t halo_elements; // TODO rename nHaloElements
    /**
     * @brief Number of matrix elements for each rank
     */
    ghost_mnnz_t* lnEnts; // TODO rename nLclEnts
    /**
     * @brief Index of first element into the global matrix for each rank
     */
    ghost_mnnz_t* lfEnt; // TODO rename firstLclEnt
    /**
     * @brief Number of matrix rows for each rank
     */
    ghost_midx_t* lnrows; // TODO rename nLclRows
    /**
     * @brief Index of first matrix row for each rank
     */
    ghost_midx_t* lfRow; // TODO rename firstLclRow
    /**
     * @brief Number of wishes (= unique RHS elements to get) from each rank
     */
    ghost_mnnz_t * wishes; // TODO rename nWishes
    /**
     * @brief Column idx of wishes from each rank
     */
    ghost_midx_t ** wishlist; // TODO rename wishes
    /**
     * @brief Number of dues (= unique RHS elements from myself) to each rank
     */
    ghost_mnnz_t * dues; // TODO rename nDues
    /**
     * @brief Column indices of dues to each rank
     */
    ghost_midx_t ** duelist; // TODO rename dues
    /**
     * @brief First index to get RHS elements coming from each rank
     */
    ghost_midx_t* hput_pos; // TODO rename
};

/*struct ghost_comm_t
{
};*/

struct ghost_acc_info_t
{
    int nDistinctDevices;
    int *nDevices;
    char **names;
};

struct ghost_mpi_c
{
    float x;
    float y;
};

struct ghost_mpi_z
{
    double x;
    double y;
};

struct ghost_matfile_header_t
{
    int32_t endianess;
    int32_t version;
    int32_t base;
    int32_t symmetry;
    int32_t datatype;
    int64_t nrows;
    int64_t ncols;
    int64_t nnz;
};

#endif
