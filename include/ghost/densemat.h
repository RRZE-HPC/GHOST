/**
 * @file densemat.h
 * @brief Types and functions related to dense matrices/vectors.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_DENSEMAT_H
#define GHOST_DENSEMAT_H

#include <hwloc/bitmap.h>

#include "config.h"
#include "types.h"
#include "context.h"
#include "perm.h"

typedef enum {
    GHOST_DENSEMAT_DEFAULT   = 0,
    GHOST_DENSEMAT_NO_HALO   = 1,
    GHOST_DENSEMAT_HOST      = 4,
    GHOST_DENSEMAT_DEVICE    = 8,
    GHOST_DENSEMAT_GLOBAL    = 16,
    GHOST_DENSEMAT_DUMMY     = 32,
    GHOST_DENSEMAT_VIEW      = 64,
    GHOST_DENSEMAT_SCATTERED = 128
} 
ghost_densemat_flags_t;

/**
 * @brief Densemat storage orders
 */
typedef enum 
{
    /**
     * @brief Row-major storage (as in C).
     */
    GHOST_DENSEMAT_ROWMAJOR,
    /**
     * @brief Column-major storage (as in Fortran).
     */
    GHOST_DENSEMAT_COLMAJOR,
    GHOST_DENSEMAT_STORAGE_ANY
}
ghost_densemat_storage_t;


/**
 * @brief Traits of the densemat.
 */
typedef struct
{
    /**
     * @brief The number of rows.
     */
    ghost_idx_t nrows;
    /**
     * @brief The number of rows of the densemat which is viewed by this densemat.
     */
    ghost_idx_t nrowsorig;
    /**
     * @brief The number of rows including halo elements.
     */
    ghost_idx_t nrowshalo;
    /**
     * @brief The padded number of rows (may differ from nrows for col-major densemats).
     */
    ghost_idx_t nrowspadded;
    /**
     * @brief The number of columns.
     */
    ghost_idx_t ncols;
    /**
     * @brief The number of columns of the densemat which is viewed by this densemat.
     */
    ghost_idx_t ncolsorig;
    /**
     * @brief The padded number of columns (may differ from ncols for row-major densemats).
     */
    ghost_idx_t ncolspadded;
    /**
     * @brief Property flags.
     */
    ghost_densemat_flags_t flags;
    /**
     * @brief The storage order.
     */
    ghost_densemat_storage_t storage;
    /**
     * @brief The data type.
     */
    ghost_datatype_t datatype;
}
ghost_densemat_traits_t;

typedef struct ghost_densemat_t ghost_densemat_t;


/**
 * @ingroup types
 *
 * @brief A dense vector/matrix.  
 * 
 * The according functions act locally and are accessed via function pointers. The first argument of
 * each member function always has to be a pointer to the vector/matrix itself.
 */
struct ghost_densemat_t
{
    /**
     * @brief The vector/matrix's traits.
     */
    ghost_densemat_traits_t traits;
    /**
     * @brief The context in which the vector/matrix is living.
     */
    ghost_context_t *context;
    /**
     * @brief The values of the vector/matrix.
     */
    char** val;
    
    /**
     * @brief Size (in bytes) of one matrix element.
     */
    size_t elSize;
    
    /**
     * @brief The leading dimensions of the densemat.
     *
     * Points to ncolspadded if the densemat has row-major storage and nrowspadded if it has col-major storage.
     */
    ghost_idx_t *stride;

    /**
     * @brief Mask out elements in the leading dimension
     */
    hwloc_bitmap_t ldmask;
    /**
     * @brief Mask out elements in the non-leading dimension (only for CUDA)
     */
    hwloc_bitmap_t trmask;

    /**
     * @brief The base pointer of the data which viewed or NULL if not a view.
     */
    struct ghost_densemat_t *viewing;
    ghost_idx_t viewing_col;
    ghost_idx_t viewing_row;
#ifdef GHOST_HAVE_CUDA
    /**
     * @brief The values of the vector/matrix on the CUDA device.
     */
    void * cu_val;
#endif

    /** 
     * @ingroup globops
     *
     * @brief Performs <em>y := a*x + y</em> with scalar a
     *
     * @param y The in-/output vector/matrix
     * @param x The input vector/matrix
     * @param a Points to the scale factor.
     */
    ghost_error_t      (*axpy) (ghost_densemat_t *y, ghost_densemat_t *x, void *a);
    /**
     * @ingroup globops
     *
     * @brief Performs <em>y := a*x + b*y</em> with scalar a and b
     *
     * @param y The in-/output vector/matrix.
     * @param x The input vector/matrix
     * @param a Points to the scale factor a.
     * @param b Points to the scale factor b.
     */
    ghost_error_t      (*axpby) (ghost_densemat_t *y, ghost_densemat_t *x, void *a, void *b);
    /**
     * @brief Clones a given number of columns of a source vector/matrix at a given
     * column and row offset.
     *
     * @param vec The source vector/matrix.
     * @param dst Where to store the new vector.
     * @param nr The number of rows to clone.
     * @param roffs The first row to clone.
     * @param nc The number of columns to clone.
     * @param coffs The first column to clone.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t (*clone) (ghost_densemat_t *vec, ghost_densemat_t **dst, ghost_idx_t nr, ghost_idx_t roffs, ghost_idx_t nc, ghost_idx_t coffs);
    /**
     * @brief Compresses a vector/matrix, i.e., make it non-scattered.
     * If the vector/matrix is a view, it will no longer be one afterwards.
     *
     * @param vec The vector/matrix.
     */
    ghost_error_t       (*compress) (ghost_densemat_t *vec);
    /**
     * @brief Collects vec from all MPI ranks and combines them into globalVec.
     * The row permutation (if present) if vec's context is used.
     *
     * @param vec The distributed vector/matrix.
     * @param globalVec The global vector/matrix.
     */
    ghost_error_t       (*collect) (ghost_densemat_t *vec, ghost_densemat_t *globalVec);
    /**
     * @brief Destroys a vector/matrix, i.e., frees all its data structures.
     *
     * @param vec The vector/matrix
     */
    void                (*destroy) (ghost_densemat_t *vec);
 
    /**
     * @brief Distributes a global vector/matrix into node-local vetors.
     *
     * @param vec The global vector/matrix.
     * @param localVec The local vector/matrix.
     */
    ghost_error_t       (*distribute) (ghost_densemat_t *vec, ghost_densemat_t *localVec);
    /**
     * @ingroup locops
     * 
     * @brief Compute the local dot product of two vectors/matrices.
     *
     * @param a The first vector/matrix.
     * @param res Where to store the result.
     * @param b The second vector/matrix.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     *
     * For the global operation see ghost_dot().
     *
     * @see ghost_dot()
     */
    ghost_error_t       (*dot) (ghost_densemat_t *a, void *res, ghost_densemat_t *b);
    /**
     * @ingroup gputransfer
     * 
     * @brief Downloads an entire vector/matrix from a compute device. Does nothing if
     * the vector/matrix is not present on the device.
     *
     * @param vec The vector/matrix.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t       (*download) (ghost_densemat_t *vec);
    /**
     * @ingroup gputransfer
     * 
     * @brief Downloads only a vector/matrix's halo elements from a compute device.
     * Does nothing if the vector/matrix is not present on the device.
     *
     * @param vec The vector/matrix.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t       (*downloadHalo) (ghost_densemat_t *vec);
    /**
     * @ingroup gputransfer
     * 
     * @brief Downloads only a vector/matrix's local elements (i.e., without halo
     * elements) from a compute device. Does nothing if the vector/matrix is not
     * present on the device.
     *
     * @param vec The vector/matrix.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t       (*downloadNonHalo) (ghost_densemat_t *vec);
    /**
     * @brief Stores the entry of the vector/matrix at a given index (row i, column j)
     * into entry.
     *
     * @param vec The vector/matrix.
     * @param entry Where to store the entry.
     * @param ghost_idx_t i The row.
     * @param ghost_idx_t j The column.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t       (*entry) (ghost_densemat_t *vec, void *entry, ghost_idx_t i, ghost_idx_t j);
    /**
     * @ingroup denseinit
     *
     * @brief Initializes a vector/matrix from a given function.
     * Malloc's memory for the vector/matrix's values if this hasn't happened before.
     *
     * @param vec The vector/matrix.
     * @param fp The function pointer. The function takes three arguments: The row index, the column index and a pointer to where to store the value at this position.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t       (*fromFunc) (ghost_densemat_t *vec, void (*fp)(ghost_idx_t, ghost_idx_t, void *));
    /**
     * @ingroup denseinit
     *
     * @brief Initializes a vector/matrix from another vector/matrix at a given column and row offset.
     * Malloc's memory for the vector/matrix's values if this hasn't happened before.
     *
     * @param vec The vector/matrix.
     * @param src The source.
     * @param roffs The first row to clone.
     * @param coffs The first column to clone.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t (*fromVec) (ghost_densemat_t *vec, ghost_densemat_t *src, ghost_idx_t roffs, ghost_idx_t coffs);
    /**
     * @ingroup denseinit
     *
     * @brief Initializes a vector/matrix from a file.
     * Malloc's memory for the vector/matrix's values if this hasn't happened before.
     *
     * @param vec The vector/matrix.
     * @param filename Path to the file.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t       (*fromFile) (ghost_densemat_t *vec, char *filename);
    /**
     * @ingroup denseinit
     *
     * @brief Initiliazes a vector/matrix from random values.
     *
     * @param vec The vector/matrix.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t       (*fromRand) (ghost_densemat_t *vec);
    /**
     * @ingroup denseinit
     *
     * @brief Initializes a vector/matrix from a given scalar value.
     *
     * @param vec The vector/matrix.
     * @param val A pointer to the value.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t       (*fromScalar) (ghost_densemat_t *vec, void *val);
    /**
     * @brief Change the memory layout between row-/col-major.
     *
     * @param vec The vector/matrix.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t       (*memtranspose) (ghost_densemat_t *vec);
    /**
     * @brief Normalize a vector/matrix, i.e., scale it such that its 2-norm is one.
     *
     * @param vec The vector/matrix.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t       (*normalize) (ghost_densemat_t *vec);
    /**
     * @brief Permute a vector/matrix with a given permutation.
     *
     * @param vec The vector/matrix.
     * @param perm The permutation.
     * @param dir The permutation direction.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t       (*permute) (ghost_densemat_t *vec, ghost_permutation_t *perm, ghost_permutation_direction_t dir);
    /**
     * @brief Create a string from the vector.
     *
     * @param vec The vector/matrix.
     * @param str Where to store the string.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t       (*string) (ghost_densemat_t *vec, char **str);
    /**
     * @brief Scale a vector/matrix with a given scalar.
     *
     * @param vec The vector/matrix.
     * @param scale The scale factor.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t       (*scale) (ghost_densemat_t *vec, void *scale);
    /**
     * @brief Write a vector/matrix to a file.
     *
     * @param vec The vector/matrix.
     * @param filename The path to the file.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t       (*toFile) (ghost_densemat_t *vec, char *filename);
    /**
     * @ingroup gputransfer
     * 
     * @brief Uploads an entire vector/matrix to a compute device. Does nothing if
     * the vector/matrix is not present on the device.
     *
     * @param vec The vector/matrix.
     */
    ghost_error_t       (*upload) (ghost_densemat_t *vec);
    /**
     * @ingroup gputransfer
     * 
     * @brief Uploads only a vector/matrix's halo elements to a compute device.
     * Does nothing if the vector/matrix is not present on the device.
     *
     * @param vec The vector/matrix.
     */
    ghost_error_t       (*uploadHalo) (ghost_densemat_t *vec);
    /**
     * @ingroup gputransfer
     * 
     * @brief Uploads only a vector/matrix's local elements (i.e., without halo
     * elements) to a compute device. Does nothing if the vector/matrix is not
     * present on the device.
     *
     * @param vec The vector/matrix.
     */
    ghost_error_t       (*uploadNonHalo) (ghost_densemat_t *vec);
    /**
     * @brief View plain data in the vector/matrix.
     * That means that the vector/matrix has no memory malloc'd but its data pointer only points to the memory provided.
     *
     * @param vec The vector/matrix.
     * @param data The plain data.
     * @param ghost_idx_t nr The number of rows.
     * @param ghost_idx_t nc The number of columns.
     * @param ghost_idx_t roffs The row offset.
     * @param ghost_idx_t coffs The column offset.
     * @param ghost_idx_t lda The number of rows per column.
     */
    ghost_error_t       (*viewPlain) (ghost_densemat_t *vec, void *data, ghost_idx_t nr, ghost_idx_t nc, ghost_idx_t roffs, ghost_idx_t coffs, ghost_idx_t lda);

    ghost_error_t  (*viewScatteredVec) (ghost_densemat_t *src, ghost_densemat_t **dst, ghost_idx_t nr, ghost_idx_t *roffs,  ghost_idx_t nc, ghost_idx_t *coffs);
    
    ghost_error_t  (*viewScatteredCols) (ghost_densemat_t *src, ghost_densemat_t **dst, ghost_idx_t nc, ghost_idx_t *coffs);
    ghost_error_t  (*viewCols) (ghost_densemat_t *src, ghost_densemat_t **dst, ghost_idx_t nc, ghost_idx_t coffs);


    /**
     * @brief Create a vector/matrix as a view of another vector/matrix.
     *
     * @param src The source vector/matrix.
     * @param nr The nunber of rows to view.
     * @param roffs The row offset.
     * @param dst Where to store the new vector.
     * @param nc The nunber of columns to view.
     * @param coffs The column offset.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t  (*viewVec) (ghost_densemat_t *src, ghost_densemat_t **dst, ghost_idx_t nr, ghost_idx_t roffs, ghost_idx_t nc, ghost_idx_t coffs);
    /**
     * @brief Scale each column of a vector/matrix with a given scale factor.
     *
     * @param vec The vector/matrix.
     * @param scale The scale factors.
     */
    ghost_error_t       (*vscale) (ghost_densemat_t *, void *);
    ghost_error_t       (*vaxpy) (ghost_densemat_t *, ghost_densemat_t *, void *);
    ghost_error_t       (*vaxpby) (ghost_densemat_t *, ghost_densemat_t *, void *, void *);
};

/**
 * @brief Initializer for densemat traits.
 */
#define GHOST_DENSEMAT_TRAITS_INITIALIZER {\
    .nrows = 0,\
    .nrowsorig = 0,\
    .nrowshalo = 0,\
    .nrowspadded = 0,\
    .ncols = 1,\
    .ncolsorig = 0,\
    .ncolspadded = 0,\
    .flags = GHOST_DENSEMAT_DEFAULT,\
    .storage = GHOST_DENSEMAT_COLMAJOR,\
    .datatype = (ghost_datatype_t)(GHOST_DT_DOUBLE|GHOST_DT_REAL),\
};

#define GHOST_BITMAP_COMPACT(bm) ((hwloc_bitmap_last(bm)-hwloc_bitmap_first(bm)) == hwloc_bitmap_weight(bm))

#ifdef __cplusplus

extern "C" {
#endif

    /**
     * @ingroup types
     *
     * @brief Create a dense matrix/vector. 
     *
     * @param vec Where to store the matrix.
     * @param ctx The context the matrix lives in or NULL.
     * @param traits The matrix traits.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_densemat_create(ghost_densemat_t **vec, ghost_context_t *ctx, ghost_densemat_traits_t traits);
    
    /**
     * @brief Get the location of the first viewed data element.
     *
     * @param vec The densemat.
     * @param ptr Where to store the pointer.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_densemat_valptr(ghost_densemat_t *vec, void **ptr);
    /**
     * @brief Create an array of chars (0 or 1) of the densemat mask.
     *
     * @param mask The ldmask.
     * @param len Length of the ldmask.
     * @param charfield Location of the char array.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_densemat_mask2charfield(hwloc_bitmap_t mask, ghost_idx_t len, char *charfield);

    bool array_strictly_ascending (ghost_idx_t *coffs, ghost_idx_t nc);

#ifdef __cplusplus
}
#endif

#ifdef GHOST_HAVE_CUDA
#include "cu_densemat.h"
#endif

#endif
