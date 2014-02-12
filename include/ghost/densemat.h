/**
 * @file densemat.h
 * @brief Types and functions related to dense matrices/vectors.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_DENSEMAT_H
#define GHOST_DENSEMAT_H

#include "config.h"
#include "types.h"
#include "context.h"

typedef struct ghost_densemat_traits_t ghost_densemat_traits_t;
typedef struct ghost_densemat_t ghost_densemat_t;

typedef enum {
    GHOST_DENSEMAT_DEFAULT   = 0,
    GHOST_DENSEMAT_RHS       = 1,
    GHOST_DENSEMAT_LHS       = 2,
    GHOST_DENSEMAT_HOST      = 4,
    GHOST_DENSEMAT_DEVICE    = 8,
    GHOST_DENSEMAT_GLOBAL    = 16,
    GHOST_DENSEMAT_DUMMY     = 32,
    GHOST_DENSEMAT_VIEW      = 64,
    GHOST_DENSEMAT_SCATTERED = 128
} ghost_densemat_flags_t;

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
    ghost_densemat_traits_t *traits;
    /**
     * @brief The context in which the vector/matrix is living.
     */
    ghost_context_t *context;
    /**
     * @brief The values of the vector/matrix.
     */
    char** val;

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
    ghost_error_t          (*axpy) (ghost_densemat_t *y, ghost_densemat_t *x, void *a);
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
    ghost_error_t          (*axpby) (ghost_densemat_t *y, ghost_densemat_t *x, void *a, void *b);
    /**
     * @brief Clones a given number of columns of a source vector/matrix at a given
     * column offset.
     *
     * @param vec The source vector/matrix.
     * @param ncols The number of columns to clone.
     * @param coloffset The first column to clone.
     *
     * @return A clone of the source vector/matrix.
     */
    ghost_densemat_t * (*clone) (ghost_densemat_t *vec, ghost_vidx_t ncols, ghost_vidx_t
            coloffset);
    /**
     * @brief Compresses a vector/matrix, i.e., make it non-scattered.
     * If the vector/matrix is a view, it will no longer be one afterwards.
     *
     * @param vec The vector/matrix.
     */
    ghost_error_t          (*compress) (ghost_densemat_t *vec);
    /**
     * @brief Collects vec from all MPI ranks and combines them into globalVec.
     * The row permutation (if present) if vec's context is used.
     *
     * @param vec The distributed vector/matrix.
     * @param globalVec The global vector/matrix.
     */
    ghost_error_t (*collect) (ghost_densemat_t *vec, ghost_densemat_t *globalVec);
    /**
     * @brief Destroys a vector/matrix, i.e., frees all its data structures.
     *
     * @param vec The vector/matrix
     */
    void          (*destroy) (ghost_densemat_t *vec);
    /**
     * @brief Distributes a global vector/matrix into node-local vetors.
     *
     * @param vec The global vector/matrix.
     * @param localVec The local vector/matrix.
     */
    ghost_error_t (*distribute) (ghost_densemat_t *vec, ghost_densemat_t *localVec);
    /**
     * @ingroup locops
     * 
     * @brief Compute the local dot product of two vectors/matrices.
     *
     * @param a The first vector/matrix.
     * @param b The second vector/matrix.
     * @param res Where to store the result.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     *
     * For the global operation see ghost_dotProduct().
     *
     * @see ghost_dotProduct()
     */
    ghost_error_t          (*dot) (ghost_densemat_t *a, ghost_densemat_t *b, void *res);
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
    ghost_error_t          (*download) (ghost_densemat_t *vec);
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
    ghost_error_t          (*downloadHalo) (ghost_densemat_t *vec);
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
    ghost_error_t          (*downloadNonHalo) (ghost_densemat_t *vec);
    /**
     * @brief Stores the entry of the vector/matrix at a given index (row i, column j)
     * into entry.
     *
     * @param vec The vector/matrix.
     * @param ghost_vidx_t i The row.
     * @param ghost_vidx_t j The column.
     * @param entry Where to store the entry.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t          (*entry) (ghost_densemat_t *vec, ghost_vidx_t i, ghost_vidx_t j,
            void *entry);
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
    ghost_error_t (*fromFunc) (ghost_densemat_t *vec, void (*fp)(int,int,void *)); // TODO ghost_vidx_t
    /**
     * @ingroup denseinit
     *
     * @brief Initializes a vector/matrix from another vector/matrix at a given column offset.
     * Malloc's memory for the vector/matrix's values if this hasn't happened before.
     *
     * @param vec The vector/matrix.
     * @param src The source vector/matrix.
     * @param ghost_vidx_t The column offset in the source vector/matrix.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t          (*fromVec) (ghost_densemat_t *vec, ghost_densemat_t *src, ghost_vidx_t offset);
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
    ghost_error_t (*fromFile) (ghost_densemat_t *vec, char *filename);
    /**
     * @ingroup denseinit
     *
     * @brief Initiliazes a vector/matrix from random values.
     *
     * @param vec The vector/matrix.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t          (*fromRand) (ghost_densemat_t *vec);
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
    ghost_error_t          (*fromScalar) (ghost_densemat_t *vec, void *val);
    /**
     * @brief Normalize a vector/matrix, i.e., scale it such that its 2-norm is one.
     *
     * @param vec The vector/matrix.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t          (*normalize) (ghost_densemat_t *vec);
    /**
     * @brief Permute a vector/matrix with a given permutation.
     *
     * @param vec The vector/matrix.
     * @param perm The permutation.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t  (*permute) (ghost_densemat_t *vec, ghost_vidx_t *perm);
    /**
     * @brief Print a vector/matrix.
     *
     * @param vec The vector/matrix.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t (*print) (ghost_densemat_t *vec);
    /**
     * @brief Scale a vector/matrix with a given scalar.
     *
     * @param vec The vector/matrix.
     * @param scale The scale factor.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t          (*scale) (ghost_densemat_t *vec, void *scale);
    /**
     * @brief Write a vector/matrix to a file.
     *
     * @param vec The vector/matrix.
     * @param filename The path to the file.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t          (*toFile) (ghost_densemat_t *vec, char *filename);
    /**
     * @ingroup gputransfer
     * 
     * @brief Uploads an entire vector/matrix to a compute device. Does nothing if
     * the vector/matrix is not present on the device.
     *
     * @param vec The vector/matrix.
     */
    ghost_error_t          (*upload) (ghost_densemat_t *vec);
    /**
     * @ingroup gputransfer
     * 
     * @brief Uploads only a vector/matrix's halo elements to a compute device.
     * Does nothing if the vector/matrix is not present on the device.
     *
     * @param vec The vector/matrix.
     */
    ghost_error_t (*uploadHalo) (ghost_densemat_t *vec);
    /**
     * @ingroup gputransfer
     * 
     * @brief Uploads only a vector/matrix's local elements (i.e., without halo
     * elements) to a compute device. Does nothing if the vector/matrix is not
     * present on the device.
     *
     * @param vec The vector/matrix.
     */
    ghost_error_t          (*uploadNonHalo) (ghost_densemat_t *vec);
    /**
     * @brief View plain data in the vector/matrix.
     * That means that the vector/matrix has no memory malloc'd but its data pointer only points to the memory provided.
     *
     * @param vec The vector/matrix.
     * @param data The plain data.
     * @param ghost_vidx_t nr The number of rows.
     * @param ghost_vidx_t nc The number of columns.
     * @param ghost_vidx_t roffs The row offset.
     * @param ghost_vidx_t coffs The column offset.
     * @param ghost_vidx_t lda The number of rows per column.
     */
    ghost_error_t          (*viewPlain) (ghost_densemat_t *vec, void *data, ghost_vidx_t nr, ghost_vidx_t nc, ghost_vidx_t roffs, ghost_vidx_t coffs, ghost_vidx_t lda);

    ghost_densemat_t * (*viewScatteredVec) (ghost_densemat_t *src, ghost_vidx_t nc, ghost_vidx_t *coffs);


    /**
     * @brief Create a vector/matrix as a view of another vector/matrix.
     *
     * @param src The source vector/matrix.
     * @param nc The nunber of columns to view.
     * @param coffs The column offset.
     *
     * @return The new vector/matrix.
     */
    ghost_densemat_t * (*viewVec) (ghost_densemat_t *src, ghost_vidx_t nc, ghost_vidx_t coffs);
    /**
     * @brief Scale each column of a vector/matrix with a given scale factor.
     *
     * @param vec The vector/matrix.
     * @param scale The scale factors.
     */
    ghost_error_t          (*vscale) (ghost_densemat_t *, void *);
    ghost_error_t          (*vaxpy) (ghost_densemat_t *, ghost_densemat_t *, void *);
    ghost_error_t          (*vaxpby) (ghost_densemat_t *, ghost_densemat_t *, void *, void *);
    ghost_error_t          (*zero) (ghost_densemat_t *);
};

struct ghost_densemat_traits_t
{
    ghost_midx_t nrows;
    ghost_midx_t nrowshalo;
    ghost_midx_t nrowspadded;
    ghost_midx_t ncols;
    ghost_densemat_flags_t flags;
    int datatype;
    size_t elSize;
    void * localdot;
};
/**
 * @brief Defined in densemat.c
 */
extern const ghost_densemat_traits_t GHOST_VTRAITS_INITIALIZER;

#define VECVAL(vec,val,__x,__y) &(val[__x][(__y)*vec->traits->elSize])
#define CUVECVAL(vec,val,__x,__y) &(val[((__x)*vec->traits->nrowspadded+(__y))*vec->traits->elSize])

#ifdef __cplusplus
template <typename v_t> ghost_error_t ghost_normalizeVector_tmpl(ghost_densemat_t *vec);
template <typename v_t> ghost_error_t ghost_vec_dotprod_tmpl(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *res);
template <typename v_t> ghost_error_t ghost_vec_vaxpy_tmpl(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *);
template <typename v_t> ghost_error_t ghost_vec_vaxpby_tmpl(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *, void *);
template <typename v_t> ghost_error_t ghost_vec_vscale_tmpl(ghost_densemat_t *vec, void *vscale);
template <typename v_t> ghost_error_t ghost_vec_fromRand_tmpl(ghost_densemat_t *vec);
template <typename v_t> ghost_error_t ghost_vec_print_tmpl(ghost_densemat_t *vec);

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
    ghost_error_t ghost_createVector(ghost_densemat_t **vec, ghost_context_t *ctx, ghost_densemat_traits_t *traits);
    ghost_error_t ghost_cloneVtraits(ghost_densemat_traits_t *t1, ghost_densemat_traits_t **t2);

    ghost_error_t ghost_vec_malloc(ghost_densemat_t *vec);
    ghost_error_t d_ghost_printVector(ghost_densemat_t *vec); 
    ghost_error_t s_ghost_printVector(ghost_densemat_t *vec); 
    ghost_error_t z_ghost_printVector(ghost_densemat_t *vec);
    ghost_error_t c_ghost_printVector(ghost_densemat_t *vec);
    ghost_error_t d_ghost_normalizeVector(ghost_densemat_t *vec); 
    ghost_error_t s_ghost_normalizeVector(ghost_densemat_t *vec); 
    ghost_error_t z_ghost_normalizeVector(ghost_densemat_t *vec);
    ghost_error_t c_ghost_normalizeVector(ghost_densemat_t *vec);
    ghost_error_t d_ghost_vec_dotprod(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *res); 
    ghost_error_t s_ghost_vec_dotprod(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *res); 
    ghost_error_t z_ghost_vec_dotprod(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *res);
    ghost_error_t c_ghost_vec_dotprod(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *res);
    ghost_error_t d_ghost_vec_vscale(ghost_densemat_t *vec1, void *vscale); 
    ghost_error_t s_ghost_vec_vscale(ghost_densemat_t *vec1, void *vscale); 
    ghost_error_t z_ghost_vec_vscale(ghost_densemat_t *vec1, void *vscale);
    ghost_error_t c_ghost_vec_vscale(ghost_densemat_t *vec1, void *vscale);
    ghost_error_t d_ghost_vec_vaxpy(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *); 
    ghost_error_t s_ghost_vec_vaxpy(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *); 
    ghost_error_t z_ghost_vec_vaxpy(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *);
    ghost_error_t c_ghost_vec_vaxpy(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *);
    ghost_error_t d_ghost_vec_vaxpby(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *, void *); 
    ghost_error_t s_ghost_vec_vaxpby(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *, void *); 
    ghost_error_t z_ghost_vec_vaxpby(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *, void *);
    ghost_error_t c_ghost_vec_vaxpby(ghost_densemat_t *vec1, ghost_densemat_t *vec2, void *, void *);
    ghost_error_t d_ghost_vec_fromRand(ghost_densemat_t *vec); 
    ghost_error_t s_ghost_vec_fromRand(ghost_densemat_t *vec); 
    ghost_error_t z_ghost_vec_fromRand(ghost_densemat_t *vec); 
    ghost_error_t c_ghost_vec_fromRand(ghost_densemat_t *vec); 
#ifdef __cplusplus
}
#endif

#if GHOST_HAVE_CUDA
#include "cu_vec.h"
#endif

#endif
