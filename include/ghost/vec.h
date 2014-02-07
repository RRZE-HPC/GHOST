#ifndef GHOST_VEC_H
#define GHOST_VEC_H

#include "config.h"
#include "types.h"
#include "context.h"


typedef struct ghost_vtraits_t ghost_vtraits_t;
typedef struct ghost_vec_t ghost_vec_t;
typedef enum {
    GHOST_VEC_DEFAULT   = 0,
    GHOST_VEC_RHS       = 1,
    GHOST_VEC_LHS       = 2,
    GHOST_VEC_HOST      = 4,
    GHOST_VEC_DEVICE    = 8,
    GHOST_VEC_GLOBAL    = 16,
    GHOST_VEC_DUMMY     = 32,
    GHOST_VEC_VIEW      = 64,
    GHOST_VEC_SCATTERED = 128
} ghost_vec_flags_t;

/**
 * @brief This struct represents a vector (dense matrix) datatype.  
 * 
 * The according functions are accessed via function pointers. The first argument of
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
     * @brief Performs <em>y := a*x + y</em> with scalar a
     *
     * @param y The in-/output vector
     * @param x The input vector
     * @param a Points to the scale factor.
     */
    ghost_error_t          (*axpy) (ghost_vec_t *y, ghost_vec_t *x, void *a);
    /**
     * @brief Performs <em>y := a*x + b*y</em> with scalar a and b
     *
     * @param y The in-/output vector.
     * @param x The input vector
     * @param a Points to the scale factor a.
     * @param b Points to the scale factor b.
     */
    ghost_error_t          (*axpby) (ghost_vec_t *y, ghost_vec_t *x, void *a, void *b);
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
    ghost_error_t          (*compress) (ghost_vec_t *vec);
    /**
     * @brief Collects vec from all MPI ranks and combines them into globalVec.
     * The row permutation (if present) if vec's context is used.
     *
     * @param vec The distributed vector.
     * @param globalVec The global vector.
     */
    ghost_error_t (*collect) (ghost_vec_t *vec, ghost_vec_t *globalVec);
    /**
     * @brief \deprecated
     */
    ghost_error_t          (*CUdownload) (ghost_vec_t *);
    /**
     * @brief \deprecated
     */
    ghost_error_t          (*CUupload) (ghost_vec_t *);
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
    ghost_error_t (*distribute) (ghost_vec_t *vec, ghost_vec_t *localVec);
    /**
     * @brief Computes the dot product of two vectors and stores the result in
     * res.
     *
     * @param a The first vector.
     * @param b The second vector.
     * @param res A pointer to where the result should be stored.
     */
    ghost_error_t          (*dotProduct) (ghost_vec_t *a, ghost_vec_t *b, void *res);
    /**
     * @brief Downloads an entire vector from a compute device. Does nothing if
     * the vector is not present on the device.
     *
     * @param vec The vector.
     */
    ghost_error_t          (*download) (ghost_vec_t *vec);
    /**
     * @brief Downloads only a vector's halo elements from a compute device.
     * Does nothing if the vector is not present on the device.
     *
     * @param vec The vector.
     */
    ghost_error_t          (*downloadHalo) (ghost_vec_t *vec);
    /**
     * @brief Downloads only a vector's local elements (i.e., without halo
     * elements) from a compute device. Does nothing if the vector is not
     * present on the device.
     *
     * @param vec The vector.
     */
    ghost_error_t          (*downloadNonHalo) (ghost_vec_t *vec);
    /**
     * @brief Stores the entry of the vector at a given index (row i, column j)
     * into entry.
     *
     * @param vec The vector.
     * @param ghost_vidx_t i The row.
     * @param ghost_vidx_t j The column.
     * @param entry Where to store the entry.
     */
    ghost_error_t          (*entry) (ghost_vec_t *vec, ghost_vidx_t i, ghost_vidx_t j,
            void *entry);
    /**
     * @brief Initializes a vector from a given function.
     * Malloc's memory for the vector's values if this hasn't happened before.
     *
     * @param vec The vector.
     * @param fp The function pointer. The function takes three arguments: The row index, the column index and a pointer to where to store the value at this position.
     */
    ghost_error_t (*fromFunc) (ghost_vec_t *vec, void (*fp)(int,int,void *)); // TODO ghost_vidx_t
    /**
     * @brief Initializes a vector from another vector at a given column offset.
     * Malloc's memory for the vector's values if this hasn't happened before.
     *
     * @param vec The vector.
     * @param src The source vector.
     * @param ghost_vidx_t The column offset in the source vector.
     */
    ghost_error_t          (*fromVec) (ghost_vec_t *vec, ghost_vec_t *src, ghost_vidx_t offset);
    /**
     * @brief Initializes a vector from a file.
     * Malloc's memory for the vector's values if this hasn't happened before.
     *
     * @param vec The vector.
     * @param filename Path to the file.
     */
    ghost_error_t (*fromFile) (ghost_vec_t *vec, char *filename);
    /**
     * @brief Initiliazes a vector from random values.
     *
     * @param vec The vector.
     */
    ghost_error_t          (*fromRand) (ghost_vec_t *vec);
    /**
     * @brief Initializes a vector from a given scalar value.
     *
     * @param vec The vector.
     * @param val A pointer to the value.
     */
    ghost_error_t          (*fromScalar) (ghost_vec_t *vec, void *val);
    /**
     * @brief Normalize a vector, i.e., scale it such that its 2-norm is one.
     *
     * @param vec The vector.
     */
    ghost_error_t          (*normalize) (ghost_vec_t *vec);
    /**
     * @brief Permute a vector with a given permutation.
     *
     * @param vec The vector.
     * @param perm The permutation.
     */
    ghost_error_t  (*permute) (ghost_vec_t *vec, ghost_vidx_t *perm);
    /**
     * @brief Print a vector.
     *
     * @param vec The vector.
     */
    ghost_error_t (*print) (ghost_vec_t *vec);
    /**
     * @brief Scale a vector with a given scalar.
     *
     * @param vec The vector.
     * @param scale The scale factor.
     */
    ghost_error_t          (*scale) (ghost_vec_t *vec, void *scale);
    /**
     * @brief Swap two vectors.
     *
     * @param vec1 The first vector.
     * @param vec2 The second vector.
     */
    ghost_error_t          (*swap) (ghost_vec_t *vec1, ghost_vec_t *vec2);
    /**
     * @brief Write a vector to a file.
     *
     * @param vec The vector.
     * @param filename The path to the file.
     */
    ghost_error_t          (*toFile) (ghost_vec_t *vec, char *filename);
    /**
     * @brief Uploads an entire vector to a compute device. Does nothing if
     * the vector is not present on the device.
     *
     * @param vec The vector.
     */
    ghost_error_t          (*upload) (ghost_vec_t *vec);
    /**
     * @brief Uploads only a vector's halo elements to a compute device.
     * Does nothing if the vector is not present on the device.
     *
     * @param vec The vector.
     */
    ghost_error_t (*uploadHalo) (ghost_vec_t *vec);
    /**
     * @brief Uploads only a vector's local elements (i.e., without halo
     * elements) to a compute device. Does nothing if the vector is not
     * present on the device.
     *
     * @param vec The vector.
     */
    ghost_error_t          (*uploadNonHalo) (ghost_vec_t *vec);
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
    ghost_error_t          (*viewPlain) (ghost_vec_t *vec, void *data, ghost_vidx_t nr, ghost_vidx_t nc, ghost_vidx_t roffs, ghost_vidx_t coffs, ghost_vidx_t lda);

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
    ghost_error_t          (*vscale) (ghost_vec_t *, void *);
    ghost_error_t          (*vaxpy) (ghost_vec_t *, ghost_vec_t *, void *);
    ghost_error_t          (*vaxpby) (ghost_vec_t *, ghost_vec_t *, void *, void *);
    ghost_error_t          (*zero) (ghost_vec_t *);

#ifdef GHOST_HAVE_CUDA
    void * cu_val;
#endif
};

struct ghost_vtraits_t
{
    ghost_midx_t nrows;
    ghost_midx_t nrowshalo;
    ghost_midx_t nrowspadded;
    ghost_midx_t nvecs;
    ghost_vec_flags_t flags;
    int datatype;
    size_t elSize;
    void * aux;
    void * localdot;
};
extern const ghost_vtraits_t GHOST_VTRAITS_INITIALIZER;
#define GHOST_VTRAITS_INIT(...) {.flags = GHOST_VEC_DEFAULT, .aux = NULL, .datatype = GHOST_DT_DOUBLE|GHOST_DT_REAL, .nrows = 0, .nrowshalo = 0, .nrowspadded = 0, .nvecs = 1, .localdot = NULL, ## __VA_ARGS__ }

#ifdef MIC
//#define SELL_LEN 8
#define VEC_PAD 16
#elif defined (AVX)
#define VEC_PAD 4 // TODO single/double precision
#elif defined (SSE)
#define VEC_PAD 2
#elif defined (CUDA)
#define VEC_PAD 256
#elif defined (VSX)
#define VEC_PAD 2
#else
#define VEC_PAD 16
#endif

#define VECVAL(vec,val,__x,__y) &(val[__x][(__y)*vec->traits->elSize])
#define CUVECVAL(vec,val,__x,__y) &(val[((__x)*vec->traits->nrowspadded+(__y))*vec->traits->elSize])

#ifdef __cplusplus
template <typename v_t> ghost_error_t ghost_normalizeVector_tmpl(ghost_vec_t *vec);
template <typename v_t> ghost_error_t ghost_vec_dotprod_tmpl(ghost_vec_t *vec, ghost_vec_t *vec2, void *res);
template <typename v_t> ghost_error_t ghost_vec_vaxpy_tmpl(ghost_vec_t *vec, ghost_vec_t *vec2, void *);
template <typename v_t> ghost_error_t ghost_vec_vaxpby_tmpl(ghost_vec_t *vec, ghost_vec_t *vec2, void *, void *);
template <typename v_t> ghost_error_t ghost_vec_vscale_tmpl(ghost_vec_t *vec, void *vscale);
template <typename v_t> ghost_error_t ghost_vec_fromRand_tmpl(ghost_vec_t *vec);
template <typename v_t> ghost_error_t ghost_vec_print_tmpl(ghost_vec_t *vec);

extern "C" {
#endif

    ghost_error_t ghost_createVector(ghost_context_t *ctx, ghost_vtraits_t *traits, ghost_vec_t **vec);
    ghost_vtraits_t * ghost_cloneVtraits(ghost_vtraits_t *t1);

    ghost_error_t ghost_vec_malloc(ghost_vec_t *vec);
    ghost_error_t d_ghost_printVector(ghost_vec_t *vec); 
    ghost_error_t s_ghost_printVector(ghost_vec_t *vec); 
    ghost_error_t z_ghost_printVector(ghost_vec_t *vec);
    ghost_error_t c_ghost_printVector(ghost_vec_t *vec);
    ghost_error_t d_ghost_normalizeVector(ghost_vec_t *vec); 
    ghost_error_t s_ghost_normalizeVector(ghost_vec_t *vec); 
    ghost_error_t z_ghost_normalizeVector(ghost_vec_t *vec);
    ghost_error_t c_ghost_normalizeVector(ghost_vec_t *vec);
    ghost_error_t d_ghost_vec_dotprod(ghost_vec_t *vec1, ghost_vec_t *vec2, void *res); 
    ghost_error_t s_ghost_vec_dotprod(ghost_vec_t *vec1, ghost_vec_t *vec2, void *res); 
    ghost_error_t z_ghost_vec_dotprod(ghost_vec_t *vec1, ghost_vec_t *vec2, void *res);
    ghost_error_t c_ghost_vec_dotprod(ghost_vec_t *vec1, ghost_vec_t *vec2, void *res);
    ghost_error_t d_ghost_vec_vscale(ghost_vec_t *vec1, void *vscale); 
    ghost_error_t s_ghost_vec_vscale(ghost_vec_t *vec1, void *vscale); 
    ghost_error_t z_ghost_vec_vscale(ghost_vec_t *vec1, void *vscale);
    ghost_error_t c_ghost_vec_vscale(ghost_vec_t *vec1, void *vscale);
    ghost_error_t d_ghost_vec_vaxpy(ghost_vec_t *vec1, ghost_vec_t *vec2, void *); 
    ghost_error_t s_ghost_vec_vaxpy(ghost_vec_t *vec1, ghost_vec_t *vec2, void *); 
    ghost_error_t z_ghost_vec_vaxpy(ghost_vec_t *vec1, ghost_vec_t *vec2, void *);
    ghost_error_t c_ghost_vec_vaxpy(ghost_vec_t *vec1, ghost_vec_t *vec2, void *);
    ghost_error_t d_ghost_vec_vaxpby(ghost_vec_t *vec1, ghost_vec_t *vec2, void *, void *); 
    ghost_error_t s_ghost_vec_vaxpby(ghost_vec_t *vec1, ghost_vec_t *vec2, void *, void *); 
    ghost_error_t z_ghost_vec_vaxpby(ghost_vec_t *vec1, ghost_vec_t *vec2, void *, void *);
    ghost_error_t c_ghost_vec_vaxpby(ghost_vec_t *vec1, ghost_vec_t *vec2, void *, void *);
    ghost_error_t d_ghost_vec_fromRand(ghost_vec_t *vec); 
    ghost_error_t s_ghost_vec_fromRand(ghost_vec_t *vec); 
    ghost_error_t z_ghost_vec_fromRand(ghost_vec_t *vec); 
    ghost_error_t c_ghost_vec_fromRand(ghost_vec_t *vec); 
#ifdef __cplusplus
}
#endif

#if GHOST_HAVE_CUDA
#include "cu_vec.h"
#endif

#endif
