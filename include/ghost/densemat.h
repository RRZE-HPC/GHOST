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
#include "perm.h"
#include "bitmap.h"

#define GHOST_DENSEMAT_CHECK_SIMILARITY(vec1,vec2)\
    if (vec1->traits.nrows != vec2->traits.nrows) {\
        ERROR_LOG("Number of rows do not match!");\
        return GHOST_ERR_INVALID_ARG;\
    }\
    if (vec1->traits.ncols != vec2->traits.ncols) {\
        ERROR_LOG("Number of cols do not match!");\
        return GHOST_ERR_INVALID_ARG;\
    }\
    if (vec1->traits.storage != vec2->traits.storage) {\
        ERROR_LOG("Storage orders do not match!");\
        return GHOST_ERR_INVALID_ARG;\
    }\
    if (vec1->traits.location != vec2->traits.location) {\
        ERROR_LOG("Locations do not match!");\
        return GHOST_ERR_INVALID_ARG;\
    }


/**
 * @brief Flags to configure a densemat.
 */
typedef enum {
    GHOST_DENSEMAT_DEFAULT   = 0,
    /**
     * @brief Do not reserve space for halo elements.
     * 
     * This is applicable, e.g., if the densemat will never be an input (RHS) 
     * densemat to a SpMV.
     */
    GHOST_DENSEMAT_NO_HALO   = 1,
    /**
     * @brief The densemat is a view of another densemat.
     */
    GHOST_DENSEMAT_VIEW      = 32,
    /**
     * @brief The densemat is scattered in leading dimension, i.e., the rows/columns are not 
     * consecutive in memory. This is only possible for views. The densemat::ldmask is a valid bitmask.
     */
    GHOST_DENSEMAT_SCATTERED_LD = 64,
    /**
     * @brief The densemat is scattered in trailing dimension, i.e., the rows/columns are not 
     * consecutive in memory. This is only possible for views. The densemat::val has one entry for each row.
     */
    GHOST_DENSEMAT_SCATTERED_TR = 128,
    /**
     * @brief The densemat has been permuted in #GHOST_PERMUTATION_ORIG2PERM 
     * direction via its ghost_densemat_t::permute() function. 
     *
     * This flag gets deleted once the densemat has been permuted back 
     * (#GHOST_PERMUTATION_PERM2ORIG).
     */
    GHOST_DENSEMAT_PERMUTED = 256
} 
ghost_densemat_flags_t;

#define GHOST_DENSEMAT_SCATTERED (GHOST_DENSEMAT_SCATTERED_LD|GHOST_DENSEMAT_SCATTERED_TR)

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
    GHOST_DENSEMAT_COLMAJOR
}
ghost_densemat_storage_t;

/**
 * @brief Densemat halo exchange communication data structure.
 */
typedef struct
{
#ifdef GHOST_HAVE_MPI
    /**
     * @brief The number of messages sent.
     */
    int msgcount;
    /**
     * @brief The request array.
     */
    MPI_Request *request;
    /**
     * @brief The status array.
     */
    MPI_Status  *status;
    /**
     * @brief Holds a pointer where to receive from each PE.
     */
    char **tmprecv;
    /**
     * @brief This is NULL if receiving is done directly into the densemat halo.
     * In other cases (i.e., col-major with ncols>1 and row-major with missing 
     * columns), this will hold space for receiving elements.
     */
    char *tmprecv_mem;
    /**
     * @brief The assembled data to be sent.
     */
    char *work;
    /**
     * @brief Offset into the work array for each PE which receives data from me. 
     */
    ghost_lidx_t *dueptr;
    /**
     * @brief Offset into the tmprecv array for each PE from which I receive data. 
     */
    ghost_lidx_t *wishptr;
    /**
     * @brief Total number of dues.
     */
    ghost_lidx_t acc_dues;
    /**
     * @brief Total number of wishes.
     */
    ghost_lidx_t acc_wishes;
    /**
     * @brief The assembled data to be sent on the CUDA device.
     */
    void *cu_work;
#endif
}
ghost_densemat_halo_comm_t;

/**
 * @brief Traits of the densemat.
 */
typedef struct
{
    /**
     * @brief The number of rows.
     */
    ghost_lidx_t nrows;
    /**
     * @brief The number of rows of the densemat which is viewed by this 
     * densemat.
     */
    ghost_lidx_t nrowsorig;
    /**
     * @brief The number of rows including padding and halo elements.
     */
    ghost_lidx_t nrowshalo;
    /**
     * @brief The padded number of rows (may differ from nrows for col-major 
     * densemats).
     */
    ghost_lidx_t nrowspadded;
    /**
     * @brief The number of columns.
     */
    ghost_lidx_t ncols;
    /**
     * @brief The number of columns of the densemat which is viewed by this 
     * densemat.
     */
    ghost_lidx_t ncolsorig;
    /**
     * @brief The padded number of columns (may differ from ncols for row-major 
     * densemats).
     */
    ghost_lidx_t ncolspadded;
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
    /**
     * @brief Location of the densemat.
     */
    ghost_location_t location;
}
ghost_densemat_traits_t;

typedef struct ghost_densemat_t ghost_densemat_t;


/**
 * @ingroup types
 *
 * @brief A dense vector/matrix.  
 * 
 * The according functions are accessed via function pointers. 
 * The first argument of each member function always has to be a pointer to the 
 * densemat itself.
 */
struct ghost_densemat_t
{
    /**
     * @brief The densemat's traits.
     */
    ghost_densemat_traits_t traits;
    /**
     * @brief The context in which the densemat is living.
     */
    ghost_context_t *context;
    /**
     * @brief The values of the densemat.
     */
    char* val;
    /**
     * @brief The source densemat (must not be a view). 
     */
    ghost_densemat_t *src; 
    /**
     * @brief Size (in bytes) of one matrix element.
     */
    size_t elSize;
    /**
     * @brief The trailing dimensions of the densemat.
     *
     * Contains nrows if the densemat has row-major storage and 
     * ncols if it has col-major storage.
     */
    ghost_lidx_t nblock;
    /**
     * @brief The leading dimensions of the densemat.
     *
     * Contains ncols if the densemat has row-major storage and 
     * nrows if it has col-major storage.
     */
    ghost_lidx_t blocklen;
    /**
     * @brief The leading dimensions of the densemat in memory.
     *
     * Points to ncolspadded if the densemat has row-major storage and 
     * nrowspadded if it has col-major storage.
     */
    ghost_lidx_t stride;
    /**
     * @brief Masked out columns for scattered views
     */
    ghost_bitmap_t colmask;
    /**
     * @brief Masked out rows for scattered views
     */
    ghost_bitmap_t rowmask;
    /**
     * @brief An MPI data type which holds one element.
     */
    ghost_mpi_datatype_t mpidt;
    /**
     * @brief The values of the densemat on the CUDA device.
     */
    char * cu_val;

    /**
     * @brief Average each entry over all it's halo siblings.
     *
     * This function collects the values for each entry which have been
     * communicated to other processes.
     * Then, it computes the average over all and stores the value.
     * This is used, e.g., in ::ghost_carp().
     *
     * @param The densemat. 
     */
    ghost_error_t (*averageHalo) (ghost_densemat_t *vec);
    /** 
     * @ingroup locops
     *
     * @brief Performs <em>y := a*x + y</em> with scalar a
     *
     * @param y The in-/output densemat
     * @param x The input densemat
     * @param a Points to the scale factor.
     */
    ghost_error_t (*axpy) (ghost_densemat_t *y, ghost_densemat_t *x, void *a);
    /**
     * @ingroup locops
     *
     * @brief Performs <em>y := a*x + b*y</em> with scalar a and b
     *
     * @param y The in-/output densemat.
     * @param x The input densemat
     * @param a Points to the scale factor a.
     * @param b Points to the scale factor b.
     */
    ghost_error_t (*axpby) (ghost_densemat_t *y, ghost_densemat_t *x, void *a, 
            void *b);
    /**
     * @brief Clones a given number of columns of a source densemat at a 
     * given column and row offset.
     *
     * @param vec The source densemat.
     * @param dst Where to store the new vector.
     * @param nr The number of rows to clone.
     * @param roffs The first row to clone.
     * @param nc The number of columns to clone.
     * @param coffs The first column to clone.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t (*clone) (ghost_densemat_t *vec, ghost_densemat_t **dst, 
            ghost_lidx_t nr, ghost_lidx_t roffs, ghost_lidx_t nc, 
            ghost_lidx_t coffs);
    /**
     * @brief Compresses a densemat, i.e., make it non-scattered.
     * If the densemat is a view, it will no longer be one afterwards.
     *
     * @param vec The densemat.
     */
    ghost_error_t (*compress) (ghost_densemat_t *vec);
    /**
     * @brief Collects vec from all MPI ranks and combines them into globalVec.
     * The row permutation (if present) if vec's context is used.
     *
     * @param vec The distributed densemat.
     * @param globvec The global densemat.
     */
    ghost_error_t (*collect) (ghost_densemat_t *vec, ghost_densemat_t *globvec);
    /**
     * @brief Destroys a densemat, i.e., frees all its data structures.
     *
     * @param vec The densemat
     */
    void          (*destroy) (ghost_densemat_t *vec);
 
    /**
     * @brief Distributes a global densemat into node-local vetors.
     *
     * @param vec The global densemat.
     * @param localVec The local densemat.
     */
    ghost_error_t (*distribute) (ghost_densemat_t *vec, 
            ghost_densemat_t *localVec);
    /**
     * @ingroup locops
     * 
     * @brief Compute the local dot product of two vectors/matrices.
     *
     * @param a The first densemat.
     * @param res Where to store the result.
     * @param b The second densemat.
     *
     * For complex data, the first vector gets conjugated (like the BLAS call dotc()).
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     *
     * For the global operation see ghost_dot().
     *
     * @see ghost_dot()
     */
    ghost_error_t (*dot) (ghost_densemat_t *a, void *res, ghost_densemat_t *b);
    /**
     * @ingroup gputransfer
     * 
     * @brief Downloads a densemat from a compute device, excluding halo elements. 
     *
     * The densemat must be present on both host and device.
     * Does nothing if the densemat is not present on the device.
     *
     * @param vec The densemat.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t (*download) (ghost_densemat_t *vec);
    /**
     * @ingroup gputransfer
     * 
     * @brief Downloads only a densemat's local elements (i.e., without halo
     * elements) from a compute device. Does nothing if the densemat is not
     * present on the device.
     *
     * @param vec The densemat.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t (*entry) (ghost_densemat_t *vec, void *entry, ghost_lidx_t i, 
            ghost_lidx_t j);
    /**
     * @ingroup denseinit
     *
     * @brief Initializes a densemat from a given function.
     * Malloc's memory for the densemat's values if this hasn't happened before.
     *
     * @param vec The densemat.
     * @param fp The function pointer. The function takes three arguments: The
     *  row index, the column index and a pointer to where to store the value at 
     *  this position.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t (*fromFunc) (ghost_densemat_t *vec, int (*fp)(ghost_gidx_t, 
                ghost_lidx_t, void *));
    /**
     * @ingroup denseinit
     *
     * @brief Initializes a densemat from another densemat at a given column and 
     * row offset.
     *
     * Malloc's memory for the densemat's values if this hasn't happened before.
     *
     * @param vec The densemat.
     * @param src The source.
     * @param roffs The first row to clone.
     * @param coffs The first column to clone.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t (*fromVec) (ghost_densemat_t *vec, ghost_densemat_t *src, 
            ghost_lidx_t roffs, ghost_lidx_t coffs);
    /**
     * @ingroup denseinit
     *
     * @brief Initializes a densemat from a file.
     * Malloc's memory for the densemat's values if this hasn't happened before.
     *
     * @param vec The densemat.
     * @param filename Path to the file.
     * @param singleFile Read from a single (global) file. Ignored in the 
     * non-MPI case.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t (*fromFile) (ghost_densemat_t *vec, char *filename, 
            bool singleFile);
    /**
     * @ingroup denseinit
     *
     * @brief Initiliazes a densemat from random values.
     *
     * @param vec The densemat.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t (*fromRand) (ghost_densemat_t *vec);
    /**
     * @brief Sets the densemat to have the same values on all processes.
     *
     * @param vec The densemat.
     * @param root The process from which to take the values.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t (*syncValues) (ghost_densemat_t *vec, ghost_mpi_comm_t, int root);
    /**
     * @ingroup denseinit
     *
     * @brief Reduces the densemats in a given communicator.
     *
     * @param vec The densemat.
     * @param comm The communicator.
     * @param dest The destination rank or GHOST_ALLREDUCE
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t (*reduce) (ghost_densemat_t *vec, ghost_mpi_comm_t comm, int dest);
    /**
     * @ingroup denseinit
     *
     * @brief Initializes a densemat from a given scalar value.
     *
     * @param vec The densemat.
     * @param val A pointer to the value.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t (*fromScalar) (ghost_densemat_t *vec, void *val);
    /**
     * @brief Initialize a halo communication data structure.
     *
     * @param vec The densemat.
     * @param comm The halo communication data structure.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t (*halocommInit) (ghost_densemat_t *vec, ghost_densemat_halo_comm_t *comm);
    /**
     * @brief Start halo communication asynchronously.
     *
     * @param vec The densemat.
     * @param comm The halo communication data structure.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t (*halocommStart) (ghost_densemat_t *vec, ghost_densemat_halo_comm_t *comm);
    /**
     * @brief Finalize halo communication.
     *
     * This includes waiting for the communication to finish and freeing the data in the comm data structure.
     *
     * @param vec The densemat.
     * @param comm The halo communication data structure.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t (*halocommFinalize) (ghost_densemat_t *vec, ghost_densemat_halo_comm_t *comm);
    /**
     * @brief Normalize a densemat, i.e., scale it such that its 2-norm is one.
     *
     * @param vec The densemat.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t (*normalize) (ghost_densemat_t *vec);
    /**
     * @brief Compute the norm of a densemat: sum_i [conj(vec_i) * vec_i]^pow
     *
     * @param vec The densemat.
     * @param norm Where to store the norm. Must be a pointer to the densemat's data type.
     * @param pow The power. Must be a pointer to the densemat's data type.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t (*norm) (ghost_densemat_t *vec, void *norm, void *pow);
    /**
     * @brief Permute a densemat with a given permutation.
     *
     * @param vec The densemat.
     * @param perm The permutation.
     * @param dir The permutation direction.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t (*permute) (ghost_densemat_t *vec, ghost_permutation_direction_t dir);
    /**
     * @ingroup stringification
     *
     * @brief Create a string from the vector.
     *
     * @param vec The densemat.
     * @param str Where to store the string.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t (*string) (ghost_densemat_t *vec, char **str);
    /**
     * @ingroup locops
     *
     * @brief Scale a densemat with a given scalar.
     *
     * @param vec The densemat.
     * @param scale The scale factor.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t (*scale) (ghost_densemat_t *vec, void *scale);
    /**
     * @brief Write a densemat to a file.
     *
     * @param vec The densemat.
     * @param filename The path to the file.
     * @param singleFile Write to a single (global) file. Ignored in the 
     * non-MPI case.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t (*toFile) (ghost_densemat_t *vec, char *filename, 
            bool singleFile);
    /**
     * @ingroup gputransfer
     * 
     * @brief Uploads a densemat to a compute device, excluding halo elements. 
     * The densemat must be present on both host and device.
     *
     * @param vec The densemat.
     */
    ghost_error_t (*upload) (ghost_densemat_t *vec);
    /**
     * @ingroup gputransfer
     * 
     * @brief Uploads only a densemat's local elements (i.e., without halo
     * elements) to a compute device. Does nothing if the densemat is not
     * present on the device.
     *
     * @param vec The densemat.
     */
    ghost_error_t (*viewPlain) (ghost_densemat_t *vec, void *data, 
            ghost_lidx_t lda);
    /**
     * @brief Create a densemat as a scattered view of another densemat.
     *
     * @param src The source densemat.
     * @param nr The nunber of rows to view.
     * @param roffs The row offsets.
     * @param dst Where to store the new vector.
     * @param nc The nunber of columns to view.
     * @param coffs The column offsets.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t (*viewScatteredVec) (ghost_densemat_t *src, 
            ghost_densemat_t **dst, ghost_lidx_t nr, ghost_lidx_t *roffs,  
            ghost_lidx_t nc, ghost_lidx_t *coffs);
    /**
     * @brief Create a densemat as a view of a scattered block of columns of 
     * another densemat.
     *
     * @param src The source densemat. 
     * @param dst Where to store the new densemat view.
     * @param nc The number of columns.
     * @param coffs The column index of each viewed column in the source 
     * densemat.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t (*viewScatteredCols) (ghost_densemat_t *src, 
            ghost_densemat_t **dst, ghost_lidx_t nc, ghost_lidx_t *coffs);
    /**
     * @brief Create a densemat as a view of a dense block of columns of another 
     * densemat.
     *
     * @param src The source densemat. 
     * @param dst Where to store the new densemat view.
     * @param nc The number of columns.
     * @param coffs The column offset in the source densemat.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t (*viewCols) (ghost_densemat_t *src, ghost_densemat_t **dst, 
            ghost_lidx_t nc, ghost_lidx_t coffs);
    /**
     * @brief Create a densemat as a view of another densemat.
     *
     * @param src The source densemat.
     * @param nr The nunber of rows to view.
     * @param roffs The row offset.
     * @param dst Where to store the new vector.
     * @param nc The nunber of columns to view.
     * @param coffs The column offset.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t (*viewVec) (ghost_densemat_t *src, ghost_densemat_t **dst, 
            ghost_lidx_t nr, ghost_lidx_t roffs, ghost_lidx_t nc, 
            ghost_lidx_t coffs);
    /**
     * @ingroup locops
     * 
     * @brief Scale each column of a densemat with a separate scale factor.
     *
     * @param vec The densemat.
     * @param scale The scale factors. Length must be the same as the number of 
     * densemat columns.
     */
    ghost_error_t (*vscale) (ghost_densemat_t *, void *);
    /** 
     * @ingroup locops
     *
     * @brief Performs <em>y := a_i*x + y</em> with separate scalars a_i for 
     * each column
     *
     * @param y The in-/output densemat
     * @param x The input densemat
     * @param a The scale factors. Length must be the same as the number of 
     * densemat columns.
     */
    ghost_error_t (*vaxpy) (ghost_densemat_t *, ghost_densemat_t *, void *);
    /**
     * @ingroup locops
     *
     * @brief Performs <em>y := a_i*x + b_i*y</em> with separate scalars a_i 
     * and b_i
     *
     * @param y The in-/output densemat.
     * @param x The input densemat
     * @param a The scale factors a. Length must be the same as the number of 
     * densemat columns.
     * @param b The scale factors b. Length must be the same as the number of 
     * densemat columns.
     */
    ghost_error_t (*vaxpby) (ghost_densemat_t *, ghost_densemat_t *, void *, 
            void *);
};

#ifdef __cplusplus
static inline ghost_densemat_flags_t operator|(const ghost_densemat_flags_t &a, const ghost_densemat_flags_t &b) {
return static_cast<ghost_densemat_flags_t>(static_cast<int>(a) | static_cast<int>(b));
}
static inline ghost_densemat_flags_t operator|=(ghost_densemat_flags_t &a, const ghost_densemat_flags_t &b) {
    a = static_cast<ghost_densemat_flags_t>(static_cast<int>(a) | static_cast<int>(b));
    return a;
}
static inline ghost_densemat_flags_t operator&=(ghost_densemat_flags_t &a, const ghost_densemat_flags_t &b) {
    a = static_cast<ghost_densemat_flags_t>(static_cast<int>(a) & static_cast<int>(b));
    return a;
}

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
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_densemat_create(ghost_densemat_t **vec, 
            ghost_context_t *ctx, ghost_densemat_traits_t traits);
    
    /**
     * @brief Get the location of the first viewed data element.
     *
     * @param vec The densemat.
     * @param ptr Where to store the pointer.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_densemat_valptr(ghost_densemat_t *vec, void **ptr);
    ghost_error_t ghost_densemat_cu_valptr(ghost_densemat_t *vec, void **ptr);
    /**
     * @brief Create an array of chars ('0' or '1') of the densemat mask.
     *
     * @param mask The ldmask.
     * @param len Length of the ldmask.
     * @param charfield Location of the char array.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_densemat_mask2charfield(ghost_bitmap_t mask, 
            ghost_lidx_t len, char *charfield);

    /**
     * @brief Check if an array consists of strictly ascending numbers.
     *
     * @param coffs The numbers.
     * @param nc Length of the array.
     *
     * @return True if each number is greater than the previous one, 
     * false otherwise.
     */
    bool array_strictly_ascending (ghost_lidx_t *coffs, ghost_lidx_t nc);

    /**
     * @brief Check if a densemat has the same storage order on all processes.
     *
     * @param uniform Where to store the result of the check.
     * @param vec The densemat.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_densemat_uniformstorage(bool *uniform, 
            ghost_densemat_t *vec);

    /**
     * @ingroup stringification
     *
     * @brief Get a string about the storage order.
     */
    char * ghost_densemat_storage_string(ghost_densemat_t *densemat);
    
    /**
     * @ingroup stringification
     *
     * @brief Get a string containing information about the densemat.
     */
    ghost_error_t ghost_densemat_info_string(char **str, 
            ghost_densemat_t *densemat);

    /**
     * @brief Common (storage-independent) functions for ghost_densemat_t::halocommInit()
     *
     * This function should not be called by a user.
     *
     * @param vec The densemat.
     * @param comm The comm data structure.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_densemat_halocommInit_common(ghost_densemat_t *vec, ghost_densemat_halo_comm_t *comm);
    /**
     * @brief Common (storage-independent) functions for ghost_densemat_t::halocommStart()
     *
     * This function should not be called by a user.
     *
     * @param vec The densemat.
     * @param comm The comm data structure.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_densemat_halocommStart_common(ghost_densemat_t *vec, ghost_densemat_halo_comm_t *comm);
    /**
     * @brief Common (storage-independent) functions for ghost_densemat_t::halocommFinalize()
     *
     * This function should not be called by a user.
     *
     * @param comm The comm data structure.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_densemat_halocommFinalize_common(ghost_densemat_halo_comm_t *comm);

    void ghost_densemat_destroy(ghost_densemat_t* vec);

    /**
     * @brief Determine the number of padded rows.
     *
     * The offset from which halo elements begin and the column indices of remote matrix entries depend on this.
     *
     * @return The number of padded rows.
     */
    ghost_lidx_t ghost_densemat_row_padding();

#ifdef __cplusplus
}
#endif

/**
 * @brief Initializer for densemat traits.
 */
extern const ghost_densemat_traits_t GHOST_DENSEMAT_TRAITS_INITIALIZER;

/**
 * @brief Initializer for densemat halo communicator.
 */
extern const ghost_densemat_halo_comm_t GHOST_DENSEMAT_HALO_COMM_INITIALIZER;

#endif
