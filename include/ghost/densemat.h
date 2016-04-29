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
#include "bitmap.h"
#include "bindensemat.h"

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
 * @brief Callback function to construct a ghost_densemat
 *
 * @param[in] row The global row index.
 * @param[in] col The column.
 * @param[out] val The value at the specified location.
 * @param[inout] arg Additional arguments.
 *
 * @return  
 */
typedef int (*ghost_densemat_srcfunc)(ghost_gidx row, ghost_lidx col, void * val, void * arg);

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
     * consecutive in memory. This is only possible for views. 
     */
    GHOST_DENSEMAT_SCATTERED_LD = 64,
    /**
     * @brief The densemat is scattered in trailing dimension, i.e., the rows/columns are not 
     * consecutive in memory. This is only possible for views. 
     */
    GHOST_DENSEMAT_SCATTERED_TR = 128,
    /**
     * @brief The densemat has been permuted in #GHOST_PERMUTATION_ORIG2PERM 
     * direction via its ghost_densemat::permute() function. 
     *
     * This flag gets deleted once the densemat has been permuted back 
     * (#GHOST_PERMUTATION_PERM2ORIG).
     */
    GHOST_DENSEMAT_PERMUTED = 256,
    /**
     * @brief By default, a densemat's location gets set to #GHOST_DENSEMAT_HOST|#GHOST_DENSEMAT_DEVICE automatically
     * when the first up-/download occurs and the GHOST type is CUDA. This behavior can be disabled by setting this flag.
     */
    GHOST_DENSEMAT_NOT_RELOCATE = 512,
    /**
     * @brief Set this flag if the number of columns should be padded according to the SIMD width.
     */
    GHOST_DENSEMAT_PAD_COLS = 1024
} 
ghost_densemat_flags;

#define GHOST_DENSEMAT_SCATTERED (GHOST_DENSEMAT_SCATTERED_LD|GHOST_DENSEMAT_SCATTERED_TR)

/**
 * @brief Densemat storage orders
 */
typedef enum 
{
    /**
     * @brief Row-major storage (as in C).
     */
    GHOST_DENSEMAT_ROWMAJOR = GHOST_BINDENSEMAT_ORDER_ROW_FIRST,
    /**
     * @brief Column-major storage (as in Fortran).
     */
    GHOST_DENSEMAT_COLMAJOR = GHOST_BINDENSEMAT_ORDER_COL_FIRST,
    GHOST_DENSEMAT_STORAGE_DEFAULT = 2
}
ghost_densemat_storage;

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
    ghost_lidx *dueptr;
    /**
     * @brief Offset into the tmprecv array for each PE from which I receive data. 
     */
    ghost_lidx *wishptr;
    /**
     * @brief Total number of dues.
     */
    ghost_lidx acc_dues;
    /**
     * @brief Total number of wishes.
     */
    ghost_lidx acc_wishes;
    /**
     * @brief The assembled data to be sent on the CUDA device.
     */
    void *cu_work;
#endif
}
ghost_densemat_halo_comm;

/**
 * @brief Traits of the densemat.
 */
typedef struct
{
    /**
     * @brief The number of rows.
     */
    ghost_lidx nrows;
    /**
     * @brief The number of rows of the densemat which is viewed by this 
     * densemat.
     */
    ghost_lidx nrowsorig;
    /**
     * @brief The number of rows including padding and halo elements.
     */
    ghost_lidx nrowshalo;
    /**
     * @brief The padded number of rows (may differ from nrows for col-major 
     * densemats).
     */
    ghost_lidx nrowspadded;
    /**
     * @brief The number of rows including padding, halo, and halo-padding elements
     * There is another padding after the halo elements to guarantee aligned access to successive columns for col-major densemats.
     */
    ghost_lidx nrowshalopadded;
    /**
     * @brief The number of columns.
     */
    ghost_lidx ncols;
    /**
     * @brief The number of columns of the densemat which is viewed by this 
     * densemat.
     */
    ghost_lidx ncolsorig;
    /**
     * @brief The padded number of columns (may differ from ncols for row-major 
     * densemats).
     */
    ghost_lidx ncolspadded;
    /**
     * @brief Property flags.
     */
    ghost_densemat_flags flags;
    /**
     * @brief The storage order.
     */
    ghost_densemat_storage storage;
    /**
     * @brief The data type.
     */
    ghost_datatype datatype;
    /**
     * @brief Location of the densemat.
     */
    ghost_location location;
}
ghost_densemat_traits;

typedef struct ghost_densemat ghost_densemat;


/**
 * @ingroup types
 *
 * @brief A dense vector/matrix.  
 * 
 * The according functions are accessed via function pointers. 
 * The first argument of each member function always has to be a pointer to the 
 * densemat itself.
 */
struct ghost_densemat
{
    /**
     * @brief The densemat's traits.
     */
    ghost_densemat_traits traits;
    /**
     * @brief The context in which the densemat is living.
     */
    ghost_context *context;
    /**
     * @brief The values of the densemat.
     */
    char* val;
    /**
     * @brief The source densemat (must not be a view). 
     */
    ghost_densemat *src; 
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
    ghost_lidx nblock;
    /**
     * @brief The leading dimensions of the densemat.
     *
     * Contains ncols if the densemat has row-major storage and 
     * nrows if it has col-major storage.
     */
    ghost_lidx blocklen;
    /**
     * @brief The leading dimensions of the densemat in memory.
     *
     * Points to ncolspadded if the densemat has row-major storage and 
     * nrowspadded if it has col-major storage.
     */
    ghost_lidx stride;
    /**
     * @brief Masked out columns for scattered views
     */
    ghost_bitmap colmask;
    /**
     * @brief Masked out rows for scattered views
     */
    ghost_bitmap rowmask;
    /**
     * @brief An MPI data type which holds one element.
     */
    ghost_mpi_datatype mpidt;
    /**
     * @brief An MPI data type which holds the entire densemat.
     */
    ghost_mpi_datatype fullmpidt;
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
    ghost_error (*averageHalo) (ghost_densemat *vec);
    /**
     * Documented in ghost_axpy()
     */
    ghost_error (*axpy) (ghost_densemat *y, ghost_densemat *x, void *a);
    /**
     * Documented in ghost_axpby()
     */
    ghost_error (*axpby) (ghost_densemat *y, ghost_densemat *x, void *a, 
            void *b);
    /**
     * Documented in ghost_axpbypcz()
     */
    ghost_error (*axpbypcz) (ghost_densemat *y, ghost_densemat *x, void *a, 
            void *b,ghost_densemat *z, void *c);
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
    ghost_error (*clone) (ghost_densemat *vec, ghost_densemat **dst, 
            ghost_lidx nr, ghost_lidx roffs, ghost_lidx nc, 
            ghost_lidx coffs);
    /**
     * @brief Compresses a densemat, i.e., make it non-scattered.
     * If the densemat is a view, it will no longer be one afterwards.
     *
     * @param vec The densemat.
     */
    ghost_error (*compress) (ghost_densemat *vec);
    /**
     * Documented in ghost_conj()
     */
    ghost_error (*conj) (ghost_densemat *vec);
    /**
     * @brief Collects vec from all MPI ranks and combines them into globalVec.
     * The row permutation (if present) if vec's context is used.
     *
     * @param vec The distributed densemat.
     * @param globvec The global densemat.
     */
    ghost_error (*collect) (ghost_densemat *vec, ghost_densemat *globvec);
 
    /**
     * @brief Distributes a global densemat into node-local vetors.
     *
     * @param vec The global densemat.
     * @param localVec The local densemat.
     */
    ghost_error (*distribute) (ghost_densemat *vec, 
            ghost_densemat *localVec);
    /**
     * Fallback dot product of two vectors.
     * This function should not be called directly, see ghost_dot() and ghost_localdot() instead.
     */
    ghost_error (*localdot_vanilla) (ghost_densemat *a, void *res, ghost_densemat *b);
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
    ghost_error (*download) (ghost_densemat *vec);
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
    ghost_error (*entry) (ghost_densemat *vec, void *entry, ghost_lidx i, 
            ghost_lidx j);
    /**
     * Documented in ghost_densemat_init_func()
     */
    ghost_error (*fromFunc) (ghost_densemat *vec, ghost_densemat_srcfunc, void *arg);
    /**
     * Documented in ghost_densemat_init_densemat()
     */
    ghost_error (*fromVec) (ghost_densemat *vec, ghost_densemat *src, 
            ghost_lidx roffs, ghost_lidx coffs);
    /**
     * Documented in ghost_densemat_init_file)
     */
    ghost_error (*fromFile) (ghost_densemat *vec, char *filename, bool singleFile);
    /**
     * Documented in ghost_densemat_init_rand()
     */
    ghost_error (*fromRand) (ghost_densemat *vec);
    /**
     * @brief Sets the densemat to have the same values on all processes.
     *
     * @param vec The densemat.
     * @param comm The communicator in which to synchronize.
     * @param root The process from which to take the values.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error (*syncValues) (ghost_densemat *vec, ghost_mpi_comm comm, int root);
    /**
     * @brief Reduces the densemats using addition in a given communicator.
     *
     * @param vec The densemat.
     * @param comm The communicator.
     * @param dest The destination rank or GHOST_ALLREDUCE
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error (*reduce) (ghost_densemat *vec, ghost_mpi_comm comm, int dest);
    /**
     * Documented in ghost_densemat_init_val()
     */
    ghost_error (*fromScalar) (ghost_densemat *vec, void *val);
    /**
     * @brief Initialize a halo communication data structure.
     *
     * @param vec The densemat.
     * @param comm The halo communication data structure.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error (*halocommInit) (ghost_densemat *vec, ghost_densemat_halo_comm *comm);
    /**
     * @brief Start halo communication asynchronously.
     *
     * @param vec The densemat.
     * @param comm The halo communication data structure.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error (*halocommStart) (ghost_densemat *vec, ghost_densemat_halo_comm *comm);
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
    ghost_error (*halocommFinalize) (ghost_densemat *vec, ghost_densemat_halo_comm *comm);
    /**
     * Documented in ghost_normalize()
     */
    ghost_error (*normalize) (ghost_densemat *vec);
    /**
     * @brief Compute the norm of a densemat: sum_i [conj(vec_i) * vec_i]^pow
     *
     * @param vec The densemat.
     * @param norm Where to store the norm. Must be a pointer to the densemat's data type.
     * @param pow The power. Must be a pointer to the densemat's data type.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error (*norm) (ghost_densemat *vec, void *norm, void *pow);
    /**
     * Documented in ghost_densemat_permute()
     */
    ghost_error (*permute) (ghost_densemat *vec, ghost_permutation_direction dir);
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
    ghost_error (*string) (ghost_densemat *vec, char **str);
    /**
     * Documented in ghost_scale()
     */
    ghost_error (*scale) (ghost_densemat *vec, void *scale);
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
    ghost_error (*toFile) (ghost_densemat *vec, char *filename, 
            bool singleFile);
    /**
     * @ingroup gputransfer
     * 
     * @brief Uploads a densemat to a compute device, excluding halo elements. 
     * The densemat must be present on both host and device.
     *
     * @param vec The densemat.
     */
    ghost_error (*upload) (ghost_densemat *vec);
    /**
     * Documented in ghost_densemat_view_plain()
     */
    ghost_error (*viewPlain) (ghost_densemat *vec, void *data, 
            ghost_lidx lda);
    /**
     * Documented in ghost_densemat_create_and_view_densemat_scattered()
     */
    ghost_error (*viewScatteredVec) (ghost_densemat *src, 
            ghost_densemat **dst, ghost_lidx nr, ghost_lidx *roffs,  
            ghost_lidx nc, ghost_lidx *coffs);
    /**
     * Documented in ghost_densemat_create_and_view_densemat_cols_scattered()
     */
    ghost_error (*viewScatteredCols) (ghost_densemat *src, 
            ghost_densemat **dst, ghost_lidx nc, ghost_lidx *coffs);
    /**
     * Documented in ghost_densemat_create_and_view_densemat_cols()
     */
    ghost_error (*viewCols) (ghost_densemat *src, ghost_densemat **dst, 
            ghost_lidx nc, ghost_lidx coffs);
    /**
     * Documented in ghost_densemat_create_and_view_densemat()
     */
    ghost_error (*viewVec) (ghost_densemat *src, ghost_densemat **dst, 
            ghost_lidx nr, ghost_lidx roffs, ghost_lidx nc, 
            ghost_lidx coffs);
    /**
     * Documented in ghost_vscale()
     */
    ghost_error (*vscale) (ghost_densemat *, void *);
    /**
     * Documented in ghost_vaxpy()
     */
    ghost_error (*vaxpy) (ghost_densemat *, ghost_densemat *, void *);
    /**
     * Documented in ghost_vaxpby()
     */
    ghost_error (*vaxpby) (ghost_densemat *, ghost_densemat *, void *, 
            void *);
    /**
     * Documented in ghost_vaxpbypcz()
     */
    ghost_error (*vaxpbypcz) (ghost_densemat *, ghost_densemat *, void *, 
            void *, ghost_densemat *, void *);
};

#ifdef __cplusplus
static inline ghost_densemat_flags operator|(const ghost_densemat_flags &a, const ghost_densemat_flags &b) {
return static_cast<ghost_densemat_flags>(static_cast<int>(a) | static_cast<int>(b));
}
static inline ghost_densemat_flags operator|=(ghost_densemat_flags &a, const ghost_densemat_flags &b) {
    a = static_cast<ghost_densemat_flags>(static_cast<int>(a) | static_cast<int>(b));
    return a;
}
static inline ghost_densemat_flags operator&=(ghost_densemat_flags &a, const ghost_densemat_flags &b) {
    a = static_cast<ghost_densemat_flags>(static_cast<int>(a) & static_cast<int>(b));
    return a;
}

extern "C" {
#endif

    /**
     * @ingroup types
     * @ingroup densecreate
     *
     * @brief Create a dense matrix/vector. 
     *
     * @param vec Where to store the matrix.
     * @param ctx The context the matrix lives in or NULL.
     * @param traits The matrix traits.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     *
     * @note No memory will be allocated in this function. Before any operation with the densemat is done,
     * an initialization function (see @ref denseinit) has to be called with the densemat.
     */
    ghost_error ghost_densemat_create(ghost_densemat **vec, 
            ghost_context *ctx, ghost_densemat_traits traits);
    
    /**
     * @brief Create an array of chars ('0' or '1') of the densemat mask.
     *
     * @param mask The ldmask.
     * @param len Length of the ldmask.
     * @param charfield Location of the char array.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_mask2charfield(ghost_bitmap mask, 
            ghost_lidx len, char *charfield);

    /**
     * @brief Check if an array consists of strictly ascending numbers.
     *
     * @param coffs The numbers.
     * @param nc Length of the array.
     *
     * @return True if each number is greater than the previous one, 
     * false otherwise.
     */
    bool array_strictly_ascending (ghost_lidx *coffs, ghost_lidx nc);

    /**
     * @brief Check if a densemat has the same storage order on all processes.
     *
     * @param uniform Where to store the result of the check.
     * @param vec The densemat.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_uniformstorage(bool *uniform, 
            ghost_densemat *vec);

    /**
     * @ingroup stringification
     *
     * @brief Get a string about the storage order.
     */
    char * ghost_densemat_storage_string(ghost_densemat_storage storage);
    
    /**
     * @ingroup stringification
     *
     * @brief Get a string containing information about the densemat.
     */
    ghost_error ghost_densemat_info_string(char **str, 
            ghost_densemat *densemat);

    /**
     * @brief Common (storage-independent) functions for ghost_densemat::halocommInit()
     *
     * This function should not be called by a user.
     *
     * @param vec The densemat.
     * @param comm The comm data structure.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_halocommInit_common(ghost_densemat *vec, ghost_densemat_halo_comm *comm);
    /**
     * @brief Common (storage-independent) functions for ghost_densemat::halocommStart()
     *
     * This function should not be called by a user.
     *
     * @param vec The densemat.
     * @param comm The comm data structure.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_halocommStart_common(ghost_densemat *vec, ghost_densemat_halo_comm *comm);
    /**
     * @brief Common (storage-independent) functions for ghost_densemat::halocommFinalize()
     *
     * This function should not be called by a user.
     *
     * @param comm The comm data structure.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_halocommFinalize_common(ghost_densemat_halo_comm *comm);

    /**
     * @ingroup teardown
     * @brief Destroys a densemat, i.e., frees all its data structures.
     * @param vec The densemat
     */
    void ghost_densemat_destroy(ghost_densemat* vec);

    /**
     * @brief Determine the number of padded rows.
     *
     * The offset from which halo elements begin and the column indices of remote matrix entries depend on this.
     *
     * @return The number of padded rows.
     */
    ghost_lidx ghost_densemat_row_padding();

#ifdef __cplusplus
}
#endif

/**
 * @brief Initializer for densemat traits.
 */
extern const ghost_densemat_traits GHOST_DENSEMAT_TRAITS_INITIALIZER;

/**
 * @brief Initializer for densemat halo communicator.
 */
extern const ghost_densemat_halo_comm GHOST_DENSEMAT_HALO_COMM_INITIALIZER;

#endif
