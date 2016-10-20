#ifndef GHOST_DENSEMAT_H
#define GHOST_DENSEMAT_H

#include "config.h"
#include "types.h"
#include "bitmap.h"
#include "bindensemat.h"
#include "context.h"

#define GHOST_CM_IDX 0
#define GHOST_RM_IDX 1

#define GHOST_DENSEMAT_CHECK_SIMILARITY(vec1,vec2)\
    if (DM_NROWS(vec1) != DM_NROWS(vec2)) {\
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
    if (!(vec1->traits.location & vec2->traits.location)) {\
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
    GHOST_DENSEMAT_DEFAULT = 0,
    /**
     * @brief The densemat is a view of another densemat.
     */
    GHOST_DENSEMAT_VIEW = 1<<0,
    /**
     * @brief The densemat is scattered in leading dimension, i.e., the rows/columns are not 
     * consecutive in memory. This is only possible for views. 
     */
    GHOST_DENSEMAT_SCATTERED_LD = 1<<1,
    /**
     * @brief The densemat is scattered in trailing dimension, i.e., the rows/columns are not 
     * consecutive in memory. This is only possible for views. 
     */
    GHOST_DENSEMAT_SCATTERED_TR = 1<<2,
    /**
    * @brief The densemat has been permuted in #GHOST_PERMUTATION_ORIG2PERM 
    * direction via its ghost_densemat::permute() function. 
    *
    * This flag gets deleted once the densemat has been permuted back 
    * (#GHOST_PERMUTATION_PERM2ORIG).
    */
    GHOST_DENSEMAT_PERMUTED = 1<<3,
   /**
     * @brief By default, a densemat's location gets set to #GHOST_DENSEMAT_HOST|#GHOST_DENSEMAT_DEVICE automatically
     * when the first up-/download occurs and the GHOST type is CUDA. This behavior can be disabled by setting this flag.
     */
    GHOST_DENSEMAT_NOT_RELOCATE = 1<<4,
    /**
     * @brief Set this flag if the number of columns should be padded according to the SIMD width.
     */
    GHOST_DENSEMAT_PAD_COLS = 1<<5,
    /**
     * @brief Destroy the densemat's map when the densemat gets destroyed.
     *
     * This flag should not be set by the user.
     */
    GHOST_DENSEMAT_FREE_MAP = 1<<6,
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
     * @brief The number of columns.
     */
    ghost_lidx ncols;
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
    /**
     * @brief If the densemat is the result of a computation, decide where to execute the computation.
     *
     * This is only relevant if all involved data is stored on both HOST and DEVICE.
     */
    ghost_location compute_at;
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
     * Equal to ncolspadded if the densemat has row-major storage and 
     * nrowshalopadded if it has col-major storage.
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
     * @brief The densemat's map. 
     */
    ghost_map *map;
};

#define DM_NROWS(dm) dm->map->dim
#define DM_GNROWS(dm) dm->map->gdim
#define DM_NROWSPAD(dm) dm->map->dimpad

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
     * @param map The map of the matrix. This may be extracted from an existing context or created separately.
     * @param traits The matrix traits.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     *
     * @note No memory will be allocated in this function. Before any operation with the densemat is done,
     * an initialization function (see @ref denseinit) has to be called with the densemat.
     */
    ghost_error ghost_densemat_create(ghost_densemat **vec, ghost_map *map, ghost_densemat_traits traits);
    
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
     * @brief Check if a densemat has the same storage order on all processes of a given communicator.
     *
     * @param uniform Where to store the result of the check.
     * @param vec The densemat.
     * @param mpicomm The communicator.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_uniformstorage(bool *uniform, 
            ghost_densemat *vec,ghost_mpi_comm mpicomm);

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
     * @param ctx The context in which to communicate.
     * @param comm The comm data structure.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_halocommInit_common(ghost_densemat *vec, ghost_context *ctx, ghost_densemat_halo_comm *comm);
    /**
     * @brief Start halo communication asynchronously.
     *
     * @param vec The densemat.
     * @param ctx The context in which to communicate.
     * @param comm The halo communication data structure.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_halocomm_start(ghost_densemat *vec, ghost_context *ctx, ghost_densemat_halo_comm *comm);
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

    /**
     * @brief Translates a ghost_densemat_storage into an consectuive index.
     *
     * @param s The storage type.
     *
     * @return The index.
     */
    int ghost_idx_of_densemat_storage(ghost_densemat_storage s); 
    
    /**
     * @ingroup denseinit
     * @brief Initializes a densemat from random values.
     * @param x The densemat.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_init_rand(ghost_densemat *x);
    
    /**
     * @ingroup denseinit
     * @brief Initializes a densemat from a scalar value.
     * @param x The densemat.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_init_val(ghost_densemat *x, void *v);

    /**
     * @brief Allocate sparse for a densemat
     *
     * @param x The densemat
     * @param needInit This will be set to one of the padding elements require initialization
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_malloc(ghost_densemat *x, int *needInit);
    /**
     * @ingroup denseinit
     * @brief Initializes a densemat from a given callback function.
     * @param x The densemat.
     * @param func The callback function pointer. 
     * @param arg The argument which should be forwarded to the callback.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_init_func(ghost_densemat *x, ghost_densemat_srcfunc func, void *arg);

    /**
     * @ingroup denseinit
     * @brief Initializes a densemat from another densemat at a given column and row offset.
     * @param x The densemat.
     * @param y The source.
     * @param roffs The first row to clone.
     * @param coffs The first column to clone.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_init_densemat(ghost_densemat *x, ghost_densemat *y, ghost_lidx roffs, ghost_lidx coffs);

    /**
     * @ingroup denseinit
     * @brief Initializes a densemat from a file.
     * @param x The densemat.
     * @param path Path to the file.
     * @param mpicomm If equal to MPI_COMM_SELF, each process will read from a separate file.
     * Else, a combined file will be read with MPI I/O.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_init_file(ghost_densemat *x, char *path, ghost_mpi_comm mpicomm);

    /**
     * @ingroup denseinit
     * @brief Initializes a complex densemat from two real ones (one holding the real, the other one the imaginary part).
     * @param vec The densemat.
     * @param re The real source densemat.
     * @param im The imaginary source densemat.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_init_real(ghost_densemat *vec, ghost_densemat *re, ghost_densemat *im);

    /**
     * @ingroup denseinit
     * @brief Initializes two real densemats from a complex one.
     * @param re The resulting real densemat holding the real part of the source.
     * @param im The resulting real densemat holding the imaginary part of the source.
     * @param src The complex source densemat.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_init_complex(ghost_densemat *re, ghost_densemat *im, ghost_densemat *src);

    /**
     * @ingroup denseinit
     * @ingroup denseview
     * @brief View plain data which is stored with a given stride 
     * @param x The densemat.
     * @param data Memory location of the data.
     * @param stride Stride of the data.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_view_plain(ghost_densemat *x, void *data, ghost_lidx stride);

    /**
     * @ingroup denseinit
     * @ingroup denseview
     * @brief Create a ghost_densemat as a view of compact data of another ghost_densemat
     * @param x The resulting scattered view.
     * @param src The source densemat with the data to be viewed.
     * @param nr The number of rows of the new densemat.
     * @param roffs The row offset into the source densemat.
     * @param nc The number of columsn of the new densemat.
     * @param coffs The column offset into the source densemat.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_create_and_view_densemat(ghost_densemat **x, ghost_densemat *src, ghost_lidx nr, ghost_lidx roffs, ghost_lidx nc, ghost_lidx coffs);

    /**
     * @ingroup denseinit
     * @ingroup denseview
     * @brief Create a ghost_densemat as a view of arbitrarily scattered data of another ghost_densemat
     * @param x The resulting scattered view.
     * @param src The source densemat with the data to be viewed.
     * @param nr The number of rows of the new densemat.
     * @param ridx The row indices to be viewed.
     * @param nc The number of columsn of the new densemat.
     * @param cidx The column indices to be viewed.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_create_and_view_densemat_scattered(ghost_densemat **x, ghost_densemat *src, ghost_lidx nr, ghost_lidx *ridx, ghost_lidx nc, ghost_lidx *cidx);

    /**
     * @ingroup denseinit
     * @ingroup denseview
     * @brief Create a ghost_densemat as a view of compact columns of another ghost_densemat
     * @param x The resulting scattered view.
     * @param src The source densemat with the data to be viewed.
     * @param nc The number of columsn of the new densemat.
     * @param coffs The column offset into the source densemat.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_create_and_view_densemat_cols(ghost_densemat **x, ghost_densemat *src, ghost_lidx nc, ghost_lidx coffs);

    /**
     * @ingroup denseinit
     * @ingroup denseview
     * @brief Create a ghost_densemat as a view of full but scattered columns of another ghost_densemat
     * @param x The resulting scattered view.
     * @param src The source densemat with the data to be viewed.
     * @param nc The number of columsn of the new densemat.
     * @param cidx The column indices to be viewed.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_create_and_view_densemat_cols_scattered(ghost_densemat **x, ghost_densemat *src, ghost_lidx nc, ghost_lidx *cidx);

    /**
     * @ingroup denseinit
     * @ingroup denseview
     * @brief Create a ghost_densemat as a clone of another ghost_densemat at a column given offset
     * @param x The clone.
     * @param src The source densemat.
     * @param nc The number of columsn of the new densemat.
     * @param coffs The column offset into the source densemat.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_clone(ghost_densemat **x, ghost_densemat *src, ghost_lidx nc, ghost_lidx coffs);

    /**
     * @ingroup stringification
     * @brief Creates a string of the densemat's contents.
     * @param x The densemat.
     * @param str Where to store the string.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     *
     * The string has to be freed by the caller.
     */
    ghost_error ghost_densemat_string(char **str, ghost_densemat *x);

    /**
     * @brief Permute a densemat in a given direction.
     * @param x The densemat.
     * @param ctx The context if a global permutation is present.
     * @param dir The permutation direction.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_permute(ghost_densemat *x, ghost_permutation_direction dir);
    
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
    ghost_error ghost_densemat_download(ghost_densemat *vec);
    
    /**
     * @ingroup gputransfer
     * 
     * @brief Uploads a densemat to a compute device, excluding halo elements. 
     * The densemat must be present on both host and device.
     *
     * @param vec The densemat.
     */
    ghost_error ghost_densemat_upload(ghost_densemat *vec);
    
    /**
     * @brief Reduces the densemats using addition in its map's communicator.
     *
     * @param vec The densemat.
     * @param dest The destination rank or GHOST_ALLREDUCE
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_reduce(ghost_densemat *vec, int dest);

    /**
     * @brief Initialize a halo communication data structure.
     *
     * @param vec The densemat.
     * @param ctx The context in which to communicate.
     * @param comm The halo communication data structure.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_halocomm_init (ghost_densemat *vec, ghost_context *ctx, ghost_densemat_halo_comm *comm);
    /**
     * @brief Start halo communication asynchronously.
     *
     * @param vec The densemat.
     * @param ctx The context in which to communicate.
     * @param comm The halo communication data structure.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_halocomm_start (ghost_densemat *vec, ghost_context *ctx, ghost_densemat_halo_comm *comm);
    /**
     * @brief Finalize halo communication.
     *
     * This includes waiting for the communication to finish and freeing the data in the comm data structure.
     *
     * @param vec The densemat.
     * @param ctx The context in which to communicate.
     * @param comm The halo communication data structure.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error  ghost_densemat_halocomm_finalize (ghost_densemat *vec, ghost_context *ctx, ghost_densemat_halo_comm *comm);
    
    /**
     * @brief Average each entry over all it's halo siblings.
     *
     * This function collects the values for each entry which have been
     * communicated to other processes.
     * Then, it computes the average over all and stores the value.
     * This is used, e.g., in ::ghost_carp().
     *
     * @param vec The densemat. 
     * @param ctx The context in which the densemat lives. 
     */
    ghost_error ghost_densemat_halo_avg (ghost_densemat *vec, ghost_context *ctx);
    
    /**
     * @brief Write a densemat to a file.
     *
     * @param vec The densemat.
     * @param filename The path to the file.
     * @param mpicomm If equal to MPI_COMM_SELF, each process will write a separate file.
     * Else, a combined file will be written with MPI I/O.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_to_file(ghost_densemat *vec, char *filename, ghost_mpi_comm mpicomm);
    
    /**
     * @brief Get a single entry of a ghost_densemat.
     *
     * @param entry Where to store the entry.
     * @param vec The densemat.
     * @param i The row.
     * @param j The column.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_entry(void *entry, ghost_densemat *vec, ghost_lidx i, ghost_lidx j);
    
    /**
     * @brief Sets the densemat to have the same values on all processes.
     *
     * @param vec The densemat.
     * @param comm The communicator in which to synchronize.
     * @param root The process from which to take the values.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_densemat_sync_vals(ghost_densemat *vec, ghost_mpi_comm comm, int root);
    
    /**
     * @brief Compresses a densemat, i.e., make it non-scattered.
     * If the densemat is a view, it will no longer be one afterwards.
     *
     * @param vec The densemat.
     */
    ghost_error ghost_densemat_compress(ghost_densemat *vec);
    /**
     * @brief Collects vec from all MPI ranks and combines them into globalVec.
     * The row permutation (if present) if vec's context is used.
     *
     * @param vec The distributed densemat.
     * @param globvec The global densemat.
     * @param ctx The context.
     */
    ghost_error ghost_densemat_collect(ghost_densemat *vec, ghost_densemat *globvec, ghost_context *ctx);
 
    /**
     * @brief Distributes a global densemat into node-local vetors.
     *
     * @param vec The global densemat.
     * @param localVec The local densemat.
     */
    ghost_error ghost_densemat_distribute(ghost_densemat *vec, ghost_densemat *localVec, ghost_context *ctx);
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
