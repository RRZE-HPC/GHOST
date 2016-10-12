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
#include "sparsemat_src.h"

#include <stdarg.h>

#define GHOST_SPARSEMAT_SORT_GLOBAL -1
#define GHOST_SPARSEMAT_SORT_LOCAL -2

/**
 * @brief Helper for sparse matrix row sorting.
 */
typedef struct {
    /**
     * @brief The row.
     */
    ghost_lidx row;
    /**
     * @brief Number of entries in the row.
     */
    ghost_lidx nEntsInRow;
} ghost_sorting_helper;


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
} ghost_sparsemat_symmetry;

typedef struct 
{
    ghost_gidx *col;
    void *val;
    ghost_lidx *rpt;
    size_t dtsize;
    ghost_gidx offs;
} 
ghost_sparsemat_rowfunc_crs_arg;

    
typedef struct ghost_sparsemat_traits ghost_sparsemat_traits;
typedef struct ghost_sparsemat ghost_sparsemat;

typedef struct{
    ghost_spmv_flags flags;
    void *alpha;
    void *beta;
    void *gamma;
    void *delta;
    void *eta;
    void *dot;
    ghost_densemat *z;
} 
ghost_spmv_opts;

typedef enum {
    GHOST_KACZ_DIRECTION_UNDEFINED,
    GHOST_KACZ_DIRECTION_FORWARD,
    GHOST_KACZ_DIRECTION_BACKWARD
}
ghost_kacz_direction;

typedef enum{
     GHOST_KACZ_NORMALIZE_YES,
     GHOST_KACZ_NORMALIZE_NO
}
ghost_kacz_normalize;

typedef enum{
     GHOST_KACZ_MODE_NORMAL,
     GHOST_KACZ_MODE_ECO,
     GHOST_KACZ_MODE_PERFORMANCE
}
ghost_kacz_mode;


typedef struct {
    void *omega;
    void *shift;// shift can be complex
    int num_shifts; //number of shifts provided
    ghost_kacz_direction direction;
    ghost_kacz_mode mode; 
    int best_block_size; //it provides the best block size if performance mode is set 
    ghost_kacz_normalize normalize;
    void *scale; //scaled values if normalize is on
    bool initialized; //internal flag
}
ghost_kacz_opts;

typedef struct {
    void *omega;
    void *shift;// shift can be complex
    int num_shifts; //number of shifts provided
    ghost_kacz_mode mode; 
    int best_block_size; //it provides the best block size if performance mode is set 
    ghost_kacz_normalize normalize;
    void *scale; //scaled values if normalize is on
    bool initialized; //internal flag
}
ghost_carp_opts; //no direction since forward followed by backward done


/**
 * @brief Create only a single chunk, i.e., use the ELLPACK storage format.
 */
#define GHOST_SELL_CHUNKHEIGHT_ELLPACK 0
/**
 * @brief A chunkheight should automatically be determined.
 */
#define GHOST_SELL_CHUNKHEIGHT_AUTO -1

/**
 * @brief The parameters to identify a SELL SpMV kernel.
 *
 * On kernel execution, GHOST will try to find an auto-generated kernel which
 * matches all of these parameters.
 */
typedef struct 
{
    /**
     * @brief The data access alignment.
     */
    ghost_alignment alignment;
    /**
     * @brief The implementation.
     */
    ghost_implementation impl;
    /**
     * @brief The matrix data type.
     */
    ghost_datatype mdt;
    /**
     * @brief The densemat data type.
     */
    ghost_datatype vdt;
    /**
     * @brief The densemat width.
     */
    int blocksz;
    /**
     * @brief The SELL matrix chunk height.
     */
    int chunkheight;
    /**
     * @brief The densemat storage order.
     */
    ghost_densemat_storage storage;

}
ghost_sellspmv_parameters;



/**
 * @brief The parameters to identify a SELL Kaczmarz kernel.
 *
 * On kernel execution, GHOST will try to find an auto-generated kernel which
 * matches all of these parameters.
 */
typedef struct 
{
    /**
     * @brief The data access alignment.
     */
    ghost_alignment alignment;
    /**
     * @brief The implementation.
     */
    ghost_implementation impl;
    /**
     * @brief The matrix data type.
     */
    ghost_datatype mdt;
    /**
     * @brief The densemat data type.
     */
    ghost_datatype vdt;
    /**
     * @brief The densemat width.
     */
    int blocksz;
    /**
     * @brief The SELL matrix chunk height.
     */
    int chunkheight;
    /**
     * @brief The densemat storage order.
     */
    ghost_densemat_storage storage;

    ghost_kacz_method method;

    /**
     * @brief The number of shifts (zero if no shift should be applied).
     */
    int nshifts;

}
ghost_kacz_parameters;

extern const ghost_spmv_opts GHOST_SPMV_OPTS_INITIALIZER;
extern const ghost_kacz_opts GHOST_KACZ_OPTS_INITIALIZER;
extern const ghost_carp_opts GHOST_CARP_OPTS_INITIALIZER;


typedef ghost_error (*ghost_spmv_kernel)(ghost_densemat*, ghost_sparsemat *, ghost_densemat*, ghost_spmv_opts);
typedef ghost_error (*ghost_kacz_kernel)(ghost_densemat*, ghost_sparsemat *, ghost_densemat*, ghost_kacz_opts);
typedef ghost_error (*ghost_kacz_shift_kernel)(ghost_densemat*, ghost_densemat*, ghost_sparsemat *, ghost_densemat*, double, double, ghost_kacz_opts);


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
     * @brief If the matrix rows have been re-ordered, do _NOT_ permute the 
     * column indices accordingly.
     */
    GHOST_SPARSEMAT_NOT_PERMUTE_COLS = 4,
    /**
     * @brief The matrix rows should be re-ordered in a certain way (defined 
     * in the traits). 
     */
    GHOST_SPARSEMAT_PERMUTE       = 8,
    /**
     * @brief Do _not_ sort the matrix cols wrt. memory location.
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
    /**
     * @brief Save the un-compressed original columns of a distributed matrix.
     */
    GHOST_SPARSEMAT_SAVE_ORIG_COLS = 256,
    /**
     * @brief Create a matrix permutation reflecting a distance-2-coloring.
     */
    GHOST_SPARSEMAT_COLOR = 512,
    /**
     * @brief If the matrix comes from a matrix market file, transpose it on read-in.
     * If this is implemented for other rowfuncs, the _MM may get removed in the future.
     */
    GHOST_SPARSEMAT_TRANSPOSE_MM = 1024,
    /**
     * @brief Re-order the matrix globally using Zoltan hypergraph partitioning.
     */
    GHOST_SPARSEMAT_ZOLTAN = 2048,
    /**
     * @brief Re-order the local part of the matrix using parallel RCM re-ordering.
     */
    GHOST_SPARSEMAT_RCM = 4096,
    /**
    * @brief Re-order the local part of the matrix using a block coloring.
    */
    GHOST_SPARSEMAT_BLOCKCOLOR = 8192,
    /**
    * @brief SETS the sparsematrix permutation as needed by the KACZ solver
    * depending on the bandwidth of the matrix
    */
    GHOST_SOLVER_KACZ = 16384,
    /**
    * @brief Sort matrix rows according to their length (SELL-C-Sigma sorting)
    */
    GHOST_SPARSEMAT_SORT_ROWS = 32768,

} ghost_sparsemat_flags;

/**
 * @brief Combination of flags which apply any permutation to a ::ghost_sparsemat 
 */
#define GHOST_SPARSEMAT_PERM_ANY (GHOST_SPARSEMAT_PERM_ANY_LOCAL|GHOST_SPARSEMAT_PERM_ANY_GLOBAL)
#define GHOST_SPARSEMAT_PERM_ANY_LOCAL (GHOST_SPARSEMAT_COLOR|GHOST_SPARSEMAT_RCM|GHOST_SPARSEMAT_BLOCKCOLOR|GHOST_SPARSEMAT_SORT_ROWS|GHOST_SOLVER_KACZ)
#define GHOST_SPARSEMAT_PERM_ANY_GLOBAL (GHOST_SPARSEMAT_SCOTCHIFY|GHOST_SPARSEMAT_ZOLTAN)

#ifdef __cplusplus
inline ghost_sparsemat_flags operator|(const ghost_sparsemat_flags &a,
        const ghost_sparsemat_flags &b)
{
    return static_cast<ghost_sparsemat_flags>(
            static_cast<int>(a) | static_cast<int>(b));
}

inline ghost_sparsemat_flags operator&(const ghost_sparsemat_flags &a,
        const ghost_sparsemat_flags &b)
{
    return static_cast<ghost_sparsemat_flags>(
            static_cast<int>(a) & static_cast<int>(b));
}
#endif


/**
 * @brief Sparse matrix traits.
 */
struct ghost_sparsemat_traits {
    /**
     * @brief Flags to the matrix.
     */
    ghost_sparsemat_flags flags;
    /**
     * @brief The matrix symmetry.
     */
    ghost_sparsemat_symmetry symmetry;
    /**
     * @brief The chunk height.
     */
    int C;
    /**
     * @brief Number of threads per row.
     */
    int T;
    /**
     * @brief The re-ordering strategy to be passed to SCOTCH.
     */
    const char * scotchStrat;
    /**
     * @brief The sorting scope if sorting should be applied.
     */
    ghost_lidx sortScope;
    /**
     * @brief The data type.
     */
    ghost_datatype datatype;
    /**
     * @brief Optimal width of block vectors multiplied with this matrix.
     *
     * Set to zero if not applicable. 
     */
    ghost_lidx opt_blockvec_width;
};

#ifdef GHOST_HAVE_MPI
#define GHOST_SCOTCH_STRAT_DEFAULT "n{ole=q{strat=g},ose=q{strat=g},osq=g,sep=m}"
#else
#define GHOST_SCOTCH_STRAT_DEFAULT "g"
#endif

/**
 * @brief Initialize sparse matrix traits with default values.
 */
extern const ghost_sparsemat_traits GHOST_SPARSEMAT_TRAITS_INITIALIZER;

/**
 * @ingroup types
 *
 * @brief A sparse matrix.
 * 
 * The according functions act locally and are accessed via function pointers. 
 * The first argument of
 * each member function always has to be a pointer to the vector itself.
 */
struct ghost_sparsemat
{
    /**
     * @brief The matrix' traits.
     */
    ghost_sparsemat_traits traits;
    /**
     * @brief The local and remote part's traits.
     */
    ghost_sparsemat_traits splittraits[2];
    /**
     * @brief The local part of the matrix (if distributed).
     */
    ghost_sparsemat *localPart;
    /**
     * @brief The remote part (i.e., the part which has remote column indices) 
     * of the matrix (if distributed).
     */
    ghost_sparsemat *remotePart;
    /**
     * @brief The context of the matrix (if distributed).
     */
    ghost_context *context;
    /**
     * @brief The matrix' name.
     */
    char *name;
    /**
     * @brief The values.
     */
    char *val;
    /**
     * @brief The column indices.
     */
    ghost_lidx *col;
    /**
     * @brief Pointer to start of each chunk.
     */
    ghost_lidx *chunkStart;
    /**
     * @brief Minimal row length in a chunk.
     */
    ghost_lidx *chunkMin;
    /**
     * @brief The length of each chunk.
     */
    ghost_lidx *chunkLen;
    /**
     * @brief Needed if T>1.
     */
    ghost_lidx *chunkLenPadded;
    /**
     * @brief Length of each row.
     *
     * Especially useful in SELL-1 kernels.
     */
    ghost_lidx *rowLen;
    /**
     * @brief Needed if T>1.
     */
    ghost_lidx *rowLenPadded; 
    /**
     * @brief The CUDA matrix.
     */
    //ghost_cu_sell *cumat;
    /**
     * @brief The values.
     */
    char * cu_val;
    /**
     * @brief The column indices.
     */
    ghost_lidx * cu_col;
    /**
     * @brief The length of each row.
     */
    ghost_lidx * cu_rowLen;
    /**
     * @brief Needed if T>1.
     */
    ghost_lidx * cu_rowLenPadded;
    /**
     * @brief Pointer to start of each chunk.
     */
    ghost_lidx * cu_chunkStart;
    /**
     * @brief The length of each chunk.
     */
    ghost_lidx * cu_chunkLen;
    /**
     * @brief Size (in bytes) of one matrix element.
     */
    size_t elSize;
    /**
     * @brief The original column indices of the matrix.
     *
     * Once a matrix gets distributed, the column indices of the matrix are 
     * being compressed.
     * That means, the local part of the matrix comes first and the remote 
     * part's column indices (compresed) thereafter.
     * If the flag ::GHOST_SPARSEMAT_SAVE_ORIG_COLS is set, the original 
     * (un-compressed) column indices are stored in this variable.
     * This is only necessary if, e.g., a distributed matrix should be written 
     * out to a file. 
     */
    ghost_gidx *col_orig;
    /**
     * @brief The number of rows.
     */
    //ghost_lidx nrows;
    /**
     * @brief The padded number of rows.
     *
     * In the SELL data format, the number of rows is padded to a multiple of C.
     */
    //ghost_lidx nrowsPadded;
    /**
     * @brief The number of columns.
     */
    //ghost_gidx ncols;
    /**
     * @brief The number of non-zero entries in the matrix.
     */
    //ghost_lidx nnz;
    /**
     * @brief The number of stored entries in the matrix.
     *
     * For CRS or SELL-1, this is equal to nnz.
     */
    ghost_lidx nEnts;
    /**
     * @brief The average width of the rows wrt. the diagonal.
     */
    double avgRowBand;
    /**
     * @brief The average of the average width of the rows wrt. the diagonal.
     */
    double avgAvgRowBand;
    /**
     * @brief A smart value quantifying the matrix bandwidth. Currently the 
     * 90-percentile of the 90-percentile of all widths.
     */
    double smartRowBand;
    /**
     * @brief The maximum row length.
     * TODO: This sould be a ghost_lidx, right?
     */
    ghost_gidx maxRowLen;
    /**
     * @brief The number of rows with length maxRowLen.
     */
    ghost_gidx nMaxRows;
    /**
     * @brief Row length variance
     */
    double variance;
    /**
     * @brief Row length standard deviation
     */
    double deviation;
    /**
     * @brief Row length coefficient of variation
     */
    double cv;
    /**
     * @brief Array of length (2*nrows-1) with nzDist[i] = number nonzeros 
     * with distance i from diagonal
     */
    ghost_gidx *nzDist;
    ghost_lidx nchunks;
};

#define SPM_NROWS(mat) mat->context->row_map->dim
#define SPM_NNZ(mat) mat->context->nnz
#define SPM_NCOLS(mat) mat->context->col_map->dim
#define SPM_GNCOLS(mat) mat->context->col_map->gdim
#define SPM_NROWSPAD(mat) mat->context->row_map->dimpad
#define SPM_NCHUNKS(mat) (mat->nchunks)


#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @ingroup types
     *
     * @brief Create a sparse matrix. 
     *
     * @param[out] mat Where to store the matrix
     * @param[in] ctx An existing context or ::GHOST_CONTEXT_INITIALIZER.
     * @param[in] traits The matrix traits. They can be specified for the full 
     * matrix, the local and the remote part.
     * @param[in] nTraits The number of traits. 
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     *
     * @note No memory will be allocated in this function. Before any operation with the densemat is done,
     * an initialization function (see @ref sparseinit) has to be called with the sparsemat.
     * As such, @ref ghost_sparsemat_create does not check the matrix datatype stored in
     * @c traits (@c GHOST_DT_NONE by default); this is done by the matrix initialization functions
     * to allow for the datatype to be determined from an input file.
     */
    ghost_error ghost_sparsemat_create(ghost_sparsemat **mat, 
            ghost_context *ctx, ghost_sparsemat_traits *traits, 
            int nTraits);
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
    ghost_error ghost_sparsemat_info_string(char **str, ghost_sparsemat *matrix);
    /**
     * @brief Obtain the global number of nonzero elements of a sparse matrix.
     *
     * @param[out] nnz Where to store the result.
     * @param[in] mat The sparse matrix.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_sparsemat_nnz(ghost_gidx *nnz, 
            ghost_sparsemat *mat);
    /**
     * @brief Obtain the global number of rows of a sparse matrix.
     
     * @param[out] nrows Where to store the result.
     * @param[in] mat The sparse matrix.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_sparsemat_nrows(ghost_gidx *nrows, 
            ghost_sparsemat *mat);
    /**
     * @ingroup stringification
     *
     * @brief Convert the matrix' symmetry information to a string.
     *
     * @param[in] symmetry The symmetry information.
     *
     * @return A string holding the symmetry information.
     */
    const char * ghost_sparsemat_symmetry_string(ghost_sparsemat_symmetry symmetry);
    /**
     * @brief Check if the symmetry information of a sparse matrix is valid.
     *
     * @param[in] symmetry The symmetry information.
     *
     * @return True if it is valid, false otherwise.
     */
    bool ghost_sparsemat_symmetry_valid(ghost_sparsemat_symmetry symmetry);
    /**
     * @brief Create a matrix permutation based on (PT-)SCOTCH
     *
     * @param[inout] mat The sparse matrix.
     * @param[in] matrixSource The matrix source. This will be casted depending 
     * on \p srcType.
     * @param[in] srcType Type of the matrix source.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_sparsemat_perm_scotch(ghost_context *ctx, ghost_sparsemat *mat);
    /**
     * @brief Create a matrix permutation based on row length sorting within a 
     * given scope.
     *
     * @param[inout] mat The sparse matrix.
     * @param[in] matrixSource The matrix source. This will be casted depending 
     * on \p srcType.
     * @param[in] srcType Type of the matrix source.
     * @param[in] scope The sorting scope.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_sparsemat_perm_sort(ghost_context *ctx, ghost_sparsemat *mat, ghost_lidx scope);

    ghost_error ghost_sparsemat_perm_spmp(ghost_context *ctx, ghost_sparsemat *mat);

    /**
     * @brief Create a matrix permutation based on 2-way coloring using ColPack.
     *
     * @param[out] ctx The context in which to store the permutations and color information.
     * @param[in] ctx The unpermuted SELL-1-1 source sparse matrix.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_sparsemat_perm_color(ghost_context *ctx, ghost_sparsemat *mat);

    ghost_error ghost_sparsemat_blockColor(ghost_context *ctx, ghost_sparsemat *mat);
 
    ghost_error ghost_sparsemat_perm_zoltan(ghost_context *ctx, ghost_sparsemat *mat);
    /**
     * @brief Sort the entries in a given row physically to have increasing 
     * column indices.
     *
     * @param[inout] col The column indices of the row.
     * @param[inout] val The values of the row.
     * @param[in] valSize The size of one entry.
     * @param[in] rowlen The length of the row.
     * @param[in] stride The stride between successive elements in the row (1 
     * for CRS, C for SELL-C).
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_sparsemat_sortrow(ghost_gidx *col, char *val, 
            size_t valSize, ghost_lidx rowlen, ghost_lidx stride);
    /**
     * @brief Common function for matrix creation from a file.
     *
     * This function does work which is independent of the storage format and 
     * it should be called 
     * at the beginning of a sparse matrix' fromFile function.
     * This function also reads the row pointer from the file and stores it 
     * into the parameter rpt.
     *
     * @param[in] mat The sparse matrix.
     * @param[in] matrixPath The matrix file path.
     * @param[out] rpt Where to store the row pointer information which may be 
     * needed afterwards.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_sparsemat_fromfile_common(ghost_sparsemat *mat, 
            char *matrixPath, ghost_lidx **rpt) ;
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
    ghost_error ghost_sparsematofile_header(ghost_sparsemat *mat,
            char *path);
    /**
     * @brief Store matrix information like bandwidth and nonzero distribution 
     * for a given matrix row.
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
    ghost_error ghost_sparsemat_registerrow(ghost_sparsemat *mat, 
            ghost_gidx row, ghost_gidx *col, ghost_lidx rowlen, 
            ghost_lidx stride);
    /**
     * @brief Finalize the storing of matrix information like bandwidth and 
     * nonzero distribution.
     *
     * This function should be after matrix creation.
     * It finalizes the processing of information obtained in 
     * ghost_sparsemat_registerrow.
     *
     * @param[out] mat The matrix.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_sparsemat_registerrow_finalize(ghost_sparsemat *mat);

    /**
     * @ingroup teardown
     * @brief Destroy a sparsemat and free all memory.
     * @param[inout] mat The matrix.
     */
    void ghost_sparsemat_destroy(ghost_sparsemat *mat);
    /**
     * @brief Select and call the right SELL SpMV kernel. 
     *
     * @param mat The matrix.
     * @param lhs The result densemat.
     * @param rhs The input densemat.
     * @param traits The SpMV traits.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_sell_spmv_selector(ghost_densemat *lhs, 
            ghost_sparsemat *mat, 
            ghost_densemat *rhs, 
            ghost_spmv_opts traits);
    
    /**
     * @brief Select and call the right SELL stringification function.
     *
     * @param mat The matrix.
     * @param str Where to store the string.
     * @param dense Print in a dense or sparse manner.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_sell_stringify_selector(ghost_sparsemat *mat, 
            char **str, int dense);
    /**
     * @ingroup stringification
     * @brief Creates a string of the sparsemat's contents.
     * @param mat The matrix.
     * @param str Where to store the string.
     * @param dense If 0, only the elements stored in the sparse matrix will 
     * be included. If 1, the matrix will be interpreted as a dense matrix.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     *
     * The string has to be freed by the caller.
     */
    ghost_error ghost_sparsemat_string(char **str, ghost_sparsemat *mat, int dense);
    /**
     * @ingroup sparseinit
     * @brief Initializes a sparsemat from a row-based callback function.
     * @param mat The matrix.
     * @param src The source.
     * @param mpicomm The MPI communicator in which to create this sparsemat.
     * @param weight The weight of each rank in the given MPI communicator.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     * 
     * Requires the matrix to have a valid and compatible datatype.
     */
    ghost_error ghost_sparsemat_init_rowfunc(ghost_sparsemat *mat, ghost_sparsemat_src_rowfunc *src, ghost_mpi_comm mpicomm, double weight);

    /**
     * @ingroup sparseinit
     * @brief Initializes a sparsemat from a binary CRS file.
     * @param mat The matrix.
     * @param path The source file.
     * @param mpicomm The MPI communicator in which to create this sparsemat.
     * @param weight The weight of each rank in the given MPI communicator.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     * 
     * Allows the matrix' datatype to be @c GHOST_DT_NONE. In this case the
     * datatype for the matrix is read from file. Otherwise the matrix 
     * datatype has to be valid and compatible.
     */
    ghost_error ghost_sparsemat_init_bin(ghost_sparsemat *mat, char *path, ghost_mpi_comm mpicomm, double weight);

    /**
     * @ingroup sparseinit
     * @brief Initializes a sparsemat from a Matrix Market file.
     * @param mat The matrix.
     * @param path The source file.
     * @param mpicomm The MPI communicator in which to create this sparsemat.
     * @param weight The weight of each rank in the given MPI communicator.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     * 
     * Allows the matrix' datatype to be @c GHOST_DT_NONE or one of the
     * incomplete datatypes @c GHOST_DT_FLOAT and @c GHOST_DT_DOUBLE. 
     * If the matrix' datatype on entry is @c GHOST_DT_FLOAT or @c GHOST_DT_DOUBLE,
     * the file will be interpreted either in single or double precision, 
     * respectively. In this case, the datatype will be completed with
     * @c GHOST_DT_REAL or @c GHOST_DT_COMPLEX as specified in the input file.
     * If the matrix' datatype on entry is @c GHOST_DT_NONE, @c GHOST_DT_DOUBLE
     * is assumed.
     * Otherwise the matrix datatype has to be valid and compatible.
     */
    ghost_error ghost_sparsemat_init_mm(ghost_sparsemat *mat, char *path, ghost_mpi_comm mpicomm, double weight);

    /**
     * @ingroup sparseinit
     * @brief Initializes a sparsemat from local CRS data.
     * @param mat The matrix.
     * @param offs The global index of this rank's first row.
     * @param n The local number of rows.
     * @param col The (global) column indices.
     * @param val The values.
     * @param rpt The row pointers.
     * @param mpicomm The MPI communicator in which to create this sparsemat.
     * @param weight The weight of each rank in the given MPI communicator.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     * 
     * Requires the matrix to have a valid and compatible datatype.
     */
    ghost_error ghost_sparsemat_init_crs(ghost_sparsemat *mat, ghost_gidx offs, ghost_lidx n, ghost_gidx *col, void *val, ghost_lidx *rpt, ghost_mpi_comm mpicomm, double weight);
    
    /**
     * @brief Write a matrix to a binary CRS file.
     *
     * @param mat The matrix. 
     * @param path Path of the file.
     */
    ghost_error ghost_sparsemat_to_bin(ghost_sparsemat *mat, char *path);
    
    /**
     * @brief Get the entire memory footprint of the matrix.
     *
     * @param mat The matrix.
     *
     * @return The memory footprint of the matrix in bytes or zero if the 
     * matrix is not valid.
     */
    size_t ghost_sparsemat_bytesize(ghost_sparsemat *mat);
    
    /**
     * @brief Select and call the right CUDA SELL SpMV kernel. 
     *
     * @param mat The matrix.
     * @param lhs The result densemat.
     * @param rhs The input densemat.
     * @param traits The SpMV traits.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_cu_sell_spmv_selector(ghost_densemat *lhs, 
            ghost_sparsemat *mat, 
            ghost_densemat *rhs, 
            ghost_spmv_opts traits);

    ghost_error ghost_cu_sell1_spmv_selector(ghost_densemat * lhs_in, 
            ghost_sparsemat *mat, ghost_densemat * rhs_in, ghost_spmv_opts traits);

   /**
     * @brief Select and call the right SELL KACZ kernel. 
     *
     * @param mat The matrix.
     * @param x The result densemat.
     * @param b The input densemat.
     * @param opts The kacz Options.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
     ghost_error ghost_sell_kacz_selector(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts);

   /**
     * @brief Select and call the right SELL KACZ kernel with complex shifts. 
     *
     * @param mat The matrix.
     * @param x_real (densemat) The real part of result.
     * @param x_imag (densemat) The imaginary part of result.
     * @param b The input densemat.
     * @param sigma_r The real part of shift
     * @param sigma_i The imaginary part of shift
     * @param opts The kacz Options.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
     ghost_error ghost_sell_kacz_shift_selector(ghost_densemat *x_real, ghost_densemat *x_imag, ghost_sparsemat *mat, ghost_densemat *b, double sigma_r, double sigma_i, ghost_kacz_opts opts);

    /**
     * @brief Perform a Kaczmarz sweep with the SELL matrix. 
     *
     * @param x Output densemat.
     * @param mat The matrix.
     * @param b Input densemat.
     * @param opts Options.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_kacz(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts); 
    ghost_error ghost_kacz_mc(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts);
    ghost_error ghost_kacz_rb(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts);
    ghost_error ghost_kacz_bmc(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts);
    ghost_error ghost_kacz_rb_with_shift(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *b, double *shift_r,  ghost_kacz_opts opts);
    ghost_error ghost_carp(ghost_sparsemat *mat, ghost_densemat *x, ghost_densemat *b, ghost_carp_opts opts);
    ghost_error checker(ghost_sparsemat *mat);
    ghost_error split_transition(ghost_sparsemat *mat);
 
    /**
     * @brief Initialize CARP 
     *
     * @param[in] mat: The sparsematrix
     * @param[in] b: The rhs
     * @param opts Options.
     */ 
    ghost_error ghost_carp_init(ghost_sparsemat *mat, ghost_densemat *b, ghost_carp_opts *opts); 
     /**
     * @brief Finds optimum parameters for CARP 
     *
     * @param[in] mat: The sparsematrix
     * @param opts Options.
     */ 
    ghost_error ghost_carp_perf_init(ghost_sparsemat *mat, ghost_carp_opts *opts);
    /**
     * @brief Prints the row distribution details of KACZ. 
     *
     * @param[in] mat: The sparsematrix
     */ 
    ghost_error kacz_analyze_print(ghost_sparsemat *mat);
    
    ghost_error ghost_sparsemat_to_mm(char *path, ghost_sparsemat *mat);

    /**
     * @brief Assemble communication information in the given context.
     * @param[inout] ctx The context.
     * @param[in] col_orig The original column indices of the sparse matrix which is bound to the context.
     * @param[out] col The compressed column indices of the sparse matrix which is bound to the context.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     * 
     * The following fields of ghost_context are being filled in this function:
     * wishes, wishlist, dues, duelist, hput_pos, wishpartners, nwishpartners, duepartners, nduepartners.
     * Additionally, the columns in col_orig are being compressed and stored in col.
     */
    ghost_error ghost_context_comm_init(ghost_context *ctx, ghost_gidx *col_orig, ghost_sparsemat *mat, ghost_lidx *col);

    ghost_error ghost_sparsemat_perm_global_cols(ghost_gidx *cols, ghost_lidx ncols, ghost_context *context);

 static inline int ghost_sparsemat_rowfunc_crs(ghost_gidx row, ghost_lidx *rowlen, ghost_gidx *col, void *val, void *crsdata)
{
    ghost_gidx *crscol = ((ghost_sparsemat_rowfunc_crs_arg *)crsdata)->col;
    ghost_lidx *crsrpt = ((ghost_sparsemat_rowfunc_crs_arg *)crsdata)->rpt;
    char *crsval = (char *)((ghost_sparsemat_rowfunc_crs_arg *)crsdata)->val;
    size_t dtsize = ((ghost_sparsemat_rowfunc_crs_arg *)crsdata)->dtsize;
    ghost_gidx offs = ((ghost_sparsemat_rowfunc_crs_arg *)crsdata)->offs;

    *rowlen = crsrpt[row-offs+1]-crsrpt[row-offs];
    memcpy(col,&crscol[crsrpt[row-offs]],*rowlen * sizeof(ghost_gidx));
    memcpy(val,&crsval[dtsize*crsrpt[row-offs]],*rowlen * dtsize);

    return 0;
}

//To calculate Bandwidth        
ghost_error calculate_bw(ghost_sparsemat *mat, void *matrixSource, ghost_sparsemat_src srcType);
ghost_error set_kacz_ratio(ghost_sparsemat *mat, void *matrixSource, ghost_sparsemat_src srcType); 

#ifdef __cplusplus
} 
#endif


#endif
