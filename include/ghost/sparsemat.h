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

/**
 * @brief Callback function to construct a ghost_sparsemat
 *
 * @param[in] row The global row index.
 * @param[out] nnz The number of values in this row.
 * @param[out] val The values in the specified row.
 * @param[out] col The column indices of the given values.
 * @param[inout] arg Additional arguments.
 *
 * @return  
 */
typedef int (*ghost_sparsemat_rowfunc)(ghost_gidx row, ghost_lidx *nnz, ghost_gidx *col, void *val, void *arg);

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
     yes,
     no
}
ghost_kacz_normalize;

typedef struct {
    void *omega;
    ghost_kacz_direction direction;
    ghost_kacz_normalize normalize;
}
ghost_kacz_opts;

/**
 * @brief internal to differentiate between different KACZ sweep methods
 * MC - Multicolored
 * BMC_RB - Block Multicolored with RCM ( condition : nrows/(2*(total_bw+1)) > threads)
 * BMC_one_trans_sweep - Block Multicolored with RCM ( condition : nrows/(total_bw+1) > threads, and transition does not overlap)
 * BMC_two_trans_sweep - Block Multicolored with RCM ( condition : nrows/(total_bw+1) > threads, and transition can overlap)
 */
typedef enum{
      MC,
      BMC_RB,
      BMC_one_sweep,
      BMC_two_sweep,
      BMC
}
ghost_kacz_method;

//TODO zone ptr can be moved here 
typedef struct {
      
      ghost_kacz_method kacz_method;
      ghost_lidx active_threads;
}
ghost_kacz_setting;
    
/**
 * @brief A CUDA SELL-C-sigma matrix.
 */
typedef struct 
{
    /**
     * @brief The values.
     */
    char * val;
    /**
     * @brief The column indices.
     */
    ghost_lidx * col;
    /**
     * @brief The length of each row.
     */
    ghost_lidx * rowLen;
    /**
     * @brief Needed if T>1.
     */
    ghost_lidx * rowLenPadded;
    /**
     * @brief Pointer to start of each chunk.
     */
    ghost_lidx * chunkStart;
    /**
     * @brief The length of each chunk.
     */
    ghost_lidx * chunkLen;
}
ghost_cu_sell;

/**
 * @brief Struct defining a SELL-C-sigma matrix.
 */
typedef struct 
{
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
    ghost_cu_sell *cumat;
}
ghost_sell;

/**
 * @brief Get the SELL data of a general sparsemat.
 *
 * @param mat The sparsemat.
 *
 * @return Pointer to the SELL data.
 */
#define SELL(mat) (mat->sell)

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

}
ghost_kacz_parameters;

extern const ghost_spmv_opts GHOST_SPMV_OPTS_INITIALIZER;
extern const ghost_kacz_opts GHOST_KACZ_OPTS_INITIALIZER;

typedef ghost_error (*ghost_spmv_kernel)(ghost_densemat*, ghost_sparsemat *, ghost_densemat*, ghost_spmv_opts);
typedef ghost_error (*ghost_kacz_kernel)(ghost_densemat*, ghost_sparsemat *, ghost_densemat*, ghost_kacz_opts);
typedef ghost_error (*ghost_kacz_shift_kernel)(ghost_densemat*, ghost_densemat*, ghost_sparsemat *, ghost_densemat*, double, double, ghost_kacz_opts);


/**
 * @brief Flags to be passed to a row-wise matrix assembly function.
 */
typedef enum {
    /**
     * @brief Default behaviour.
     */
    GHOST_SPARSEMAT_ROWFUNC_DEFAULT = 0
} ghost_sparsemat_rowfunc_flags;

/**
 * @brief Defines a rowfunc-based sparsemat source.
 */
typedef struct {
    /**
     * @brief The callback function which assembled the matrix row-wise.
     * @note The function func may be called several times for each row concurrently by multiple threads.
     */
    ghost_sparsemat_rowfunc func;
    /**
     * @brief Maximum row length of the matrix.
     */
    ghost_lidx maxrowlen;
    /**
     * @brief 0 for C, 1 for Fortran-like indexing.
     */
    int base;
    /**
     * @brief Flags to the row function.
     */
    ghost_sparsemat_rowfunc_flags flags;
    void *arg;
} ghost_sparsemat_src_rowfunc;

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

} ghost_sparsemat_flags;


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
    ghost_gidx sortScope;
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
     * @brief Pointer to actual SELL sparse matrix data
     * 
     * This is a relict from times where we had CRS and SELL and may be removed in the future.
     */
    ghost_sell *sell;
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
     * @brief The number of colors from distance-2 coloring.
     */
    ghost_lidx ncolors;
    /**
     * @brief The number of rows with each color (length: ncolors+1).
     */
    ghost_lidx *color_ptr;
     /**
     * @brief The number of total zones (odd+even)
     **/
    ghost_lidx nzones;
    /**
    * @brief Pointer to odd-even (Red-Black coloring) zones of a matrix (length: nzones+1)  
    * Ordering [even_begin_1 odd_begin_1 even_begin_2 odd_begin_2 ..... nrows]
    **/
    ghost_lidx *zone_ptr;
    /**
    * @brief details regarding kacz is stored here
    */ 
    ghost_kacz_setting kacz_setting;
    /**
     * @brief The number of rows.
     */
    ghost_lidx nrows;
    /**
     * @brief The padded number of rows.
     *
     * In the SELL data format, the number of rows is padded to a multiple of C.
     */
    ghost_lidx nrowsPadded;
    /**
     * @brief The number of columns.
     */
    ghost_gidx ncols;
    /**
     * @brief The number of non-zero entries in the matrix.
     */
    ghost_lidx nnz;
    /**
     * @brief The number of stored entries in the matrix.
     *
     * For CRS or SELL-1, this is equal to nnz.
     */
    ghost_lidx nEnts;
    /**
     * @brief The bandwidth of the lower triangular part of the matrix.
     */
    ghost_gidx lowerBandwidth;
    /**
     * @brief The bandwidth of the upper triangular part of the matrix.
     */
    ghost_gidx upperBandwidth;
    /**
     * @brief The bandwidth of the matrix.
     */
    ghost_gidx bandwidth;
    /**
     * @brief The maximum column index in the matrix
     * (Required for example if we permute the (local + remote) part of matrix
     */
    ghost_gidx maxColRange; 
    /**
     * @brief Store the ratio between nrows and bandwidth
     */ 	
    double kaczRatio;
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
    /**
     * Compute a sparse matrix-vector product.
     * This function should not be called directly, see ghost_spmv().
     */
    ghost_spmv_kernel spmv;
    /**
     * Solve using Kacz kernel.
     * This function should not be called directly, see ghost_carp().
     */
    ghost_kacz_kernel kacz;
    /**
     * Solve using Kacz kernel with shift.
     * This function should not be called directly, see ghost_carp_shift().
     */
    ghost_kacz_shift_kernel kacz_shift;
     /**
     * Documented in ghost_sparsemat_string()
     */
    ghost_error (*string) (ghost_sparsemat *mat, char **str, int dense);
    /**
     * @ingroup stringification
     *
     * @brief Return the name of the storage format.
     *
     * @param mat The matrix.
     *
     * @return A string containing the storage format name. 
     */
    const char *  (*formatName) (ghost_sparsemat *mat);
    /**
     * Documented in ghost_sparsemat_init_bin() 
     */
    ghost_error (*fromFile)(ghost_sparsemat *mat, char *path);
    /**
     * Documented in ghost_sparsemat_init_mm()
     */
    ghost_error (*fromMM)(ghost_sparsemat *mat, char *path);
    /**
     * Documented in ghost_sparsemat_init_crs()
     */
    ghost_error (*fromCRS)(ghost_sparsemat *mat, ghost_gidx offs, ghost_lidx n, ghost_gidx *col, void *val, ghost_lidx *rpt);
    /**
     * Documented in ghost_sparsemat_init_rowfunc() 
     */
    ghost_error (*fromRowFunc)(ghost_sparsemat *, 
            ghost_sparsemat_src_rowfunc *src);
    /**
     * @brief Write a matrix to a binary CRS file.
     *
     * @param mat The matrix. 
     * @param path Path of the file.
     */
    ghost_error (*toFile)(ghost_sparsemat *mat, char *path);
    /**
     * @brief Upload the matrix to the CUDA device.
     *
     * @param mat The matrix.
     */
    ghost_error (*upload)(ghost_sparsemat * mat);
    /**
     * @brief Get the entire memory footprint of the matrix.
     *
     * @param mat The matrix.
     *
     * @return The memory footprint of the matrix in bytes or zero if the 
     * matrix is not valid.
     */
    size_t     (*byteSize)(ghost_sparsemat *mat);
    /**
     * @brief Split the matrix into a local and a remote part.
     *
     * @param mat The matrix.
     */
    ghost_error       (*split)(ghost_sparsemat *mat);
};


#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @ingroup types
     *
     * @brief Create a sparse matrix. 
     *
     * @param[out] mat Where to store the matrix
     * @param[in] ctx The context the matrix lives in.
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
    ghost_error ghost_sparsemat_perm_scotch(ghost_sparsemat *mat, void *matrixSource, ghost_sparsemat_src srcType);
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
    ghost_error ghost_sparsemat_perm_sort(ghost_sparsemat *mat, 
            void *matrixSource, ghost_sparsemat_src srcType, ghost_gidx scope);

    ghost_error ghost_sparsemat_perm_spmp(ghost_sparsemat *mat, void *matrixSource, ghost_sparsemat_src srcType);

    /**
     * @brief Create a matrix permutation based on 2-way coloring using ColPack.
     *
     * @param[inout] mat The sparse matrix.
     * @param[in] matrixSource The matrix source. This will be casted depending 
     * on \p srcType.
     * @param[in] srcType Type of the matrix source.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_sparsemat_perm_color(ghost_sparsemat *mat, 
            void *matrixSource, ghost_sparsemat_src srcType);

    ghost_error ghost_sparsemat_blockColor(ghost_sparsemat *mat, void *matrixSource, ghost_sparsemat_src srcType);
 
    ghost_error ghost_sparsemat_perm_zoltan(ghost_sparsemat *mat, void *matrixSource, ghost_sparsemat_src srcType);
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
    ghost_error ghost_carp(ghost_sparsemat *mat, ghost_densemat *x, ghost_densemat *b, void *omega);
    ghost_error ghost_carp_shift(ghost_sparsemat *mat, ghost_densemat *x_real, ghost_densemat *x_imag, ghost_densemat *b, double sigma_r, double sigma_i, void *omega);
    ghost_error checker(ghost_sparsemat *mat);
    ghost_error split_transition(ghost_sparsemat *mat);

    /**
     * @brief Prints the row distribution details of KACZ. 
     *
     * @param[in] mat: The sparsematrix
     */ 
    ghost_error kacz_analyze_print(ghost_sparsemat *mat);
    
    /**
    * @brief Writes a matrix to file 
    *
    *@param A sparse matrix to write
    *@param name Name of file 
   */                
    ghost_error sparsemat_write(ghost_sparsemat *A, char *name);

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

    ghost_error ghost_sparsemat_from_bincrs(ghost_sparsemat *mat, char *path);
    ghost_error ghost_sparsemat_from_mm(ghost_sparsemat *mat, char *path);
    ghost_error ghost_sparsemat_from_crs(ghost_sparsemat *mat, ghost_gidx offs, ghost_lidx n, ghost_gidx *col, void *val, ghost_lidx *rpt);

    ghost_error ghost_sparsemat_perm_global_cols(ghost_gidx *cols, ghost_lidx ncols, ghost_context *context);


    ghost_error ghost_sparsemat_fromfunc_common(ghost_lidx *rl, ghost_lidx *rlp, ghost_lidx *cl, ghost_lidx *clp, ghost_lidx **chunkptr, char **val, ghost_gidx **col, ghost_sparsemat_src_rowfunc *src, ghost_sparsemat *mat, ghost_lidx C, ghost_lidx P);

    ghost_error ghost_sparsemat_fromfunc_common_dummy(ghost_lidx *rl, ghost_lidx *rlp, ghost_lidx *cl, ghost_lidx *clp, ghost_lidx **chunkptr, char **val, ghost_gidx **col, ghost_sparsemat_src_rowfunc *src, ghost_sparsemat *mat, ghost_lidx C, ghost_lidx P);

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


extern const ghost_sparsemat_src_rowfunc 
GHOST_SPARSEMAT_SRC_ROWFUNC_INITIALIZER;


#endif
