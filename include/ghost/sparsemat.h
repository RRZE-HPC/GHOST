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

typedef int (*ghost_sparsemat_fromRowFunc_t)(ghost_gidx, ghost_lidx *, 
        ghost_gidx *, void *, void *);

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
ghost_spmv_traits;

extern const ghost_spmv_traits GHOST_SPMV_TRAITS_INITIALIZER;

typedef ghost_error (*ghost_spmv_kernel)(ghost_densemat*, ghost_sparsemat *, ghost_densemat*, ghost_spmv_traits);

/**
 * @brief Flags to be passed to a row-wise matrix assembly function.
 */
typedef enum {
    /**
     * @brief Default behaviour.
     */
    GHOST_SPARSEMAT_FROMROWFUNC_DEFAULT = 0
} ghost_sparsemat_fromRowFunc_flags;

/**
 * @brief Defines a rowfunc-based sparsemat source.
 */
typedef struct {
    /**
     * @brief The callback function which assembled the matrix row-wise.
     */
    ghost_sparsemat_fromRowFunc_t func;
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
    ghost_sparsemat_fromRowFunc_flags flags;
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
    GHOST_SPARSEMAT_TRANSPOSE_MM = 1024
} ghost_sparsemat_flags_t;


/**
 * @brief Sparse matrix traits.
 */
struct ghost_sparsemat_traits {
    /**
     * @brief Flags to the matrix.
     */
    ghost_sparsemat_flags_t flags;
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
     * @brief Pointer to actual sparse matrix data which may be one of 
     * ghost_crs_t ghost_sell.
     */
    void *data;
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
     * @brief Calculate y = gamma * (A - I*alpha) * x + beta * y.
     *
     * @param mat The matrix A. 
     * @param res The vector y.
     * @param rhs The vector x.
     * @param flags A number of flags to control the operation's behaviour.
     *
     * For detailed information on the flags check the documentation of 
     * ghost_spmv_flags.
     */
    ghost_spmv_kernel spmv;
    /**
     * @brief Destroy the matrix, i.e., free all of its data.
     *
     * @param mat The matrix.
     *
     * Returns if the matrix is NULL.
     */
    void       (*destroy) (ghost_sparsemat *mat);
    /**
     * @ingroup stringification
     *
     * @brief Turns the matrix into a string.
     *
     * @param mat The matrix.
     * @param str Where to store the string.
     * @param dense If 0, only the elements stored in the sparse matrix will 
     * be included.
     * If 1, the matrix will be interpreted as a dense matrix.
     *
     * @return The stringified matrix.
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
     * @brief Create the matrix from a matrix file in GHOST's binary CRS format.
     *
     * @param mat The matrix. 
     * @param path Path to the file.
     */
    ghost_error (*fromFile)(ghost_sparsemat *mat, char *path);
    /**
     * @brief Create the matrix from a Matrix Market file.
     *
     * @param mat The matrix. 
     * @param path Path to the file.
     */
    ghost_error (*fromMM)(ghost_sparsemat *mat, char *path);
    /**
     * @brief Create the matrix from CRS data.
     *
     * @param mat The matrix. 
     * @param offs The global index of this rank's first row.
     * @param n The local number of rows.
     * @param col The (global) column indices.
     * @param val The values.
     * @param rpt The row pointers.
     */
    ghost_error (*fromCRS)(ghost_sparsemat *mat, ghost_gidx offs, ghost_gidx n, ghost_gidx *col, void *val, ghost_lidx *rpt);
    /**
     * @brief Create the matrix from a function which defined the matrix row 
     * by row.
     *
     * @param mat The matrix.
     * @param src The source.
     *
     * The function func may be called several times for each row concurrently 
     * by multiple threads.
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

    /**
     * @brief Perform a forward or backward Kaczmarz sweep on the system Ax=b.
     *
     * @param mat The matrix.
     * @param lhs The vector b.
     * @param rhs The vector x.
     * @param omega The scaling factor omega.
     * @param forward 1 if forward, 0 if backward sweep should be done.
     */
    ghost_error (*kacz) (ghost_sparsemat *mat, ghost_densemat *lhs, 
            ghost_densemat *rhs, void *omega, int forward);
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
    ghost_error ghost_sparsemat_string(char **str, ghost_sparsemat *matrix);
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
     * @brief Common function for matrix creation from a function.
     *
     * This function does work which is independent of the storage format and 
     * it should be called 
     * at the beginning of a sparse matrix' fromFunc function.
     *
     * @param[in] mat The sparse matrix.
     * @param[in] src The matrix source function.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    //ghost_error ghost_sparsemat_fromfunc_common(ghost_sparsemat *mat, 
    //        ghost_sparsemat_src_rowfunc *src);
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
     * @brief Common (independent of the storage formats) destruction of matrix 
     * data.
     *
     * @param[inout] mat The matrix.
     */
    void ghost_sparsemat_destroy_common(ghost_sparsemat *mat);


    ghost_error ghost_sparsemat_from_bincrs(ghost_sparsemat *mat, char *path);
    ghost_error ghost_sparsemat_from_mm(ghost_sparsemat *mat, char *path);
    ghost_error ghost_sparsemat_from_crs(ghost_sparsemat *mat, ghost_gidx offs, ghost_gidx n, ghost_gidx *col, void *val, ghost_lidx *rpt);

    ghost_error ghost_sparsemat_perm_global_cols(ghost_gidx *cols, ghost_lidx ncols, ghost_context *context);

    ghost_error ghost_sparsemat_fromfunc_common(ghost_lidx *rl, ghost_lidx *rlp, ghost_lidx *cl, ghost_lidx *clp, ghost_lidx **chunkptr, char **val, ghost_gidx **col, ghost_sparsemat_src_rowfunc *src, ghost_sparsemat *mat, ghost_lidx C, ghost_lidx P);

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

        

#ifdef __cplusplus
} 
#endif


extern const ghost_sparsemat_src_rowfunc 
GHOST_SPARSEMAT_SRC_ROWFUNC_INITIALIZER;


#endif
