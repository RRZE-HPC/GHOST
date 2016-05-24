/**
 * @file context.h
 * @brief Types and functions related to GHOST contexts.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_CONTEXT_H
#define GHOST_CONTEXT_H

#include "config.h"
#include "types.h"
#include "error.h"

typedef struct ghost_context ghost_context;


typedef enum
{
    /**
     * @brief A local permutation only contains local indices in the range [0..perm->len-1].
     */
    GHOST_PERMUTATION_LOCAL,
    /**
     * @brief A global permutation contains global indices in the range [0..context->gnrows-1].
     */
    GHOST_PERMUTATION_GLOBAL
}
ghost_permutation_scope_t;

typedef enum
{
    GHOST_PERMUTATION_ORIG2PERM,
    GHOST_PERMUTATION_PERM2ORIG
}
ghost_permutation_direction;

typedef enum
{
    GHOST_PERMUTATION_SYMMETRIC,
    GHOST_PERMUTATION_UNSYMMETRIC
}
ghost_permutation_method;
    
typedef struct
{
    /**
     * @brief Gets an original index and returns the corresponding permuted position.
     *
     * NULL if no permutation applied to the matrix.
     */
    ghost_gidx *perm;
    /**
     * @brief Gets an index in the permuted system and returns the original index.
     *
     * NULL if no permutation applied to the matrix.
     */
    ghost_gidx *invPerm;
    /**
     * @brief Gets an original index and returns the corresponding permuted position of columns.
     *
     * NULL if no permutation applied to the matrix, or if the perm=colPerm.
     */
    ghost_gidx *colPerm;
    /**
     * @brief Gets an index in the permuted system and returns the original index of columns.
     *
     * NULL if no permutation applied to the matrix, or if the invPerm=invColPerm.
     */
    ghost_gidx *colInvPerm;
    /**
    * @brief A flag to indicate whether symmetric or unsymmetric permutation is carried out
    * 	     (internal) Its necessary for destruction of permutations, since we need to know whether 
    * 	     both perm and colPerm point to same array. 
    *
    * GHOST_PERMUTATION_SYMMETRIC - if symmetric (both point to same array)
    * GHOST_PERMUTATION_UNSYMMETRIC - if unsymmetric (points to different array) 
    */
    ghost_permutation_method method;   


    ghost_gidx *cu_perm;

    ghost_permutation_scope_t scope;
    ghost_gidx len;
}
ghost_permutation;

/**
 * @brief This struct holds all possible flags for a context.
 */
typedef enum {
    GHOST_CONTEXT_DEFAULT = 0, 
    /**
     * @brief Distribute work among the ranks by number of nonzeros.
     */
    GHOST_CONTEXT_DIST_NZ = 4,
    /**
     * @brief Distribute work among the ranks by number of rows.
     */
    GHOST_CONTEXT_DIST_ROWS = 8
} ghost_context_flags_t;

/**
 * @brief The GHOST context.
 *
 * The context is relevant in the MPI-parallel case. It holds information about data distribution and communication.
 */
struct ghost_context
{
    /**
     * @brief Row pointers
     * 
     * if the context is distributed by nnz, the row pointers are being read
     * at context creation in order to create the distribution. once the matrix
     * is being created, the row pointers are distributed
     */
    ghost_gidx *rpt;
    /**
     * @brief The global number of non-zeros
     */
    ghost_gidx gnnz;
    /**
     * @brief The global number of rows
     */
    ghost_gidx gnrows;
    /**
     * @brief The global number of columns.
     */
    ghost_gidx gncols;
    /**
     * @brief The context's property flags.
     */
    ghost_context_flags_t flags;
    /**
     * @brief The weight of this context.
     *
     * The weight is used for work distribution between processes.
     *
     * @see ghost_context_create for use of the weight.
     */
    double weight;
    /**
     * @brief The context's MPI communicator.
     */
    ghost_mpi_comm mpicomm;
    /**
     * @brief The matrix' global permutation.
     */
    ghost_permutation *perm_global;
    /**
     * @brief The matrix' local permutation.
     */
    ghost_permutation *perm_local;
    /**
     * @brief Number of remote elements with unique colidx
     */
    ghost_lidx halo_elements; // TODO rename nHaloElements
    /**
     * @brief Number of matrix elements for each rank
     */
    ghost_lidx* lnEnts; // TODO rename nLclEnts
    /**
     * @brief Index of first element into the global matrix for each rank
     */
    ghost_gidx* lfEnt; // TODO rename firstLclEnt
    /**
     * @brief Number of matrix rows for each rank
     */
    ghost_lidx* lnrows; // TODO rename nLclRows
    /**
     * @brief Index of first matrix row for each rank
     */
    ghost_gidx* lfRow; // TODO rename firstLclRow
    /**
     * @brief Number of wishes (= unique RHS elements to get) from each rank
     */
    ghost_lidx * wishes; // TODO rename nWishes
    /**
     * @brief Column idx of wishes from each rank
     */
    ghost_lidx ** wishlist; // TODO rename wishes
    /**
     * @brief Number of dues (= unique RHS elements from myself) to each rank
     */
    ghost_lidx * dues; // TODO rename nDues
    /**
     * @brief Column indices of dues to each rank
     */
    ghost_lidx ** duelist; // TODO rename dues
    /**
     * @brief Column indices of dues to each rank (CUDA)
     */
    ghost_lidx ** cu_duelist; // TODO rename dues
    /**
     * @brief First index to get RHS elements coming from each rank
     */
    ghost_lidx* hput_pos; // TODO rename
    int *duepartners;
    int nduepartners;
    int *wishpartners;
    int nwishpartners;
};


/**
 * @brief Possible sources of a sparse matrix. 
 */
typedef enum {
    /**
     * @brief The matrix comes from a binary CRS file.
     */
    GHOST_SPARSEMAT_SRC_FILE,
    /**
     * @brief The matrix comes from a Matrix Market file.
     */
    GHOST_SPARSEMAT_SRC_MM,
    /**
     * @brief The matrix is generated by a function.
     */
    GHOST_SPARSEMAT_SRC_FUNC,
    /**
     * @brief Empty source.
     */
    GHOST_SPARSEMAT_SRC_NONE
} ghost_sparsemat_src;


#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @brief Create a context. 
     *
     * @param[out] context Where to store the created context.
     * @param[in] gnrows The global number of rows for the context. If gnrows is 0 a valid matrix file path has to be provided in the argument matrixSource from which the number of rows will be read.
     * @param[in] gncols The global number of columns for the context. If gncols is 0 a valid matrix file path has to be provided in the argument matrixSource from which the number of columns will be read.
     * @param[in] flags Flags to the context.
     * @param[in] matrixSource The sparse matrix source.     
     * @param[in] srcType The type of the sparse matrix source.     
     * @param[in] comm The MPI communicator in which the context is present.
     * @param[in] weight This influences the work distribution amon ranks. If set to 0., it is automatically determined. 
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     * 
     * The matrix source can either be one of ghost_sparsemat_src. The concrete type has to be specified in the srcType parameter. 
     * It must not be ::GHOST_SPARSEMAT_SRC_NONE in the following cases:
     * -# If gnrows or gncols are given as less than 1, the source has to be a a pointer to a ghost_sparsemat_src_rowfunc and srcType has to be set to ::GHOST_SPARSEMAT_SRC_FILE. 
     * -# If the flag GHOST_CONTEXT_WORKDIST_NZE is set in the flags, the source may be of any type (except ::GHOST_SPARSEMAT_SRC_NONE of course)
     * 
     * In all other cases, i.e., gnrows and gncols are correctly set and the distribution of matrix rows across ranks should be done by the number of rows, the matrixSource parameter will be ignored. 
     *
     * Each rank will get a portion of work which depends on the distribution scheme as set in the flags multiplied with the given weight divided by the sum of the weights of all ranks in the context's MPI communicator.
     * Example: Rank A is of type ::GHOST_TYPE_CUDA using a GPU with a memory bandwidth of 150 GB/s and rank B is of type ::GHOST_TYPE_WORK using a CPU socket with a memory bandwidth of 50 GB/s. 
     * The work is to be distributed by rows and the matrix contains 8 million rows. 
     * A straight-forward distribution of work would assume a weight of 1.5 for A and 0.5 for B.
     * Thus, A would be assigned 6 million matrix rows and B 2 million.
     * 
     */
    ghost_error ghost_context_create(ghost_context **context, ghost_gidx gnrows, ghost_gidx gncols, ghost_context_flags_t flags, void *matrixSource, ghost_sparsemat_src srcType, ghost_mpi_comm comm, double weight); 
    
    /**
     * @ingroup stringification
     * @brief Create a string containing information on the context.
     *
     * @param str Where to store the string.
     * @param context The context to stringify.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_context_string(char **str, ghost_context *context);
    /**
     * @brief Free the context's resources.
     *
     * @param ctx The context.
     *
     * If the context is NULL it will be ignored.
     */
    void ghost_context_destroy(ghost_context *ctx);

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
    ghost_error ghost_context_comm_init(ghost_context *ctx, ghost_gidx *col_orig, ghost_lidx *col);
    
    /**
     * @brief Get the name of the work distribution scheme.
     *
     * @param flags The context flags.
     *
     * @return A string holding a sensible name of the work distribution scheme.
     */
    char * ghost_context_workdist_string(ghost_context_flags_t flags);
    /**
     * @brief Create a global inverse permutation from a present global permutation
     *
     * @param context A context with a valid global permutation
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
ghost_error ghost_global_perm_inv(ghost_gidx *toPerm, ghost_gidx *fromPerm, ghost_context *context);

#ifdef __cplusplus
} //extern "C"
#endif

#endif
