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

typedef struct ghost_context_t ghost_context_t;

/**
 * @brief This struct holds all possible flags for a context.
 */
typedef enum {
    GHOST_CONTEXT_DEFAULT = 0, 
    /**
     * @brief The context will hold the same data on each process.
     */
    GHOST_CONTEXT_REDUNDANT = 1, 
    /**
     * @brief The context will be distributed across the ranks.
     */
    GHOST_CONTEXT_DISTRIBUTED = 2,
    /**
     * @brief Distribute work among the ranks by number of nonzeros.
     */
    GHOST_CONTEXT_DIST_NZ = 4,
    /**
     * @brief Distribute work among the ranks by number of rows.
     */
    GHOST_CONTEXT_DIST_ROWS = 8,
    /**
     * @brief Obtain matrix row information from a provided matrix file.
     */
    GHOST_CONTEXT_ROWS_FROM_FILE = 16,
    /**
     * @brief Obtain matrix row information from a provided matrix function.
     */
    GHOST_CONTEXT_ROWS_FROM_FUNC = 32
} ghost_context_flags_t;

struct ghost_context_t
{
    //ghost_spmvsolver_t *spmvsolvers;

    // if the context is distributed by nnz, the row pointers are being read
    // at context creation in order to create the distribution. once the matrix
    // is being created, the row pointers are distributed
    ghost_midx_t *rpt;

   // ghost_comm_t *communicator;
    ghost_midx_t gnrows;
    ghost_midx_t gncols;
    int flags;
    double weight;

    ghost_midx_t *rowPerm;    // may be NULL
    ghost_midx_t *invRowPerm; // may be NULL

    ghost_mpi_comm_t mpicomm;
    
    /**
     * @brief Number of remote elements with unique colidx
     */
    ghost_midx_t halo_elements; // TODO rename nHaloElements
    /**
     * @brief Number of matrix elements for each rank
     */
    ghost_mnnz_t* lnEnts; // TODO rename nLclEnts
    /**
     * @brief Index of first element into the global matrix for each rank
     */
    ghost_mnnz_t* lfEnt; // TODO rename firstLclEnt
    /**
     * @brief Number of matrix rows for each rank
     */
    ghost_midx_t* lnrows; // TODO rename nLclRows
    /**
     * @brief Index of first matrix row for each rank
     */
    ghost_midx_t* lfRow; // TODO rename firstLclRow
    /**
     * @brief Number of wishes (= unique RHS elements to get) from each rank
     */
    ghost_mnnz_t * wishes; // TODO rename nWishes
    /**
     * @brief Column idx of wishes from each rank
     */
    ghost_midx_t ** wishlist; // TODO rename wishes
    /**
     * @brief Number of dues (= unique RHS elements from myself) to each rank
     */
    ghost_mnnz_t * dues; // TODO rename nDues
    /**
     * @brief Column indices of dues to each rank
     */
    ghost_midx_t ** duelist; // TODO rename dues
    /**
     * @brief First index to get RHS elements coming from each rank
     */
    ghost_midx_t* hput_pos; // TODO rename
};



#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @brief Create a context. 
     *
     * @param[out] ctx Where to store the created context.
     * @param[in] gnrows The global number of rows for the context. If gnrows<1 a valid matrix file path has to be provided in the argument matrixSource from which the number of rows will be read.
     * @param[in] gncols The global number of columns for the context. If gncols<1 a valid matrix file path has to be provided in the argument matrixSource from which the number of columns will be read.
     * @param[in] flags Flags to the context.
     * @param[in] matrixSource The matrix source.     
     * @param[in] comm The MPI communicator in which the context is present.
     * @param[in] weight This influences the work distribution amon ranks. 
     *
     * @return 
     * 
     * The matrix source can either be a matrix file or a function which creates the matrix. 
     * It has to be provided in the following cases:
     * -# If gnrows or gncols are given as less than 1, the source has to be a valid matrix file. 
     * -# If the flag GHOST_CONTEXT_WORKDIST_NZE is set in the flags, the source has to be valid file or function with the according flag GHOST_CONTEXT_ROWS_FROM_FILE or GHOST_CONTEXT_ROWS_FROM_FUNC set in flags.
     * 
     * In all other cases, i.e., gnrows and gncols are correctly set and the distribution of matrix rows across ranks should be done by the number of rows, this argument will be ignored. 
     *
     * Each rank will get a portion of work which depends on the distribution scheme as set in the flags multiplied with the given weight divided by the sum of the weights of all ranks in the context's MPI communicator.
     * Example: Rank A is of type GHOST_TYPE_CUDAMGMT using a GPU with a memory bandwidth of 150 GB/s and rank B is of type GHOST_TYPE_COMPUTE using a CPU socket with a memory bandwidth of 50 GB/s. 
     * The work is to be distributed by rows and the matrix contains 8 million rows. 
     * A straight-forward distribution of work would assume a weight of 1.5 for A and 0.5 for B.
     * Thus, A would be assigned 6 million matrix rows and B 2 million.
     * 
     */
    ghost_error_t ghost_createContext(ghost_context_t **ctx, ghost_midx_t gnrows, ghost_midx_t gncols, ghost_context_flags_t flags, void *matrixSource, ghost_mpi_comm_t comm, double weight); 
    
    ghost_error_t ghost_printContextInfo(char **str, ghost_context_t *context);
    ghost_error_t ghost_globalIndex(ghost_context_t *ctx, ghost_midx_t lidx, ghost_midx_t *gidx);
    /**
     * @brief Free the context's resources.
     *
     * @param ctx The context.
     *
     * If the context is NULL it will be ignored.
     */
    void ghost_destroyContext(ghost_context_t *ctx);

    /**
     * @brief Assemble communication information in the given context.
     * @param ctx The context.
     * @param col The column indices of the sparse matrix which is bound to the context.
     *
     * @return GHOST_SUCCESS on success or any ghost_error_t on failure. 
     * 
     * The following fields of ghost_context_t are being filled in this function:
     * wishes, wishlist, dues, duelist, hput_pos.
     */
    ghost_error_t ghost_setupCommunication(ghost_context_t *ctx, ghost_midx_t *col);

#ifdef __cplusplus
} //extern "C"
#endif

#endif
