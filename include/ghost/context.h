/**
 * @file ghost_context.h
 */
#ifndef __GHOST_CONTEXT_H__
#define __GHOST_CONTEXT_H__

#include "config.h"
#include "types.h"
#include "error.h"

/**
 * @brief This struct holds all possible flags for a context.
 */
typedef enum ghost_context_flags_t {
    GHOST_CONTEXT_DEFAULT = 0, 
    GHOST_CONTEXT_GLOBAL = 1, 
    GHOST_CONTEXT_DISTRIBUTED = 2,
    /**
     * @brief Distribute work among the ranks by number of nonzeros instead of number of rows
     */
    GHOST_CONTEXT_WORKDIST_NZE = 4,
    GHOST_CONTEXT_NO_COMBINED_SOLVERS = 8,
    GHOST_CONTEXT_NO_SPLIT_SOLVERS = 16
} ghost_context_flags_t;

    


#ifdef __cplusplus
extern "C" {
#endif

ghost_error_t ghost_createContext(ghost_context_t **ctx, ghost_midx_t gnrows, ghost_midx_t gncols, ghost_context_flags_t context_flags, char *matrixPath, MPI_Comm comm, double weight); 
void              ghost_freeContext(ghost_context_t *);
int ghost_setupCommunication(ghost_context_t *ctx, ghost_midx_t *col);

#ifdef __cplusplus
} //extern "C"
#endif

#endif
