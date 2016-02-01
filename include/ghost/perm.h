/**
 * @file perm.h
 * @brief Types for handling permutations.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_PERM_H
#define GHOST_PERM_H

#include "types.h"

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

    ghost_gidx *cu_perm;

    ghost_permutation_scope_t scope;
    ghost_gidx len;
}
ghost_permutation;

#endif
