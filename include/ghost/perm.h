/**
 * @file perm.h
 * @brief Types for handling permutations.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_PERM_H
#define GHOST_PERM_H

typedef enum
{
    GHOST_PERMUTATION_LOCAL,
    GHOST_PERMUTATION_GLOBAL
}
ghost_permutation_scope_t;

typedef enum
{
    GHOST_PERMUTATION_ORIG2PERM,
    GHOST_PERMUTATION_PERM2ORIG
}
ghost_permutation_direction_t;

typedef struct
{
    /**
     * @brief Gets an original index and returns the corresponding permuted position.
     *
     * NULL if no permutation applied to the matrix.
     */
    ghost_idx_t *perm;
    /**
     * @brief Gets an index in the permuted system and returns the original index.
     *
     * NULL if no permutation applied to the matrix.
     */
    ghost_idx_t *invPerm;
    ghost_permutation_scope_t scope;
    ghost_idx_t len;
}
ghost_permutation_t;

#endif
