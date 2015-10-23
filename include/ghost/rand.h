/**
 * @file rand.h
 * @brief Functions for handling random number generation states.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_RAND_H
#define GHOST_RAND_H

#include "error.h"

typedef enum
{
    GHOST_RAND_SEED_PU = 1,
    GHOST_RAND_SEED_RANK = 2,
    GHOST_RAND_SEED_TIME = 4,
} ghost_rand_seed_t;

#define GHOST_RAND_SEED_ALL (GHOST_RAND_SEED_PU|GHOST_RAND_SEED_RANK|GHOST_RAND_SEED_TIME)

#ifdef __cplusplus
extern "C" {
#endif
    /**
     * @brief Get a random state for the calling thread.
     *
     * @param s Where to store the state.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     *
     * The random state is determined from the calling OpenMP thread index.
     * OpenMP do not have to be pinned in order to return correct random seeds.
     * This function is used for thread-safe random number generation.
     */
    ghost_error_t ghost_rand_get(unsigned int **s);
    /**
     * @brief Create a random seed for each PU of the machine.
     *
     * @return 
     *
     * This assumes that there at most as many OpenMP threads as there are PUs.
     */
    ghost_error_t ghost_rand_create();
    /**
     * @brief Destroy the random states.
     */
    void ghost_rand_destroy();
    /**
     * @brief Set the random number seed.
     *
     * @param which Which components of the seed should be set. Components can be combined with OR.
     * @param seed The value to which the component should be set.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     *
     * The random number seed used by GHOST is a combination of the time, the rank and the PU.
     * Each of those components can be set individually with this function.
     * This assumes that there at most as many OpenMP threads as there are PUs.
     * It is not possible to set the PU seed on CUDA devices.
     */
    ghost_error_t ghost_rand_seed(ghost_rand_seed_t which, unsigned int seed);
    /**
     * @brief Get the CUDA random seed.
     *
     * @return The CUDA random seed.
     */
    int ghost_rand_cu_seed_get();

#ifdef __cplusplus
}
#endif
#endif
