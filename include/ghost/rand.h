/**
 * @file rand.h
 * @brief Functions for handling random number generation states.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_RAND_H
#define GHOST_RAND_H

#include "error.h"

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
     * @brief Create a fix seed for each PU of the machine.
     *
     * @param global_seed The seed.
     *
     * @return 
     *
     * This assumes that there at most as many OpenMP threads as there are PUs.
     */
    ghost_error_t ghost_rand_seed(unsigned int global_seed);
    /**
     * @brief Destroy the random states.
     */
    void ghost_rand_destroy();
    /**
     * @brief Set the random seeds to a scalar.
     *
     * @param seed The seed.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     *
     * This sets the random seed to the same value for each thread and results in a predictable sequence of random numbers.
     * This is only sensible for testing purposes.
     */
    ghost_error_t ghost_rand_set1(unsigned int seed);

#ifdef __cplusplus
}
#endif
#endif
