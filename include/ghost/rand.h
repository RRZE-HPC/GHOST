#ifndef GHOST_RAND_H
#define GHOST_RAND_H

#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif
    /**
     * @brief Get a random state for the calling OpenMP thread.
     *
     * @param s Where to store the state.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     *
     * This function is used for thread-safe random number generation.
     */
    ghost_error_t ghost_rand_get(unsigned int *s);
    ghost_error_t ghost_rand_create();
    void ghost_rand_destroy();

#ifdef __cplusplus
}
#endif
#endif
