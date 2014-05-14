/**
 * @ingroup task @{
 * @file thpool.h
 * @brief Types and functions for the thread pool.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_THPOOL_H
#define GHOST_THPOOL_H

#include <pthread.h>
#include <semaphore.h>
#include <hwloc.h>

#include "error.h"

/**
 * @brief The thread pool consisting of all shepherd threads that will ever handle a GHOST task.
 */
typedef struct ghost_thpool_t {
    /**
     * @brief The pthread to thread.
     *
     * Length: #nThreads.
     */
    pthread_t *threads;
    /**
     * @brief The total number of threads in the thread pool.
     */
    int nThreads;
    /**
     * @brief Counts the number of initialized threads.
     * @deprecated Check whether this is needed. 
     *
     * Only when all threads are initialized and have reached their start routine, ghost_thpool_create() will return.
     */
    sem_t *sem;
} ghost_thpool_t;

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @brief Creates or resizes a thread pool.
     *
     * @param nThreads The number of threads in the thread pool.
     * @param func The start routine for each thread.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     *
     * In GHOST, the start routine works on a task queue.
     * In a usual scenario one would like to have one shepherd thread for each processing unit.
     * This assures that N tasks can run at the same time if each one occupies on processing unit.
     * If this function is called a second, third etc. time, the size of the thread pool will be resized to the given nThreads. 
     */
    ghost_error_t ghost_thpool_create(int nThreads, void *(*func)(void *));
    /**
     * @brief Destroy the thread pool.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     *
     * Before free'ing the resources, all pthreads in the thread pool are being joined (this is also where an error could occur).
     */
    ghost_error_t ghost_thpool_destroy();
    /**
     * @brief Get the thread pool
     *
     * @param thpool Where to store the thread pool.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_thpool_get(ghost_thpool_t **thpool);

    ghost_error_t ghost_thpool_key(pthread_key_t *key);
    

#ifdef __cplusplus
}
#endif

#endif

/** @} */
