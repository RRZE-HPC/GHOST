#ifndef GHOST_THPOOL_H
#define GHOST_THPOOL_H

#include <pthread.h>
#include <semaphore.h>
#include <hwloc.h>

#include "error.h"

/**
 * @brief The thread pool consisting of all threads that will ever do some
 * tasking-related work.
 */
typedef struct ghost_thpool_t {
    /**
     * @brief The pthread of each GHOST thread.
     *
     * Length: #nThreads.
     */
    pthread_t *threads;
    /**
     * @brief The total number of threads in the thread pool
     */
    int nThreads;
    /**
     * @brief Counts the number of readily  
     */
    sem_t *sem; // counts the number of initialized threads
} ghost_thpool_t;

ghost_error_t ghost_thpool_create(int nThreads, void *(*func)(void *));
ghost_error_t ghost_thpool_destroy();
ghost_error_t ghost_thpool_get(ghost_thpool_t **thpool);

#endif

