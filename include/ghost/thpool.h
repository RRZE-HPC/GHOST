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
     */
    pthread_t *threads;
    /**
     * @brief The PU (Processing Unit) of each GHOST thread.
     */
    hwloc_obj_t *PUs;
    /**
     * @brief The cpuset this thread pool is covering. 
     */
    hwloc_bitmap_t cpuset;
    /**
     * @brief A bitmap with one bit per PU where 1 means that a PU is busy and 0 means that it is
     * idle.
     */
       hwloc_bitmap_t busy;
    /**
     * @brief The total number of threads in the thread pool
     */
    int nThreads;
    /**
     * @brief The number of LDs covered by the pool's threads.
     */
    int nLDs;
    /**
     * @brief Counts the number of readily  
     */
    sem_t *sem; // counts the number of initialized threads
} ghost_thpool_t;

ghost_error_t ghost_thpool_init(hwloc_cpuset_t cpuset, void *(*func)(void *));
ghost_error_t ghost_thpool_finish();
ghost_error_t ghost_getThreadpool(ghost_thpool_t **thpool);
int nBusyCoresAtLD(hwloc_bitmap_t bm, int ld);
int nIdleCoresAtLD(int ld);
int nThreadsPerLD(int ld);
int nIdleCores();
int coreIdx(int LD, int t);

#endif

