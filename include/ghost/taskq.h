/**
 * @ingroup task @{
 * @file taskq.h
 * @brief Types and functions for the task queue.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_TASKQ_H
#define GHOST_TASKQ_H

#include <pthread.h>
#include <hwloc.h>
#include "error.h"
#include "task.h"

/**
 * @brief The task queue.
 */
typedef struct {
    /**
     * @brief The first (= highest priority) task in the queue
     */
    ghost_task_t *head;
    /**
     * @brief The last (= lowest priority) task in the queue
     */
    ghost_task_t *tail;
    /**
     * @brief Serialize access to the queue
     */
    pthread_mutex_t mutex;
} ghost_taskq_t;

#ifdef __cplusplus
extern "C" {
#endif


/**
 * @brief Initializes a task queues.
 * @return GHOST_SUCCESS on success or GHOST_FAILURE on failure.
 */
ghost_error_t ghost_taskq_create();
ghost_error_t ghost_taskq_destroy();

ghost_error_t ghost_taskq_waitall();
ghost_error_t ghost_taskq_waitsome(ghost_task_t **, int, int*);
ghost_error_t ghost_taskq_add(ghost_task_t *task);
ghost_error_t ghost_taskq_startroutine(void *(**func)(void *));

#ifdef __cplusplus
}// extern "C"
#endif

#endif

/** @} */
