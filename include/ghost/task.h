#ifndef GHOST_TASK_H
#define GHOST_TASK_H

#include <pthread.h>
#include <hwloc.h>
#include "error.h"

#define GHOST_TASK_LD_UNDEFINED -2 // initializer
//#define GHOST_TASK_LD_ANY -1 // execute task on any LD

#define GHOST_TASK_DEFAULT 0
#define GHOST_TASK_PRIO_HIGH 1 // task will be added to the head of the queue
#define GHOST_TASK_LD_STRICT 2 // task _must_ be executed on the defined LD
#define GHOST_TASK_USE_PARENTS 4 // task can use the parent's resources if added from within a task 
#define GHOST_TASK_NO_PIN 8 
#define GHOST_TASK_ONLY_HYPERTHREADS 16
#define GHOST_TASK_NO_HYPERTHREADS 32


#define GHOST_TASK_FILL_LD -1 // use all threads of the given LD
#define GHOST_TASK_FILL_ALL -2 // use all threads of all LDs

#define GHOST_THPOOL_NTHREADS_FULLNODE ((int*)(0xBEEF))
#define GHOST_THPOOL_FTHREAD_DEFAULT ((int *)(0xBEEF))
#define GHOST_THPOOL_LEVELS_FULLSMT 0

typedef enum ghost_task_state_t {
    GHOST_TASK_INVALID, // task has not been enqueued
    GHOST_TASK_ENQUEUED, // task has been enqueued
    GHOST_TASK_RUNNING, // task is currently running
    GHOST_TASK_FINISHED // task has finished
} ghost_task_state_t;

/**
 * @brief This structure represents a GHOST task.
 *
 * This data structure holds all necessary information for
 * a task. The members #nThreads, #LD, #flags, #func and #arg have to be set by
 * the user in ghost_task_init(). All other members are set by the library at
 * some point.
 */
typedef struct ghost_task_t {
    /**
     * @brief The number of threads the task should use. (user-defined)
     */
    int nThreads;
    /**
     * @brief The index of the queue in which the task should be present and
     * (preferrably) running. (user-defined)
     */
    int LD;
    /**
     * @brief Optional flags for the task. (user-defined)
     */
    int flags;
    /**
     * @brief The function to be executed by the task. (user-defined)
     */
    void *(*func)(void *);
    /**
     * @brief The arguments to the task's function. (user-defined)
     */
    void *arg;

    // set by the library
    /**
     * @brief The current state of the task. (set by the library)
     */
    int *state;
    /**
     * @brief The list of cores where the task's threads are running. (set by the library)
     */
    int *cores;
    /**
     * @brief Map of cores this task is using. (set by the library)
     */
    hwloc_bitmap_t coremap;
    /**
     * @brief Map of cores a child of this task is using. (set by the library)
     */
    hwloc_bitmap_t childusedmap;
    /**
     * @brief The return value of the task's funtion. (set by the library)
     */
    void *ret;
    /**
     * @brief Pointer to the next task in the queue. (set by the library)
     */
    struct ghost_task_t *next; 
    /**
     * @brief Pointer to the previous task in the queue. (set by the library)
     */
    struct ghost_task_t *prev;
    /**
     * @brief The adding task if the task has been added from within a task.
     * (set by the library)
     */
    struct ghost_task_t *parent;
    /**
     * @brief Indicator that the task is finished. (set by the library)
     */
    pthread_cond_t *finishedCond;
    /**
     * @brief Protect accesses to the task's members. (set by the library)
     */
    pthread_mutex_t *mutex;
    /**
     * @brief Set to one as soon as the task's resources have been free'd.
     * This can be the case when the task waits for a child-task to finish or
     * when the task itself is finished.
     */
    int freed;
} ghost_task_t;

#ifdef __cplusplus
extern "C" {
#endif

ghost_error_t ghost_task_create(ghost_task_t **task, int nThreads, int LD, void *(*func)(void *), void *arg, int flags);
ghost_error_t ghost_task_enqueue(ghost_task_t *);
ghost_error_t ghost_task_wait(ghost_task_t *);
ghost_task_state_t ghost_task_test(ghost_task_t *);
void ghost_task_destroy(ghost_task_t *); 
ghost_error_t ghost_task_unpin(ghost_task_t *task);

char *ghost_task_stateString(ghost_task_state_t state);


#ifdef __cplusplus
}// extern "C"
#endif

#endif
