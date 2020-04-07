#define _XOPEN_SOURCE 500
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>
#include <errno.h>
#include <unistd.h>
#include <strings.h>

#include "ghost/types.h"
#include "ghost/locality.h"
#include "ghost/task.h"
#include "ghost/thpool.h"
#include "ghost/pumap.h"
#include "ghost/util.h"
#include "ghost/machine.h"
#include "ghost/log.h"
#include "ghost/taskq.h"

#ifdef GHOST_HAVE_OPENMP
#include <omp.h>
#endif


ghost_error ghost_task_unpin(ghost_task *task)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_TASKING);
    unsigned int pu;
    ghost_thpool *ghost_thpool = NULL;
    ghost_thpool_get(&ghost_thpool);
    if (!(task->flags & GHOST_TASK_NOT_PIN)) {
        hwloc_bitmap_foreach_begin(pu, task->coremap);
        if (task->parent && !(task->parent->flags & GHOST_TASK_NOT_ALLOW_CHILD)
            && hwloc_bitmap_isset(task->parent->coremap, pu)) {
            hwloc_bitmap_clr(task->parent->childusedmap, pu);
        } else {
            ghost_pumap_setidle_idx(pu);
        }
        hwloc_bitmap_foreach_end();
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_TASKING);
    return GHOST_SUCCESS;
}

ghost_error ghost_task_string(char **str, ghost_task *t)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_TASKING | GHOST_FUNCTYPE_UTIL);
    GHOST_CALL_RETURN(ghost_malloc((void **)str, 1));
    memset(*str, '\0', 1);

    ghost_header_string(str, "Task %p", (void *)t);
    ghost_line_string(str, "No. of threads", NULL, "%d", t->nThreads);
    ghost_line_string(str, "NUMA node", NULL, "%d", t->LD);
    ghost_footer_string(str);

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_TASKING | GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}

ghost_error ghost_task_enqueue(ghost_task *t)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_TASKING);

    if (!ghost_tasking_enabled()) {
        t->ret = t->func(t->arg);
        t->state = GHOST_TASK_FINISHED;
    } else {
        if (!taskq) { ghost_taskq_create(); }

        pthread_mutex_lock(t->stateMutex);
        t->state = GHOST_TASK_INVALID;
        pthread_mutex_unlock(t->stateMutex);

        pthread_mutex_lock(t->mutex);
        hwloc_bitmap_zero(t->coremap);
        hwloc_bitmap_zero(t->childusedmap);
        pthread_mutex_unlock(t->mutex);

        if (t->parent != NULL) {
            GHOST_DEBUG_LOG(1, "Task's parent overwritten!");
        } else {
            GHOST_CALL_RETURN(ghost_task_cur(&t->parent));
        }

        pthread_mutex_lock(t->stateMutex);
        ghost_taskq_add(t);
        t->state = GHOST_TASK_ENQUEUED;
        pthread_mutex_unlock(t->stateMutex);

        GHOST_DEBUG_LOG(1, "Task added successfully");
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_TASKING);
    return GHOST_SUCCESS;
}

ghost_task_state ghost_taskest(ghost_task *t)
{
    if (!t) { return GHOST_TASK_INVALID; }
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_TASKING);
    ghost_task_state state;
    pthread_mutex_lock(t->stateMutex);
    state = t->state;
    pthread_mutex_unlock(t->stateMutex);

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_TASKING);
    return state;
}

ghost_error ghost_task_wait(ghost_task *task)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_TASKING);

    if (!ghost_tasking_enabled()) {
        return GHOST_SUCCESS;
    } else {
        GHOST_DEBUG_LOG(1, "Waiting for task %p whose state is %s", (void *)task,
            ghost_task_state_string(task->state));


        //    ghost_task *parent = (ghost_task *)pthread_getspecific(ghost_thread_key);
        //    if (parent != NULL) {
        //    GHOST_WARNING_LOG("Waiting on a task from within a task ===> free'ing the parent
        //    task's resources, idle PUs: %d",NIDLECORES); ghost_task_unpin(parent);
        //    GHOST_WARNING_LOG("Now idle PUs: %d",NIDLECORES);
        //    }
        ghost_task *cur;
        ghost_task_cur(&cur);
        if (cur == task) { GHOST_WARNING_LOG("Should wait on myself. Bad idea!"); }

        pthread_mutex_lock(task->stateMutex);
        while (task->state != GHOST_TASK_FINISHED) {
            GHOST_DEBUG_LOG(1, "Waiting for signal @ cond %p from task %p",
                (void *)task->finishedCond, (void *)task);
            pthread_cond_wait(task->finishedCond, task->stateMutex);
        }
        pthread_mutex_unlock(task->stateMutex);
        GHOST_DEBUG_LOG(1, "Finished waitung for task %p!", (void *)task);
    }
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_TASKING);
    return GHOST_SUCCESS;
}

const char *ghost_task_state_string(ghost_task_state state)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_TASKING | GHOST_FUNCTYPE_UTIL);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_TASKING | GHOST_FUNCTYPE_UTIL);
    switch (state) {
        case GHOST_TASK_INVALID:
            return "Invalid";
            break;
        case GHOST_TASK_CREATED:
            return "Created";
            break;
        case GHOST_TASK_ENQUEUED:
            return "Enqueued";
            break;
        case GHOST_TASK_RUNNING:
            return "Running";
            break;
        case GHOST_TASK_FINISHED:
            return "Finished";
            break;
        default:
            return "Unknown";
            break;
    }
}

void ghost_task_destroy(ghost_task *t)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_TASKING | GHOST_FUNCTYPE_TEARDOWN);
    if (t) {
        if (t->state != GHOST_TASK_FINISHED) {
            GHOST_WARNING_LOG("The task is not finished but should be destroyed!");
        }
        sem_destroy(t->progressSem);
        pthread_cond_destroy(t->finishedCond);
        hwloc_bitmap_free(t->coremap);
        hwloc_bitmap_free(t->childusedmap);
        free(t->progressSem);
        free(t->finishedCond);
        pthread_mutex_destroy(t->mutex);
        pthread_mutex_destroy(t->stateMutex);
        pthread_mutex_destroy(t->finishedMutex);
        free(t->mutex);
        free(t->stateMutex);
        free(t->finishedMutex);
        free(t);
    }
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_TASKING | GHOST_FUNCTYPE_TEARDOWN);
}

ghost_error ghost_task_create(ghost_task **t, int nThreads, int LD, void *(*func)(void *),
    void *arg, ghost_task_flags flags, ghost_task **depends, int ndepends)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_TASKING | GHOST_FUNCTYPE_SETUP);
    ghost_error ret = GHOST_SUCCESS;

    GHOST_CALL_GOTO(ghost_malloc((void **)t, sizeof(ghost_task)), err, ret);

    if (nThreads == GHOST_TASK_FILL_LD) {
        if (LD < 0) {
            GHOST_WARNING_LOG("FILL_LD does only work when the LD is given! Not creating task!");
            return GHOST_ERR_INVALID_ARG;
        }
        ghost_pumap_npu(&(*t)->nThreads, LD);
    } else if (nThreads == GHOST_TASK_FILL_ALL) {
#ifdef GHOST_HAVE_OPENMP
        GHOST_CALL_GOTO(ghost_pumap_npu(&((*t)->nThreads), GHOST_NUMANODE_ANY), err, ret);
#else
        (*t)->nThreads = 1; // TODO is this the correct behavior?
#endif
    } else {
        (*t)->nThreads = nThreads;
    }

    (*t)->LD = LD;
    (*t)->func = func;
    (*t)->arg = arg;
    (*t)->flags = flags;
    (*t)->depends = depends;
    (*t)->ndepends = ndepends;

    //    GHOST_CALL_GOTO(ghost_malloc((void **)&(*t)->cores,sizeof(int)*(*t)->nThreads),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&(*t)->progressSem, sizeof(sem_t)), err, ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&(*t)->finishedCond, sizeof(pthread_cond_t)), err, ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&(*t)->mutex, sizeof(pthread_mutex_t)), err, ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&(*t)->finishedMutex, sizeof(pthread_mutex_t)), err, ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&(*t)->stateMutex, sizeof(pthread_mutex_t)), err, ret);
    sem_init((*t)->progressSem, 0, 0);
    pthread_mutex_init((*t)->mutex, NULL);
    pthread_mutex_init((*t)->finishedMutex, NULL);
    pthread_mutex_init((*t)->stateMutex, NULL);
    pthread_cond_init((*t)->finishedCond, NULL);
    (*t)->state = GHOST_TASK_CREATED;
    (*t)->coremap = hwloc_bitmap_alloc();
    (*t)->childusedmap = hwloc_bitmap_alloc();
    if (!(*t)->coremap || !(*t)->childusedmap) {
        GHOST_ERROR_LOG("Could not allocate hwloc bitmaps");
        ret = GHOST_ERR_HWLOC;
        goto err;
    }

    (*t)->next = NULL;
    (*t)->prev = NULL;
    (*t)->parent = NULL;
    (*t)->ret = NULL;

    goto out;
err:
    if (*t) {
        // free((*t)->cores); (*t)->cores = NULL;
        free((*t)->finishedCond);
        (*t)->finishedCond = NULL;
        free((*t)->mutex);
        (*t)->mutex = NULL;
        free((*t)->ret);
        (*t)->ret = NULL;
        hwloc_bitmap_free((*t)->coremap);
        (*t)->coremap = NULL;
        hwloc_bitmap_free((*t)->childusedmap);
        (*t)->childusedmap = NULL;
    }
    free(*t);
    *t = NULL;
out:

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_TASKING | GHOST_FUNCTYPE_SETUP);
    return ret;
}

ghost_error ghost_task_cur(ghost_task **task)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_TASKING);

    ghost_thpool *thpool;
    ghost_thpool_get(&thpool);

    if (!thpool) {
        *task = NULL;
    } else {
        pthread_key_t key;
        GHOST_CALL_RETURN(ghost_thpool_key(&key));
        *task = (ghost_task *)pthread_getspecific(key);
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_TASKING);
    return GHOST_SUCCESS;
}

bool ghost_tasking_enabled() { return false; }
