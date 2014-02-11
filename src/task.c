#define _XOPEN_SOURCE 500
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>
#include <errno.h>
#include <unistd.h>

#include "ghost/types.h"
#include "ghost/constants.h"
#include "ghost/locality.h"
#include "ghost/task.h"
#include "ghost/thpool.h"
#include "ghost/pumap.h"
#include "ghost/util.h"
#include "ghost/machine.h"
#include "ghost/log.h"
#include "ghost/taskq.h"

#if GHOST_HAVE_OPENMP
#include <omp.h>
#endif



ghost_error_t ghost_task_unpin(ghost_task_t *task)
{
    ghost_thpool_t *ghost_thpool = NULL;
    ghost_thpool_get(&ghost_thpool);
    if (!(task->flags & GHOST_TASK_NO_PIN)) {
        for (int t=0; t<task->nThreads; t++) {
            if ((task->flags & GHOST_TASK_USE_PARENTS) && 
                    task->parent && 
                    hwloc_bitmap_isset(task->parent->coremap,task->cores[t])) 
            {
                hwloc_bitmap_clr(task->parent->childusedmap,task->cores[t]);
            } else {
                ghost_pumap_setIdleIdx(task->cores[t]);
            }
        }
    }
    //    task->freed = 1;

    return GHOST_SUCCESS;


}

ghost_error_t ghost_task_print(ghost_task_t *t) 
{
    ghost_printHeader("Task %p",(void *)t);
    ghost_printLine("No. of threads",NULL,"%d",t->nThreads);
    ghost_printLine("LD",NULL,"%d",t->LD);
    ghost_printFooter();

    return GHOST_SUCCESS;
}

ghost_error_t ghost_task_enqueue(ghost_task_t *t)
{
    // if a task is initialized _once_ but added several times, this has to be done each time it is added
    pthread_cond_init(t->finishedCond,NULL);
    pthread_mutex_init(t->mutex,NULL);
    t->state = GHOST_TASK_INVALID;
    memset(t->cores,0,sizeof(int)*t->nThreads);

    hwloc_bitmap_zero(t->coremap);
    hwloc_bitmap_zero(t->childusedmap);
    t->parent = (ghost_task_t *)pthread_getspecific(ghost_thread_key);
    //    t->freed = 0;

    //DEBUG_LOG(1,"Task %p w/ %d threads goes to queue %p (LD %d)",(void *)t,t->nThreads,(void *)taskq,t->LD);
    ghost_taskq_addTask(t);
    //taskq_additem(taskq,t);
    //ghost_task_destroy(&commTask);
    t->state = GHOST_TASK_ENQUEUED;


    DEBUG_LOG(1,"Task added successfully");

    return GHOST_SUCCESS;
}

ghost_task_state_t ghost_task_test(ghost_task_t * t)
{
    if (t == NULL || t->state == NULL) {
        return GHOST_TASK_INVALID;
    }
    return t->state;
}

ghost_error_t ghost_task_wait(ghost_task_t * task)
{
    DEBUG_LOG(1,"Waiting for task %p whose state is %d",(void *)task,task->state);


    //    ghost_task_t *parent = (ghost_task_t *)pthread_getspecific(ghost_thread_key);
    //    if (parent != NULL) {
    //    WARNING_LOG("Waiting on a task from within a task ===> free'ing the parent task's resources, idle PUs: %d",NIDLECORES);
    //    ghost_task_unpin(parent);
    //    WARNING_LOG("Now idle PUs: %d",NIDLECORES);
    //    }

    pthread_mutex_lock(task->mutex);
    if (task->state != GHOST_TASK_FINISHED) {
        DEBUG_LOG(1,"Waiting for signal @ cond %p from task %p",(void *)task->finishedCond,(void *)task);
        pthread_cond_wait(task->finishedCond,task->mutex);
    } else {
        DEBUG_LOG(1,"Task %p has already finished",(void *)task);
    }

    // pin again if have been unpinned

    pthread_mutex_unlock(task->mutex);
    DEBUG_LOG(1,"Finished waitung for task %p!",(void *)task);

    return GHOST_SUCCESS;

}

char *ghost_task_stateString(ghost_task_state_t state)
{
    switch (state) {
        case GHOST_TASK_INVALID: 
            return "Invalid";
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

void ghost_task_destroy(ghost_task_t *t)
{
    if (t) {
        pthread_mutex_destroy(t->mutex);
        pthread_cond_destroy(t->finishedCond);

        free(t->cores);
        hwloc_bitmap_free(t->coremap);
        hwloc_bitmap_free(t->childusedmap);
        //free(t->ret);
        free(t->mutex);
        free(t->finishedCond);
    }
    free(t); t = NULL;
}

ghost_error_t ghost_task_create(ghost_task_t **t, int nThreads, int LD, void *(*func)(void *), void *arg, int flags)
{
    GHOST_CALL_RETURN(ghost_malloc((void **)t,sizeof(ghost_task_t)));
    /*  if (ghost_thpool == NULL) {
        WARNING_LOG("The thread pool is not initialized. Something went terribly wrong.");*/
    /*        int nt = ghost_getNumberOfPhysicalCores()/ghost_getNumberOfRanks(ghost_node_comm);
              int ft = ghost_getRank(ghost_node_comm)*nt;
              int poolThreads[] = {nt,nt};
              int firstThread[] = {ft,ft};
              int levels = ghost_getNumberOfHwThreads()/ghost_getNumberOfPhysicalCores();
    //DEBUG_LOG(1,"Trying to initialize a task but the thread pool has not yet been initialized. Doing the init now with %d threads!",nt*levels);
    //ghost_thpool_init(poolThreads,firstThread,levels);*/
    //        }
    /* if (taskq == NULL) {
       DEBUG_LOG(1,"Trying to initialize a task but the task queues have not yet been initialized. Doing the init now...");
       ghost_taskq_init();
       }
     */
    if (nThreads == GHOST_TASK_FILL_LD) {
        if (LD < 0) {
            WARNING_LOG("FILL_LD does only work when the LD is given! Not adding task!");
            return GHOST_ERR_INVALID_ARG;
        }
        ghost_getNumberOfPUs(&(*t)->nThreads,LD);
    } 
    else if (nThreads == GHOST_TASK_FILL_ALL) {
#ifdef GHOST_HAVE_OPENMP
        GHOST_CALL_RETURN(ghost_getNumberOfPUs(&(*t)->nThreads,GHOST_NUMANODE_ANY));
#else
        (*t)->nThreads = 1; //TODO is this the correct behavior?
#endif
    } 
    else {
        (*t)->nThreads = nThreads;
    }

    (*t)->LD = LD;
    (*t)->func = func;
    (*t)->arg = arg;
    (*t)->flags = flags;

    //    t->freed = 0;
    GHOST_CALL_RETURN(ghost_malloc((void **)&(*t)->cores,sizeof(int)*(*t)->nThreads));
    GHOST_CALL_RETURN(ghost_malloc((void **)&(*t)->finishedCond,sizeof(pthread_cond_t)));
    GHOST_CALL_RETURN(ghost_malloc((void **)&(*t)->mutex,sizeof(pthread_mutex_t)));
    GHOST_CALL_RETURN(ghost_malloc((void **)&(*t)->ret,sizeof(void *)));
    (*t)->state = GHOST_TASK_CREATED;
    (*t)->coremap = hwloc_bitmap_alloc();
    (*t)->childusedmap = hwloc_bitmap_alloc();
    (*t)->next = NULL;
    (*t)->prev = NULL;
    (*t)->parent = NULL;

    return GHOST_SUCCESS;
}

