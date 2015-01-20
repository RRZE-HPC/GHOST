#define _XOPEN_SOURCE 500
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>
#include <errno.h>
#include <unistd.h>

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



ghost_error_t ghost_task_unpin(ghost_task_t *task)
{
    unsigned int pu;
    ghost_thpool_t *ghost_thpool = NULL;
    ghost_thpool_get(&ghost_thpool);
    if (!(task->flags & GHOST_TASK_NOT_PIN)) {
        hwloc_bitmap_foreach_begin(pu,task->coremap);
            if (task->parent && 
                    !(task->parent->flags & GHOST_TASK_NOT_ALLOW_CHILD) && 
                    hwloc_bitmap_isset(task->parent->coremap,pu)) 
            {
                hwloc_bitmap_clr(task->parent->childusedmap,pu);
            } else {
                ghost_pumap_setidle_idx(pu);
            }
        hwloc_bitmap_foreach_end();
    }

    return GHOST_SUCCESS;
}

ghost_error_t ghost_task_string(char **str, ghost_task_t *t) 
{
    GHOST_CALL_RETURN(ghost_malloc((void **)str,1));
    memset(*str,'\0',1);

    ghost_header_string(str,"Task %p",(void *)t);
    ghost_line_string(str,"No. of threads",NULL,"%d",t->nThreads);
    ghost_line_string(str,"NUMA node",NULL,"%d",t->LD);
    ghost_footer_string(str);

    return GHOST_SUCCESS;
}

ghost_error_t ghost_task_enqueue(ghost_task_t *t)
{
    pthread_mutex_lock(t->mutex);
    t->state = GHOST_TASK_INVALID;

    hwloc_bitmap_zero(t->coremap);
    hwloc_bitmap_zero(t->childusedmap);
    GHOST_CALL_RETURN(ghost_task_cur(&t->parent));

    ghost_taskq_add(t);
    t->state = GHOST_TASK_ENQUEUED;
    pthread_mutex_unlock(t->mutex);

    DEBUG_LOG(1,"Task added successfully");

    return GHOST_SUCCESS;
}

ghost_task_state_t ghost_task_test(ghost_task_t * t)
{
    if (!t) {
        return GHOST_TASK_INVALID;
    }
    return t->state;
}

ghost_error_t ghost_task_wait(ghost_task_t * task)
{
    DEBUG_LOG(1,"Waiting for task %p whose state is %s",(void *)task,ghost_task_state_string(task->state));


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

char *ghost_task_state_string(ghost_task_state_t state)
{
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

void ghost_task_destroy(ghost_task_t *t)
{
    if (t) {
        pthread_mutex_destroy(t->mutex);
        pthread_cond_destroy(t->finishedCond);

//        free(t->cores);
        hwloc_bitmap_free(t->coremap);
        hwloc_bitmap_free(t->childusedmap);
        //free(t->ret);
        free(t->mutex);
        free(t->finishedCond);
    }
    free(t); t = NULL;
}

ghost_error_t ghost_task_create(ghost_task_t **t, int nThreads, int LD, void *(*func)(void *), void *arg, ghost_task_flags_t flags, ghost_task_t **depends, int ndepends)
{
    ghost_error_t ret = GHOST_SUCCESS;

    GHOST_CALL_RETURN(ghost_malloc((void **)t,sizeof(ghost_task_t)));
    
    if (nThreads == GHOST_TASK_FILL_LD) {
        if (LD < 0) {
            WARNING_LOG("FILL_LD does only work when the LD is given! Not creating task!");
            return GHOST_ERR_INVALID_ARG;
        }
        ghost_pumap_npu(&(*t)->nThreads,LD);
    } 
    else if (nThreads == GHOST_TASK_FILL_ALL) {
#ifdef GHOST_HAVE_OPENMP
        GHOST_CALL_GOTO(ghost_pumap_npu(&((*t)->nThreads),GHOST_NUMANODE_ANY),err,ret);
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
    (*t)->depends = depends;
    (*t)->ndepends = ndepends;

//    GHOST_CALL_GOTO(ghost_malloc((void **)&(*t)->cores,sizeof(int)*(*t)->nThreads),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&(*t)->finishedCond,sizeof(pthread_cond_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&(*t)->mutex,sizeof(pthread_mutex_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&(*t)->ret,sizeof(void *)),err,ret);
    pthread_mutex_init((*t)->mutex,NULL);
    pthread_cond_init((*t)->finishedCond,NULL);
    (*t)->state = GHOST_TASK_CREATED;
    (*t)->coremap = hwloc_bitmap_alloc();
    (*t)->childusedmap = hwloc_bitmap_alloc();
    if (!(*t)->coremap || !(*t)->childusedmap) {
        ERROR_LOG("Could not allocate hwloc bitmaps");
        ret = GHOST_ERR_HWLOC;
        goto err;
    }

    (*t)->next = NULL;
    (*t)->prev = NULL;
    (*t)->parent = NULL;

    goto out;
err:
    if (*t) {
        //free((*t)->cores); (*t)->cores = NULL;
        free((*t)->finishedCond); (*t)->finishedCond = NULL;
        free((*t)->mutex); (*t)->mutex = NULL;
        free((*t)->ret); (*t)->ret = NULL;
        hwloc_bitmap_free((*t)->coremap); (*t)->coremap = NULL;
        hwloc_bitmap_free((*t)->childusedmap); (*t)->childusedmap = NULL;
    }
    free(*t); *t = NULL;
out:

    return ret;
}

ghost_error_t ghost_task_cur(ghost_task_t **task)
{
    pthread_key_t key;
    GHOST_CALL_RETURN(ghost_thpool_key(&key));
    *task = (ghost_task_t *)pthread_getspecific(key);

    return GHOST_SUCCESS;

}

