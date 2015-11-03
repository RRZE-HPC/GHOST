#define _XOPEN_SOURCE 500
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>
#include <errno.h>
#include <unistd.h>

#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/core.h"
#include "ghost/locality.h"
#include "ghost/task.h"
#include "ghost/taskq.h"
#include "ghost/thpool.h"
#include "ghost/pumap.h"
#include "ghost/util.h"
#include "ghost/machine.h"
#include "ghost/log.h"
#include "ghost/omp.h"
#include "ghost/bitmap.h"

#ifdef GHOST_HAVE_MKL
#include <mkl.h>
#endif

#ifdef GHOST_HAVE_INSTR_LIKWID
#include <likwid.h>
#endif

#ifdef GHOST_HAVE_OPENMP
#include <omp.h>
#endif

#ifdef GHOST_HAVE_CUDA
#include "ghost/cu_util.h"
#endif


/**
 * @brief The task queue created by ghost_taskq_init().
 */
ghost_taskq_t *taskq = NULL;

/**
 * @brief This is set to 1 if the tasqs are about to be killed. 
 The threads will exit their infinite loops in this case.
 */
static int killed = 0;

/**
 * @brief Protects access to global variables.
 */
static pthread_mutex_t globalMutex;

/**
 * @brief This is waited for in ghost_task_waitsome() and broadcasted in ghost_task_wait() when the task has finished.
 */
static pthread_cond_t anyTaskFinishedCond; 

/**
 * @brief The mutex to protect anyTaskFinishedCond.
 */
static pthread_mutex_t anyTaskFinishedMutex;

static int num_pending_tasks = 0;
/**
 * @brief Holds the number of valid thread counts for tasks.
 * This is usually the number of PUs+1 (for zero-PU tasks)
 */
static int nthreadcount = 0;

static void * thread_main(void *arg);

static pthread_cond_t ** newTaskCond_by_threadcount;
static pthread_mutex_t * newTaskMutex_by_threadcount;
static int * num_shep_by_threadcount;
static int * waiting_shep_by_threadcount;
static int * num_tasks_by_threadcount;

static pthread_key_t threadcount_key;
static pthread_key_t mutex_key;


ghost_error_t ghost_taskq_create()
{
    int t,s;
    int npu;

    pthread_mutex_init(&globalMutex,NULL);
    pthread_mutex_lock(&globalMutex);

    ghost_machine_npu(&npu,GHOST_NUMANODE_ANY);
    nthreadcount = npu+1;

    GHOST_CALL_RETURN(ghost_malloc((void **)&taskq,sizeof(ghost_taskq_t)));
    pthread_mutex_init(&(taskq->mutex),NULL);

    pthread_mutex_lock(&(taskq->mutex));

    GHOST_CALL_RETURN(ghost_malloc((void **)&newTaskMutex_by_threadcount,(nthreadcount)*sizeof(pthread_mutex_t)));
    GHOST_CALL_RETURN(ghost_malloc((void **)&newTaskCond_by_threadcount,(nthreadcount)*sizeof(pthread_cond_t *)));
    GHOST_CALL_RETURN(ghost_malloc((void **)&num_shep_by_threadcount,(nthreadcount)*sizeof(int)));
    GHOST_CALL_RETURN(ghost_malloc((void **)&waiting_shep_by_threadcount,(nthreadcount)*sizeof(int)));
    GHOST_CALL_RETURN(ghost_malloc((void **)&num_tasks_by_threadcount,(nthreadcount)*sizeof(int)));
    taskq->tail = NULL;
    taskq->head = NULL;

    pthread_cond_init(&anyTaskFinishedCond,NULL);
    pthread_mutex_init(&anyTaskFinishedMutex,NULL);
    pthread_key_create(&threadcount_key,NULL);
    pthread_key_create(&mutex_key,NULL);


    for (t=0; t<nthreadcount; t++) {
        pthread_mutex_init(&newTaskMutex_by_threadcount[t],NULL);
        num_shep_by_threadcount[t] = 1;
        waiting_shep_by_threadcount[t] = 0; 
        num_tasks_by_threadcount[t] = 0; 
        GHOST_CALL_RETURN(ghost_malloc((void **)&newTaskCond_by_threadcount[t],num_shep_by_threadcount[t]*sizeof(pthread_cond_t)));
        for (s=0; s<num_shep_by_threadcount[t]; s++) {
            pthread_cond_init(&newTaskCond_by_threadcount[t][s],NULL);
        }
    }

    void *(*threadFunc)(void *);
    ghost_taskq_startroutine(&threadFunc);
    ghost_thpool_create(nthreadcount,threadFunc);

    pthread_mutex_unlock(&(taskq->mutex));
    pthread_mutex_unlock(&globalMutex);
    return GHOST_SUCCESS;
}

/**
 * @brief Deletes a given task from a given queue.
 *
 * @param q
 * @param t
 *
 * @return GHOST_SUCCESS on success or GHOST_FAILURE on failure.
 */
static int taskq_deleteTask(ghost_taskq_t *q, ghost_task_t *t)
{
    if (t == q->head) {
        DEBUG_LOG(1,"Removing head from queue %p",(void *)q);
        q->head = t->next;
        if (q->head != NULL)
            q->head->prev = NULL;
    }
    if (t == q->tail) {
        DEBUG_LOG(1,"Removing tail from queue %p",(void *)q);
        q->tail = t->prev;
        if (q->tail != NULL)
            q->tail->next = NULL;
    }

    if (t->prev != NULL)
        t->prev->next = t->next;

    if (t->next != NULL)
        t->next->prev = t->prev;


    return GHOST_SUCCESS;
}


/**
 * @brief Try to find a task in the given queue. 
 If there is a suited task, delete it from the queue, reserve enough cores in order to execute the task
 and pin the task's threads to the reserved cores
 *
 * @param q
 *
 * @return A pointer to the selected task or NULL if no suited task could be found. 
 */
static ghost_task_t * taskq_findDeleteAndPinTask(ghost_taskq_t *q, int nthreads)
{
    if (q == NULL) {
        WARNING_LOG("Tried to find a job but the queue is NULL");
        return NULL;
    }
    ghost_task_t *curTask = q->head;

    DEBUG_LOG(1,"Try to find a suitable task");

    while(curTask != NULL)
    {
        pthread_mutex_lock(curTask->mutex);
        if (curTask->nThreads != nthreads) {
            DEBUG_LOG(2,"Incorrect thread count! Try next task...");
            pthread_mutex_unlock(curTask->mutex);
            curTask = curTask->next;
            continue;
        }
        int d;
        for (d=0; d<curTask->ndepends; d++) {
            pthread_mutex_lock(curTask->depends[d]->stateMutex);
            if (curTask->depends[d]->state != GHOST_TASK_FINISHED) {
                pthread_mutex_unlock(curTask->depends[d]->stateMutex);
                break;
            }
            pthread_mutex_unlock(curTask->depends[d]->stateMutex);
        }
        if (d<curTask->ndepends) {
            pthread_mutex_unlock(curTask->mutex);
            curTask = curTask->next;
            continue;
        }


        if (curTask->flags & GHOST_TASK_NOT_PIN) {
            taskq_deleteTask(q,curTask);    
            ghost_thread_unpin();
            if( curTask->nThreads > 0 ) {
                ghost_omp_nthread_set(curTask->nThreads);
#ifdef GHOST_HAVE_MKL
                mkl_set_num_threads(curTask->nThreads); 
#endif
#pragma omp parallel
                ghost_thread_unpin();
            }
            pthread_mutex_unlock(curTask->mutex);
            return curTask;
        }

        int totalPUs;
        if (curTask->flags & GHOST_TASK_LD_STRICT) {
            ghost_pumap_npu(&totalPUs,curTask->LD);
            if (curTask->nThreads > totalPUs) {
                PERFWARNING_LOG("More threads requested than PUs available in the requested strict locality domain! Will reduce the number of threads!");
                curTask->nThreads = totalPUs;
            }
        } else {
            ghost_pumap_npu(&totalPUs,GHOST_NUMANODE_ANY);
            if (curTask->nThreads > totalPUs) {
                PERFWARNING_LOG("More threads requested than PUs available for this process! Will reduce the number of threads!");
                curTask->nThreads = totalPUs;
            }
        }



        hwloc_obj_t numanode;
        ghost_machine_numanode(&numanode,curTask->LD);

        int availcores = 0;
        if (curTask->flags & GHOST_TASK_LD_STRICT) {
            ghost_pumap_nidle(&availcores,curTask->LD);
        } else {
            ghost_pumap_nidle(&availcores,GHOST_NUMANODE_ANY);
        }
        hwloc_bitmap_t parentscores = hwloc_bitmap_alloc(); 
        if (curTask->parent && !(curTask->parent->flags & GHOST_TASK_NOT_ALLOW_CHILD)) {
            pthread_mutex_lock(curTask->parent->mutex);
            hwloc_bitmap_andnot(parentscores,curTask->parent->coremap,curTask->parent->childusedmap);
            if (curTask->flags & GHOST_TASK_LD_STRICT) {
                hwloc_bitmap_and(parentscores,parentscores,numanode->cpuset);
            }
            availcores += hwloc_bitmap_weight(parentscores);
        }
        if (availcores < curTask->nThreads) {
            DEBUG_LOG(1,"Skipping task %p because it needs %d threads and only %d threads are available",(void *)curTask,curTask->nThreads,availcores);
            hwloc_bitmap_free(parentscores);
            if (curTask->parent && !(curTask->parent->flags & GHOST_TASK_NOT_ALLOW_CHILD)) {
                pthread_mutex_unlock(curTask->parent->mutex);
            }
            pthread_mutex_unlock(curTask->mutex);
            curTask = curTask->next;
            continue;
        }

        DEBUG_LOG(1,"Deleting task itself");
        taskq_deleteTask(q,curTask);    
        DEBUG_LOG(1,"Determining task's threads");
        if( curTask->nThreads > 0 ) {
            ghost_omp_nthread_set(curTask->nThreads);
#ifdef GHOST_HAVE_MKL
            mkl_set_num_threads(curTask->nThreads); 
#endif
        }

        int curThread;
        ghost_pumap_t *pumap;
        ghost_pumap_get(&pumap);


        hwloc_bitmap_t mybusy = hwloc_bitmap_alloc();
        if (curTask->parent && !(curTask->parent->flags & GHOST_TASK_NOT_ALLOW_CHILD)) {
            hwloc_bitmap_andnot(mybusy,pumap->busy,parentscores);
        } else {
            hwloc_bitmap_copy(mybusy,pumap->busy);
        }

        hwloc_bitmap_t myfree = hwloc_bitmap_alloc();
        hwloc_bitmap_andnot(myfree,pumap->cpuset,mybusy);

        hwloc_topology_t topology;
        ghost_topology_get(&topology);
        hwloc_obj_t freepu = NULL;
        freepu = hwloc_get_next_obj_inside_cpuset_by_type(topology,myfree,HWLOC_OBJ_PU,NULL);

        while(freepu) {
            if ((curTask->flags & GHOST_TASK_ONLY_HYPERTHREADS) && 
                    (freepu->sibling_rank == 0)) {
                hwloc_bitmap_clr(myfree,freepu->os_index);
            }
            if ((curTask->flags & GHOST_TASK_NO_HYPERTHREADS) && 
                    (freepu->sibling_rank > 0)) {
                hwloc_bitmap_clr(myfree,freepu->os_index);
            }
            freepu = hwloc_get_next_obj_inside_cpuset_by_type(topology,myfree,HWLOC_OBJ_PU,freepu);
        }


        DEBUG_LOG(1,"Pinning task's threads");
        {
            int curCore = hwloc_bitmap_first(myfree);
            // pin me
            ghost_thread_pin(curCore);
            if( curTask->nThreads > 0 ) {
                int *cores = NULL;
                ghost_malloc((void **)&cores,sizeof(int)*curTask->nThreads);
                for(curThread=0; curThread<curTask->nThreads; curThread++) {
                    cores[curThread] = curCore;
                    hwloc_bitmap_set(mybusy,curCore);
                    curCore = hwloc_bitmap_next(myfree,curCore);
                }
                // pin
            
#pragma omp parallel private(curThread)
                {
                    curThread = ghost_omp_threadnum();
                    DEBUG_LOG(1,"Thread %d (%d): Core # %d is idle, using it",curThread,
                            (int)pthread_self(),cores[curThread]);

                    ghost_thread_pin(cores[curThread]);

//#ifdef GHOST_HAVE_INSTR_LIKWID
//                    likwid_markerThreadInit();
//#endif
                }
                free(cores);
            }
        }

        hwloc_bitmap_or(curTask->coremap,curTask->coremap,mybusy);
        if (curTask->parent && !(curTask->parent->flags & GHOST_TASK_NOT_ALLOW_CHILD)) {
            hwloc_bitmap_or(curTask->parent->childusedmap,curTask->parent->childusedmap,mybusy);
            pthread_mutex_unlock(curTask->parent->mutex);
        }
        ghost_pumap_setbusy(mybusy);



        hwloc_bitmap_free(mybusy);
        hwloc_bitmap_free(parentscores);
        hwloc_bitmap_free(myfree);
        DEBUG_LOG(1,"Pinning successful, returning");

            pthread_mutex_unlock(curTask->mutex);
        return curTask;
    }

    DEBUG_LOG(1,"Could not find and delete a task, returning NULL");
    return NULL;


}

ghost_error_t ghost_taskq_startroutine(void *(**func)(void *))
{
    if (!func) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }
    *func =  &thread_main;

    return GHOST_SUCCESS;
}

/**
 * @brief The main routine of each thread in the thread pool.
 *
 * @param arg The core at which the thread is running.
 *
 * @return NULL 
 */
static void * thread_main(void *arg)
{
#ifdef GHOST_HAVE_CUDA
    ghost_type_t ghost_type;
    ghost_type_get(&ghost_type);
    if (ghost_type == GHOST_TYPE_CUDA) {
        int cu_device;
        ghost_cu_device(&cu_device);
        ghost_cu_init(cu_device);
    }
#endif
    ghost_task_t *myTask = NULL;

    pthread_key_t key;
    ghost_thpool_key(&key);
    int nthreads = (int)(intptr_t)arg;
        
    pthread_mutex_lock(&newTaskMutex_by_threadcount[nthreads]);
    int shepidx = num_shep_by_threadcount[nthreads]-1;
    pthread_mutex_unlock(&newTaskMutex_by_threadcount[nthreads]);

    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE,NULL);
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS,NULL);

    ghost_thpool_t *ghost_thpool = NULL;
    ghost_thpool_get(&ghost_thpool);
    sem_post(ghost_thpool->sem);

    DEBUG_LOG(1,"Shep #%d T%d entering loop.",shepidx,nthreads);


    while (1) // as long as there are jobs stay alive
    {
        pthread_mutex_lock(&newTaskMutex_by_threadcount[nthreads]);
        waiting_shep_by_threadcount[nthreads]++; 
        while(num_tasks_by_threadcount[nthreads] == 0) {
            DEBUG_LOG(1,"No tasks with %d threads --> shep #%d (%d) waiting for them on cond %p",nthreads,shepidx,(int)pthread_self(),(void *)&(newTaskCond_by_threadcount[nthreads][shepidx]));
            pthread_cond_wait(&(newTaskCond_by_threadcount[nthreads][shepidx]),&newTaskMutex_by_threadcount[nthreads]);
            DEBUG_LOG(1,"Shep #%d (%d) woken up by new task with %d threads, actual number: %d",shepidx,(int)pthread_self(),nthreads,num_tasks_by_threadcount[nthreads]);
        }
        num_tasks_by_threadcount[nthreads]--;
        pthread_mutex_unlock(&newTaskMutex_by_threadcount[nthreads]);

        pthread_mutex_lock(&globalMutex);
        if (killed) {
            pthread_mutex_unlock(&globalMutex);
            break;
        }
        pthread_mutex_unlock(&globalMutex);

        pthread_mutex_lock(&taskq->mutex);
        myTask = taskq_findDeleteAndPinTask(taskq,nthreads);
        pthread_mutex_unlock(&taskq->mutex);

        pthread_mutex_lock(&newTaskMutex_by_threadcount[nthreads]);
        waiting_shep_by_threadcount[nthreads]--; 
        pthread_mutex_unlock(&newTaskMutex_by_threadcount[nthreads]);

        if (!myTask) {
            pthread_mutex_lock(&newTaskMutex_by_threadcount[nthreads]);
            num_tasks_by_threadcount[nthreads]++;
            pthread_mutex_unlock(&newTaskMutex_by_threadcount[nthreads]);
            continue;
        } 
        
        pthread_mutex_lock(&newTaskMutex_by_threadcount[nthreads]);
        
        
        DEBUG_LOG(1,"Found task with %d threads. Similar shephs waiting: %d",nthreads,waiting_shep_by_threadcount[nthreads]);
        
        if (waiting_shep_by_threadcount[nthreads] == 0) {
            DEBUG_LOG(1,"Adding another shepherd thread for %d-thread tasks",nthreads);
            num_shep_by_threadcount[nthreads]++;
            newTaskCond_by_threadcount[nthreads] = realloc(newTaskCond_by_threadcount[nthreads],sizeof(pthread_cond_t)*num_shep_by_threadcount[nthreads]);
            pthread_cond_init(&(newTaskCond_by_threadcount[nthreads][num_shep_by_threadcount[nthreads]-1]),NULL);
            void *(*threadFunc)(void *);
            ghost_taskq_startroutine(&threadFunc);
            pthread_mutex_unlock(&newTaskMutex_by_threadcount[nthreads]);
            // protect threadpool (possible reallocation) by global mutex!
            pthread_mutex_lock(&globalMutex);
            ghost_thpool_thread_add(threadFunc,nthreads);
            pthread_mutex_unlock(&globalMutex);
        } else {
            pthread_mutex_unlock(&newTaskMutex_by_threadcount[nthreads]);
        }

        pthread_mutex_lock(myTask->stateMutex);
        myTask->state = GHOST_TASK_RUNNING;    
        pthread_mutex_unlock(myTask->stateMutex);
        

        DEBUG_LOG(1,"Starting exeuction of task %p",(void *)myTask);
        pthread_setspecific(key,myTask);
        myTask->ret = myTask->func(myTask->arg);
        pthread_setspecific(key,NULL);
        DEBUG_LOG(1,"Task %p finished",(void *)myTask);
        
        pthread_mutex_lock(&anyTaskFinishedMutex);
        num_pending_tasks--;
        pthread_mutex_unlock(&anyTaskFinishedMutex);
        pthread_cond_broadcast(&anyTaskFinishedCond);
        
        pthread_mutex_lock(myTask->mutex);
        ghost_task_unpin(myTask);
        pthread_mutex_unlock(myTask->mutex);
        
        pthread_mutex_lock(myTask->stateMutex);
        myTask->state = GHOST_TASK_FINISHED;  
        pthread_cond_broadcast(myTask->finishedCond);
        pthread_mutex_unlock(myTask->stateMutex);


    }
    return NULL;
}


/**
 * @brief Helper function to add a task to a queue
 *
 * @param t The task
 *
 * @return GHOST_SUCCESS on success or GHOST_FAILURE on failure.
 */
ghost_error_t ghost_taskq_add(ghost_task_t *t)
{

    if (taskq==NULL) {
        WARNING_LOG("Tried to add a task to a queue which is NULL");
        return GHOST_ERR_INVALID_ARG;
    }

    pthread_mutex_lock(&taskq->mutex);
    if ((taskq->tail == NULL) || (taskq->head == NULL)) {
        DEBUG_LOG(1,"Adding task %p to empty queue",(void *)t);
        taskq->head = t;
        taskq->tail = t;
        t->next = NULL;
        t->prev = NULL;
    } else {
        if (t->flags & GHOST_TASK_PRIO_HIGH) 
        {
            DEBUG_LOG(1,"Adding high-priority task %p to non-empty queue",(void *)t);
            taskq->head->prev = t;
            t->next = taskq->head;
            t->prev = NULL;
            taskq->head = t;

        } else
        {
            DEBUG_LOG(1,"Adding normal-priority task %p to non-empty queue",(void *)t);
            taskq->tail->next = t;
            t->prev = taskq->tail;
            t->next = NULL;
            taskq->tail = t;
        }
    }
    
    ghost_task_t *cur;
    ghost_task_cur(&cur);
    if (cur) {
        DEBUG_LOG(1,"Adding task from within another task");
    }

    pthread_mutex_lock(&newTaskMutex_by_threadcount[t->nThreads]);

    num_tasks_by_threadcount[t->nThreads]++;

    int s = 0;
    for (;s<num_shep_by_threadcount[t->nThreads];s++) {
        DEBUG_LOG(1,"Sending signal to cond [%d][%d]",t->nThreads,s);
        pthread_cond_signal(&(newTaskCond_by_threadcount[t->nThreads][s]));
    }
    pthread_mutex_unlock(&newTaskMutex_by_threadcount[t->nThreads]);
    pthread_mutex_lock(&anyTaskFinishedMutex);
    num_pending_tasks++;
    pthread_mutex_unlock(&anyTaskFinishedMutex);
    pthread_mutex_unlock(&taskq->mutex);
    
    return GHOST_SUCCESS;
}

/**
 * @brief Execute all outstanding threads and free the task queues' resources
 *
 * @return GHOST_SUCCESS on success or GHOST_FAILURE on failure.
 */
ghost_error_t ghost_taskq_destroy()
{
    if (taskq == NULL) {
        return GHOST_SUCCESS;
    }

    pthread_mutex_lock(&globalMutex);
    killed = 1;
    pthread_mutex_unlock(&globalMutex);

    DEBUG_LOG(1,"Wake up all threads");    

    int t,n;
    for (t=0; t<nthreadcount; t++) {
        pthread_mutex_lock(&newTaskMutex_by_threadcount[t]);
        num_tasks_by_threadcount[t]=num_shep_by_threadcount[t];                    
        for (n=0; n<num_shep_by_threadcount[t]; n++) {
            pthread_cond_signal(&newTaskCond_by_threadcount[t][n]);
        }
        pthread_mutex_unlock(&newTaskMutex_by_threadcount[t]);
    }
    if (taskq) {
        pthread_mutex_destroy(&taskq->mutex);
        free(taskq->head); taskq->head = NULL;
        free(taskq->tail); taskq->tail = NULL;
    }


    DEBUG_LOG(1,"Free task queues");    
    free(taskq); taskq = NULL;

    return GHOST_SUCCESS;    
}

/**
 * @brief Wait for all tasks in all queues to be finished.
 *
 * @return GHOST_SUCCESS on success or GHOST_FAILURE on failure.
 */
ghost_error_t ghost_taskq_waitall()
{
    if (!taskq) {
        return GHOST_SUCCESS;
    }

    int canremain = 0;

    ghost_task_t *cur;
    ghost_task_cur(&cur);
    if (cur) {
        WARNING_LOG("This function has been called inside a task! I will allow one task (this one) to remain active in order to avoid deadlocks.");
        canremain = 1;
    }
    


    pthread_mutex_lock(&anyTaskFinishedMutex);
    while(num_pending_tasks > canremain) {
        pthread_cond_wait(&anyTaskFinishedCond,&anyTaskFinishedMutex);
    }
    pthread_mutex_unlock(&anyTaskFinishedMutex);

    return GHOST_SUCCESS;
}


/**
 * @brief Wait for some tasks out of a given list of tasks.
 *
 * @param tasks The list of task pointers that should be waited for.
 * @param nt The length of the list.
 * @param index Indicating which tasks of the list are now finished.
 *
 * @return GHOST_SUCCESS on success or GHOST_FAILURE on failure.
 */
ghost_error_t ghost_taskq_waitsome(ghost_task_t ** tasks, int nt, int *index)
{
    int t;
    int ret = 0;

    for (t=0; t<nt; t++)
    { // look if one of the tasks is already finished
        pthread_mutex_lock(tasks[t]->stateMutex);
        if (tasks[t]->state == GHOST_TASK_FINISHED) 
        { // one of the tasks is already finished
            DEBUG_LOG(1,"One of the tasks has already finished");
            ret = 1;
            index[t] = 1;
        } else {
            index[t] = 0;
        }
        pthread_mutex_unlock(tasks[t]->stateMutex);
    }
    if (ret)
        return GHOST_SUCCESS;

    pthread_mutex_lock(&anyTaskFinishedMutex);
    pthread_cond_wait(&anyTaskFinishedCond,&anyTaskFinishedMutex);
    pthread_mutex_unlock(&anyTaskFinishedMutex);

    for (t=0; t<nt; t++)
    { // again look which tasks are finished
        pthread_mutex_lock(tasks[t]->stateMutex);
        if (tasks[t]->state == GHOST_TASK_FINISHED) 
        {
            index[t] = 1;
        } else {
            index[t] = 0;
        }
        pthread_mutex_unlock(tasks[t]->stateMutex);
    }

    return GHOST_SUCCESS;
}
