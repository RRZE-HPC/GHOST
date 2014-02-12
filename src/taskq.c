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
#include "ghost/constants.h"
#include "ghost/locality.h"
#include "ghost/task.h"
#include "ghost/taskq.h"
#include "ghost/thpool.h"
#include "ghost/pumap.h"
#include "ghost/util.h"
#include "ghost/machine.h"
#include "ghost/log.h"
#include "ghost/omp.h"

#ifdef GHOST_HAVE_INSTR_LIKWID
#include <likwid.h>
#endif

#ifdef GHOST_HAVE_OPENMP
#include <omp.h>
#endif


extern int ghost_cu_device;

/**
 * @brief The task queue created by ghost_taskq_init().
 */
static ghost_taskq_t *taskq = NULL;


/**
 * @brief Holds the total number of tasks the queue. 
 This semaphore is being waited on by the threads. 
 If a task is added, the first thread to return from wait gets the chance to check if it can execute the new task.
 */
static sem_t taskSem;

static pthread_cond_t newTaskCond;
static pthread_mutex_t newTaskMutex;

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

/**
 * @brief Each of the threads in the thread pool gets assigned the task it executes via pthread_setspecific. 
 This is the key to this specific data. It is exported in ghost_taskq.h
 */
pthread_key_t ghost_thread_key = 0;


static void * thread_main(void *arg);

ghost_error_t ghost_taskq_create()
{
    GHOST_CALL_RETURN(ghost_malloc((void **)&taskq,sizeof(ghost_taskq_t)));

    taskq->tail = NULL;
    taskq->head = NULL;
    pthread_mutex_init(&(taskq->mutex),NULL);

    pthread_key_create(&ghost_thread_key,NULL);
    pthread_cond_init(&newTaskCond,NULL);
    pthread_mutex_init(&newTaskMutex,NULL);
    pthread_mutex_init(&globalMutex,NULL);
    sem_init(&taskSem, 0, 0);
    pthread_cond_init(&anyTaskFinishedCond,NULL);
    pthread_mutex_init(&anyTaskFinishedMutex,NULL);
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
    pthread_mutex_lock(&q->mutex);
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

    pthread_mutex_unlock(&q->mutex);

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
static ghost_task_t * taskq_findDeleteAndPinTask(ghost_taskq_t *q)
{
    if (q == NULL) {
        WARNING_LOG("Tried to find a job but the queue is NULL");
        return NULL;
    }
    if (q->head == NULL) {
        DEBUG_LOG(1,"Empty queue, returning NULL!");
        return NULL;
    }

    ghost_thpool_t *ghost_thpool = NULL;
    ghost_thpool_get(&ghost_thpool);

    ghost_task_t *curTask = q->head;

    while(curTask != NULL)
    {
        if (curTask->flags & GHOST_TASK_NO_PIN) {
            taskq_deleteTask(q,curTask);    
            ghost_ompSetNumThreads(curTask->nThreads);
            return curTask;
        }

        hwloc_obj_t numaNode;
        ghost_getNumaNode(&numaNode,curTask->LD);

        int availcores = 0;
        if (curTask->flags & GHOST_TASK_LD_STRICT) {
            ghost_pumap_getNumberOfIdlePUs(&availcores,curTask->LD);
        } else {
            ghost_pumap_getNumberOfIdlePUs(&availcores,GHOST_NUMANODE_ANY);
        }
        hwloc_bitmap_t parentscores = hwloc_bitmap_alloc(); // TODO free
        if ((curTask->flags & GHOST_TASK_USE_PARENTS) && curTask->parent) {
            //char *a, *b, *c;
            //hwloc_bitmap_list_asprintf(&a,curTask->parent->coremap);
            //hwloc_bitmap_list_asprintf(&b,curTask->parent->childusedmap);

            hwloc_bitmap_andnot(parentscores,curTask->parent->coremap,curTask->parent->childusedmap);
            //hwloc_bitmap_list_asprintf(&c,parentscores);
            //WARNING_LOG("(%lu) %s = %s andnot %s (%p)",(unsigned long)pthread_self(),c,a,b,curTask->parent->childusedmap);
            if (curTask->flags & GHOST_TASK_LD_STRICT) {
                hwloc_bitmap_and(parentscores,parentscores,numaNode->cpuset);
            }
            availcores += hwloc_bitmap_weight(parentscores);
        }
        if (availcores < curTask->nThreads) {
            DEBUG_LOG(1,"Skipping task %p because it needs %d threads and only %d threads are available",(void *)curTask,curTask->nThreads,availcores);
            curTask = curTask->next;
            continue;
        }
        /*        if ((curTask->flags & GHOST_TASK_LD_STRICT) && (nIdleCoresAtLD(ghost_thpool->busy,curTask->LD) < curTask->nThreads)) {
                  DEBUG_LOG(1,"Skipping task %p because there are not enough idle cores at its strict LD %d: %d < %d",curTask,curTask->LD,nIdleCoresAtLD(ghost_thpool->busy,curTask->LD),curTask->nThreads);
                  curTask = curTask->next;
                  continue;
                  }
                  if (NIDLECORES < curTask->nThreads) {
                  DEBUG_LOG(1,"Skipping task %p because it needs %d threads and only %d threads are idle",curTask,curTask->nThreads,NIDLECORES);
                  curTask = curTask->next;
                  continue;
                  }*/

//        DEBUG_LOG(1,"Thread %d: Found a suiting task: %p! task->nThreads=%d, nIdleCores[LD%d]=%d, nIdleCores=%d",(int)pthread_self(),(void *)curTask,curTask->nThreads,curTask->LD,nIdleCoresAtLD(curTask->LD),nIdleCores());

        DEBUG_LOG(1,"Deleting task itself");
        taskq_deleteTask(q,curTask);    
        DEBUG_LOG(1,"Pinning the task's threads");
        ghost_ompSetNumThreads(curTask->nThreads);

        //        if (curTask->flags & GHOST_TASK_NO_PIN) {
        //            return curTask;
        //        }

        int reservedCores = 0;

        int t = 0;
        int curThread;
        ghost_pumap_t *pumap;
        ghost_pumap_get(&pumap);


        hwloc_bitmap_t mybusy = hwloc_bitmap_alloc();
        if ((curTask->flags & GHOST_TASK_USE_PARENTS) && curTask->parent) {
            /* char *a, *b;
               hwloc_bitmap_list_asprintf(&a,ghost_thpool->busy);
               hwloc_bitmap_list_asprintf(&b,parentscores);
               INFO_LOG("Need %d cores, available cores: %d (busy %s) + %d from parent (free in parent %s)",curTask->nThreads,NIDLECORES,a,hwloc_bitmap_weight(parentscores),b);*/
            hwloc_bitmap_andnot(mybusy,pumap->busy,parentscores);
        } else {
            hwloc_bitmap_copy(mybusy,pumap->busy);
        }

#pragma omp parallel
        { 

#pragma omp for ordered 
            for (curThread=0; curThread<curTask->nThreads; curThread++) {
#pragma omp ordered
                for (; t<ghost_thpool->nThreads; t++) {
                    int core = pumap->PUs[curTask->LD][t]->os_index;
                    //core = coreIdx(curTask->LD,t);
                    if ((curTask->flags & GHOST_TASK_ONLY_HYPERTHREADS) && 
                            (pumap->PUs[curTask->LD][t]->sibling_rank == 0)) {
                        //    WARNING_LOG("only HT");
                        continue;
                    }
                    if ((curTask->flags & GHOST_TASK_NO_HYPERTHREADS) && 
                            (pumap->PUs[curTask->LD][t]->sibling_rank > 0)) {
                        //    WARNING_LOG("no HT");
                        continue;
                    }


                    //if (!hwloc_bitmap_isset(ghost_thpool->busy,core)) {
                    if (!hwloc_bitmap_isset(mybusy,core)) {
                        DEBUG_LOG(1,"Thread %d (%d): Core # %d is idle, using it",ghost_ompGetThreadNum(),
                                (int)pthread_self(),core);

                       //                            hwloc_bitmap_set(ghost_thpool->busy,core);
                        hwloc_bitmap_set(mybusy,core);
                        DEBUG_LOG(2,"Pinning thread %lu to core %d",(unsigned long)pthread_self(),pumap->PUs[curTask->LD][t]->os_index);
                        ghost_setCore(core);
                        hwloc_bitmap_set(curTask->coremap,core);
                        curTask->cores[reservedCores] = core;
                        reservedCores++;
                        t++;
                        break;
                    }
                }
                }
#ifdef GHOST_HAVE_INSTR_LIKWID
                LIKWID_MARKER_THREADINIT;
#endif
            }
            if ((curTask->flags & GHOST_TASK_USE_PARENTS) && curTask->parent) {
                //char *a;
                //hwloc_bitmap_list_asprintf(&a,curTask->parent->childusedmap);
                //WARNING_LOG("### %p %s",curTask->parent->childusedmap,a);
                hwloc_bitmap_or(curTask->parent->childusedmap,curTask->parent->childusedmap,mybusy);
                //hwloc_bitmap_list_asprintf(&a,curTask->parent->childusedmap);
                //WARNING_LOG("### %p %s",curTask->parent->childusedmap,a);
            }
            ghost_pumap_setBusy(mybusy);
//            hwloc_bitmap_or(pumap->busy,pumap->busy,mybusy);

            if (reservedCores < curTask->nThreads) {
                WARNING_LOG("Too few cores reserved! %d < %d This should not have happened...",reservedCores,curTask->nThreads);
            }

            DEBUG_LOG(1,"Pinning successful, returning");

            return curTask;
        }

        DEBUG_LOG(1,"Could not find and delete a task, returning NULL");
        return NULL;


    }

ghost_error_t ghost_taskq_getStartRoutine(void *(**func)(void *))
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
        ghost_cu_init(ghost_cu_device);
#endif
        //    kmp_set_blocktime(200);
        //    kmp_set_library_throughput();
        //    UNUSED(arg);
        ghost_task_t *myTask;

        ghost_thpool_t *ghost_thpool = NULL;
        ghost_thpool_get(&ghost_thpool);
        sem_post(ghost_thpool->sem);

        DEBUG_LOG(1,"Shepherd thread %lu in thread_main() called with %"PRIdPTR,(unsigned long)pthread_self(), (intptr_t)arg);
        while (1) // as long as there are jobs stay alive
        {
            // TODO wait for condition when unpinned or new task
            if (sem_wait(&taskSem)) // TODO wait for a signal in order to avoid entering the loop when nothing has changed
            {
                if (errno == EINTR) {
                    continue;
                }
                ERROR_LOG("Waiting for tasks failed: %s. Will try again.",strerror(errno));
                continue;
            }

            pthread_mutex_lock(&globalMutex);
            if (killed) // thread has been woken by the finish() function
            {
                pthread_mutex_unlock(&globalMutex);
                DEBUG_LOG(2,"Thread %d: Not executing any further tasks",(int)pthread_self());
                sem_post(&taskSem); // wake up another thread
                break;
            }
            pthread_mutex_unlock(&globalMutex);

            //    WARNING_LOG("1 %d : %d",(intptr_t)arg,kmp_get_blocktime());
            //    kmp_set_blocktime((intptr_t)arg);
            //    WARNING_LOG("2 %d : %d",(intptr_t)arg,kmp_get_blocktime());

            pthread_mutex_lock(&newTaskMutex);
            pthread_mutex_lock(&globalMutex);
            myTask = taskq_findDeleteAndPinTask(taskq);
            pthread_mutex_unlock(&globalMutex);

            if (myTask  == NULL) // no suiting task found
            {
                DEBUG_LOG(1,"Thread %d: Could not find a suited task in any queue",(int)pthread_self());
                pthread_cond_wait(&newTaskCond,&newTaskMutex);
                pthread_mutex_unlock(&newTaskMutex);
                sem_post(&taskSem);
                continue;
            }
            pthread_mutex_unlock(&newTaskMutex);

            pthread_mutex_lock(myTask->mutex);
            myTask->state = GHOST_TASK_RUNNING;    
            pthread_mutex_unlock(myTask->mutex);

            DEBUG_LOG(1,"Thread %d: Finally executing task %p",(int)pthread_self(),(void *)myTask);

            pthread_setspecific(ghost_thread_key,myTask);

#ifdef __INTEL_COMPILER
            //kmp_set_blocktime(0);
#endif
            myTask->ret = myTask->func(myTask->arg);
            //    WARNING_LOG("2 %d : %d",(intptr_t)arg,kmp_get_blocktime());
#ifdef __INTEL_COMPILER
            //kmp_set_blocktime(200);
#endif
            pthread_setspecific(ghost_thread_key,NULL);

            DEBUG_LOG(1,"Thread %lu: Finished executing task: %p. Free'ing resources and waking up another thread"
                    ,(unsigned long)pthread_self(),(void *)myTask);

            pthread_mutex_lock(&globalMutex);
            ghost_task_unpin(myTask);
            pthread_mutex_unlock(&globalMutex);

            //    kmp_set_blocktime(200);
            pthread_mutex_lock(&newTaskMutex);
            pthread_cond_broadcast(&newTaskCond);
            pthread_mutex_unlock(&newTaskMutex);

            pthread_mutex_lock(myTask->mutex); 
            DEBUG_LOG(1,"Thread %d: Finished with task %p. Setting state to finished...",(int)pthread_self(),(void *)myTask);
            myTask->state = GHOST_TASK_FINISHED;
            pthread_cond_broadcast(myTask->finishedCond);
            pthread_mutex_unlock(myTask->mutex);
            DEBUG_LOG(1,"Thread %d: Finished with task %p. Sending signal to all waiters (cond: %p).",(int)pthread_self(),(void *)myTask,(void *)myTask->finishedCond);
        
            pthread_mutex_lock(&anyTaskFinishedMutex);
            pthread_cond_broadcast(&anyTaskFinishedCond);
            pthread_mutex_unlock(&anyTaskFinishedMutex);
            
            pthread_mutex_lock(&globalMutex);
            if (killed) // exit loop
            {
                pthread_mutex_unlock(&globalMutex);
                break;
            }
            pthread_mutex_unlock(&globalMutex);
        }
        return NULL;
    }


    /**
     * @brief Helper function to add a task to a queue
     *
     * @param q The queue
     * @param t The task
     *
     * @return GHOST_SUCCESS on success or GHOST_FAILURE on failure.
     */
ghost_error_t ghost_taskq_addTask(ghost_task_t *t)
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
        pthread_mutex_unlock(&taskq->mutex);

        sem_post(&taskSem);
        pthread_mutex_lock(&newTaskMutex);
        pthread_cond_broadcast(&newTaskCond);
        pthread_mutex_unlock(&newTaskMutex);

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
        if (sem_post(&taskSem)){
            WARNING_LOG("Error in sem_post: %s",strerror(errno));
            return GHOST_ERR_UNKNOWN;
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
        ghost_task_t *t;

        pthread_mutex_lock(&globalMutex);
        t = taskq->head;
        pthread_mutex_unlock(&globalMutex);
        while (t != NULL)
        {
            DEBUG_LOG(1,"Waitall: Waiting for task %p",(void *)t);
            ghost_task_wait(t);
            t = t->next;
        }
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
        pthread_t threads[nt];

        for (t=0; t<nt; t++)
        { // look if one of the tasks is already finished
            pthread_mutex_lock(tasks[t]->mutex);
            if (tasks[t]->state == GHOST_TASK_FINISHED) 
            { // one of the tasks is already finished
                DEBUG_LOG(1,"One of the tasks has already finished");
                ret = 1;
                index[t] = 1;
            } else {
                index[t] = 0;
            }
            pthread_mutex_unlock(tasks[t]->mutex);
        }
        if (ret)
            return GHOST_SUCCESS;

        DEBUG_LOG(1,"None of the tasks has already finished. Waiting for (at least) one of them...");


        for (t=0; t<nt; t++)
        {
            pthread_create(&threads[t],NULL,(void *(*)(void *))&ghost_task_wait,tasks[t]);
        }

        pthread_mutex_lock(&anyTaskFinishedMutex);
        pthread_cond_wait(&anyTaskFinishedCond,&anyTaskFinishedMutex);
        pthread_mutex_unlock(&anyTaskFinishedMutex);

        for (t=0; t<nt; t++)
        { // again look which tasks are finished
            pthread_mutex_lock(tasks[t]->mutex);
            if (tasks[t]->state == GHOST_TASK_FINISHED) 
            {
                index[t] = 1;
            } else {
                index[t] = 0;
            }
            pthread_mutex_unlock(tasks[t]->mutex);
        }

        return GHOST_SUCCESS;
    }
