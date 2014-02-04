/**
 * @file ghost_taskq.c
 * @author Moritz Kreutzer (moritz.kreutzer@fau.de)
 * @date August 2013
 *
 * In this file, the task queue functionality of GHOST is implemented.
 */

#define _XOPEN_SOURCE 500
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>
#include <errno.h>
#include <unistd.h>

#define NIDLECORES (ghost_thpool->nThreads-hwloc_bitmap_weight(ghost_thpool->busy))

#include "ghost/constants.h"
#include "ghost/affinity.h"
#include "ghost/task.h"
#include "ghost/util.h"

#if GHOST_HAVE_OPENMP
#include <omp.h>
#endif


extern int ghost_cu_device;

/**
 * @brief The task queue created by ghost_taskq_init().
 */
static ghost_taskq_t *taskq = NULL;

/**
 * @brief The thread pool created by ghost_thpool_init(). This variable is exported in ghost_taskq.h
 */
ghost_thpool_t *ghost_thpool = NULL;

/**
 * @brief Holds the total number of tasks in all queues. 
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

static int** coreidx;

//static void * thread_main(void *arg);
static void * thread_main(void *arg);
static int ghost_task_unpin(ghost_task_t *task);
static int nIdleCoresAtLD(hwloc_bitmap_t, int);
static int nBusyCoresAtLD(hwloc_bitmap_t, int);
static int nThreadsPerLD(int ld);
static int firstThreadOfLD(int ld);

/**
 * @brief Compare two integers
 *
 * @param x
 * @param y
 *
 * @return 
 */
static int intcomp(const void *x, const void *y) 
{
    return (*(int *)x - *(int *)y);
}

/**
 * @brief Initializes a thread pool with a given number of threads.
 * @param nThreads
 * @return GHOST_SUCCESS on success or GHOST_FAILURE on failure.
 * 
 * A number of pthreads will be created and each one will have thread_main() as start routine.
 * In order to make sure that each thread has entered the infinite loop, a wait on a semaphore is
 * performed before this function returns.
 */
ghost_error_t ghost_thpool_init(hwloc_cpuset_t cpuset)
{
    static int initialized = 0;
    int t,q,i;
    int totalThreads;
    hwloc_obj_t obj;

    if (initialized) {
        WARNING_LOG("The thread pool has already been initialized.");
        return GHOST_ERR_UNKNOWN;
    }
    if (!cpuset) {
        WARNING_LOG("The thread pool's cpuset is NULL.");
        return GHOST_ERR_INVALID_ARG;
    }
    initialized=1;

    totalThreads = hwloc_bitmap_weight(cpuset);

    ghost_thpool = (ghost_thpool_t*)ghost_malloc(sizeof(ghost_thpool_t));
    ghost_thpool->PUs = (hwloc_obj_t *)ghost_malloc(totalThreads*sizeof(hwloc_obj_t));
    ghost_thpool->nThreads = totalThreads;
    ghost_thpool->threads = (pthread_t *)ghost_malloc(ghost_thpool->nThreads*sizeof(pthread_t));
    ghost_thpool->sem = (sem_t*)ghost_malloc(sizeof(sem_t));
    sem_init(ghost_thpool->sem, 0, 0);
    sem_init(&taskSem, 0, 0);
    pthread_cond_init(&newTaskCond,NULL);
    pthread_mutex_init(&newTaskMutex,NULL);
    pthread_mutex_init(&globalMutex,NULL);

    pthread_key_create(&ghost_thread_key,NULL);

    ghost_thpool->cpuset = hwloc_bitmap_alloc();
    hwloc_bitmap_copy(ghost_thpool->cpuset,cpuset);
    ghost_thpool->busy = hwloc_bitmap_alloc();

    int cpu;
    hwloc_obj_t runner;

    int npus = hwloc_get_nbobjs_inside_cpuset_by_type(topology,ghost_thpool->cpuset,HWLOC_OBJ_PU);

    for (i=0; i<npus; i++) {
        ghost_thpool->PUs[i] = hwloc_get_obj_inside_cpuset_by_type(topology,ghost_thpool->cpuset,HWLOC_OBJ_PU,i);
    }

    int nodes[totalThreads];
    for (i=0; i<totalThreads; i++) {
        nodes[i] = 0;
        obj = ghost_thpool->PUs[i];
        for (runner=obj; runner; runner=runner->parent) {
            if (!hwloc_compare_types(runner->type,HWLOC_OBJ_NODE)) {
                nodes[i] = runner->logical_index;
                break;
            }
        }
        //INFO_LOG("Thread # %3d running @ PU %3u (OS: %3u), SMT level %2d, NUMA node %u",i,obj->logical_index,obj->os_index,obj->sibling_rank,nodes[i]);
    }

    qsort(nodes,totalThreads,sizeof(int),intcomp);
    ghost_thpool->nLDs = 1;

    for (i=1; i<totalThreads; i++) {
        if (nodes[i] != nodes[i-1]) {
            ghost_thpool->nLDs++;
        }
    }

    // holds core indices sorted for each locality domain
    coreidx = (int **)ghost_malloc(sizeof(int *)*ghost_thpool->nLDs);

    int li;
    // sort the cores according to the locality domain
    for (q=0; q<ghost_thpool->nLDs; q++) {
        int localthreads = nThreadsPerLD(q);
        coreidx[q] = (int *)ghost_malloc(sizeof(int)*ghost_thpool->nThreads);

        for (t=0; t<localthreads; t++) { // my own threads
            coreidx[q][t] = firstThreadOfLD(q)+t;
            //WARNING_LOG("1 coreidx[%d][%d] = %d",q,t,coreidx[q][t]);
        }
        for (li=0; t-localthreads<firstThreadOfLD(q); t++, li++) { // earlier LDs
            coreidx[q][t] = li;
            //WARNING_LOG("2 coreidx[%d][%d] = %d",q,t,coreidx[q][t]);
        }    
        for (; t<ghost_thpool->nThreads; t++) { // later LDs
            coreidx[q][t] = t;
            //WARNING_LOG("3 coreidx[%d][%d] = %d",q,t,coreidx[q][t]);
        }    
    }

    //    WARNING_LOG("Creating %d threads for the thread pool",ghost_thpool->nThreads);
    for (t=0; t<ghost_thpool->nThreads; t++){
        pthread_create(&(ghost_thpool->threads[t]), NULL, thread_main, (void *)(intptr_t)t);
    }
    for (t=0; t<ghost_thpool->nThreads; t++){
        sem_wait(ghost_thpool->sem);
    }
    DEBUG_LOG(1,"All threads are initialized and waiting for tasks");

    return GHOST_SUCCESS;
}

static int nThreadsPerLD(int ld)
{
    int i,n = 0;
    hwloc_obj_t obj,runner;
    for (i=0; i<ghost_thpool->nThreads; i++) {    
        obj = ghost_thpool->PUs[i];
        for (runner=obj; runner; runner=runner->parent) {
            if (runner->type <= HWLOC_OBJ_NODE) {
                if ((int)runner->logical_index == ld) {
                    n++;
                }
                break;
            }
        }
    }

    return n;
}

static int nIdleCoresAtLD(hwloc_bitmap_t bm, int ld)
{
    int begin = firstThreadOfLD(ld), end = begin+nThreadsPerLD(ld);
    hwloc_bitmap_t LDBM = hwloc_bitmap_alloc();

    hwloc_bitmap_set_range(LDBM,begin,end-1);
    hwloc_bitmap_t idle = hwloc_bitmap_alloc();
    hwloc_bitmap_andnot(idle,LDBM,bm);

    int w = hwloc_bitmap_weight(idle);

    hwloc_bitmap_free(LDBM);
    hwloc_bitmap_free(idle);
    return w;
}

static int nBusyCoresAtLD(hwloc_bitmap_t bm, int ld)
{
    int begin = firstThreadOfLD(ld), end = begin+nThreadsPerLD(ld);
    hwloc_bitmap_t LDBM = hwloc_bitmap_alloc();

    hwloc_bitmap_set_range(LDBM,begin,end-1);
    hwloc_bitmap_t busy = hwloc_bitmap_alloc();
    hwloc_bitmap_and(busy,LDBM,bm);

    int w = hwloc_bitmap_weight(busy);

    hwloc_bitmap_free(LDBM);
    hwloc_bitmap_free(busy);
    return w;
}

static int firstThreadOfLD(int ld)
{
    int i;
    hwloc_obj_t obj,runner;

    for (i=0; i<ghost_thpool->nThreads; i++) {    
        obj = ghost_thpool->PUs[i];
        for (runner=obj; runner; runner=runner->parent) {
            if (runner->type <= HWLOC_OBJ_NODE) {
                if ((int)runner->logical_index == ld) {
                    return i;
                }
            }
        }
    }

    return -1;
}


/**
 * @brief Initializes a task queues.
 * @return GHOST_SUCCESS on success or GHOST_FAILURE on failure.
 */
ghost_error_t ghost_taskq_init()
{
    taskq=(ghost_taskq_t *)ghost_malloc(sizeof(ghost_taskq_t));

    taskq->tail = NULL;
    taskq->head = NULL;
    pthread_mutex_init(&(taskq->mutex),NULL);

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
        DEBUG_LOG(1,"Removing head from queue %p",q);
        q->head = t->next;
        if (q->head != NULL)
            q->head->prev = NULL;
    }
    if (t == q->tail) {
        DEBUG_LOG(1,"Removing tail from queue %p",q);
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


    ghost_task_t *curTask = q->head;

    while(curTask != NULL)
    {
        if (curTask->flags & GHOST_TASK_NO_PIN) {
            taskq_deleteTask(q,curTask);    
            ghost_ompSetNumThreads(curTask->nThreads);
            return curTask;
        }


        int availcores = 0;
        if (curTask->flags & GHOST_TASK_LD_STRICT) {
            availcores = nIdleCoresAtLD(ghost_thpool->busy,curTask->LD);
        } else {
            availcores = NIDLECORES;
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
                //availcores += nBusyCoresAtLD(curTask->parent->coremap,curTask->LD);
                availcores += nBusyCoresAtLD(parentscores,curTask->LD);
            } else {
                //availcores += hwloc_bitmap_weight(curTask->parent->coremap);
                availcores += hwloc_bitmap_weight(parentscores);
            }
        } else { 
        }
        if (availcores < curTask->nThreads) {
            DEBUG_LOG(1,"Skipping task %p because it needs %d threads and only %d threads are available",curTask,curTask->nThreads,availcores);
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

        DEBUG_LOG(1,"Thread %d: Found a suiting task: %p! task->nThreads=%d, nIdleCores[LD%d]=%d, nIdleCores=%d",(int)pthread_self(),curTask,curTask->nThreads,curTask->LD,nIdleCoresAtLD(ghost_thpool->busy,curTask->LD),NIDLECORES);

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

        hwloc_bitmap_t mybusy = hwloc_bitmap_alloc();
        if ((curTask->flags & GHOST_TASK_USE_PARENTS) && curTask->parent) {
            /* char *a, *b;
               hwloc_bitmap_list_asprintf(&a,ghost_thpool->busy);
               hwloc_bitmap_list_asprintf(&b,parentscores);
               INFO_LOG("Need %d cores, available cores: %d (busy %s) + %d from parent (free in parent %s)",curTask->nThreads,NIDLECORES,a,hwloc_bitmap_weight(parentscores),b);*/
            hwloc_bitmap_andnot(mybusy,ghost_thpool->busy,parentscores);
        } else {
            hwloc_bitmap_copy(mybusy,ghost_thpool->busy);
        }

#pragma omp parallel
        { 

#pragma omp for ordered 
            for (curThread=0; curThread<curTask->nThreads; curThread++) {
#pragma omp ordered
                for (; t<ghost_thpool->nThreads; t++) {
                    int core = coreidx[curTask->LD][t];
                    if ((curTask->flags & GHOST_TASK_ONLY_HYPERTHREADS) && 
                            (ghost_thpool->PUs[core]->sibling_rank == 0)) {
                        //    WARNING_LOG("only HT");
                        continue;
                    }
                    if ((curTask->flags & GHOST_TASK_NO_HYPERTHREADS) && 
                            (ghost_thpool->PUs[core]->sibling_rank > 0)) {
                        //    WARNING_LOG("no HT");
                        continue;
                    }


                    //if (!hwloc_bitmap_isset(ghost_thpool->busy,core)) {
                    if (!hwloc_bitmap_isset(mybusy,core)) {
                        DEBUG_LOG(1,"Thread %d (%d): Core # %d is idle, using it",ghost_ompGetThreadNum(),
                                (int)pthread_self(),core);

                        //                            hwloc_bitmap_set(ghost_thpool->busy,core);
                        hwloc_bitmap_set(mybusy,core);
                        DEBUG_LOG(2,"Pinning thread %lu to core %d",(unsigned long)pthread_self(),ghost_thpool->PUs[core]->os_index);
                        ghost_setCore(ghost_thpool->PUs[core]->os_index);
                        hwloc_bitmap_set(curTask->coremap,core);
                        curTask->cores[reservedCores] = core;
                        reservedCores++;
                        t++;
                        break;
                    }
                }
                }
#if GHOST_HAVE_INSTR_LIKWID
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
            hwloc_bitmap_or(ghost_thpool->busy,ghost_thpool->busy,mybusy);

            if (reservedCores < curTask->nThreads) {
                WARNING_LOG("Too few cores reserved! %d < %d This should not have happened...",reservedCores,curTask->nThreads);
            }

            DEBUG_LOG(1,"Pinning successful, returning");

            return curTask;
        }

        DEBUG_LOG(1,"Could not find and delete a task, returning NULL");
        return NULL;


    }

    /**
     * @brief The main routine of each thread in the thread pool.
     *
     * @param arg The core at which the thread is running.
     *
     * @return NULL 
     */
    void * thread_main(void *arg)
    {
#if GHOST_HAVE_CUDA
        ghost_CUDA_init(ghost_cu_device);
#endif
        //    kmp_set_blocktime(200);
        //    kmp_set_library_throughput();
        //    UNUSED(arg);
        ghost_task_t *myTask;

        sem_post(ghost_thpool->sem);

        DEBUG_LOG(1,"Shepherd thread %lu in thread_main() called with %"PRIdPTR,(unsigned long)pthread_self(), (intptr_t)arg);
        while (!killed) // as long as there are jobs stay alive
        {
            // TODO wait for condition when unpinned or new task
            if (sem_wait(&taskSem)) // TODO wait for a signal in order to avoid entering the loop when nothing has changed
            {
                if (errno == EINTR)
                    continue;
                ABORT("Waiting for tasks failed: %s",strerror(errno));
            }

            if (killed) // thread has been woken by the finish() function
            {
                DEBUG_LOG(2,"Thread %d: Not executing any further tasks",(int)pthread_self());
                sem_post(&taskSem); // wake up another thread
                break;
            }

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
            *(myTask->state) = GHOST_TASK_RUNNING;    
            pthread_mutex_unlock(myTask->mutex);


            DEBUG_LOG(1,"Thread %d: Finally executing task at core %d: %p",(int)pthread_self(),ghost_getCore(),myTask);

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
                    ,(unsigned long)pthread_self(),myTask);

            pthread_mutex_lock(&globalMutex);
            ghost_task_unpin(myTask);
            pthread_mutex_unlock(&globalMutex);

            //    kmp_set_blocktime(200);
            pthread_mutex_lock(&newTaskMutex);
            pthread_cond_broadcast(&newTaskCond);
            pthread_mutex_unlock(&newTaskMutex);

            pthread_mutex_lock(myTask->mutex); 
            DEBUG_LOG(1,"Thread %d: Finished with task %p. Setting state to finished...",(int)pthread_self(),myTask);
            *(myTask->state) = GHOST_TASK_FINISHED;
            pthread_cond_broadcast(myTask->finishedCond);
            pthread_mutex_unlock(myTask->mutex);
            DEBUG_LOG(1,"Thread %d: Finished with task %p. Sending signal to all waiters (cond: %p).",(int)pthread_self(),myTask,myTask->finishedCond);
        }
        return NULL;
    }


    static int ghost_task_unpin(ghost_task_t *task)
    {
        if (!(task->flags & GHOST_TASK_NO_PIN)) {
            for (int t=0; t<task->nThreads; t++) {
                if ((task->flags & GHOST_TASK_USE_PARENTS) && 
                        task->parent && 
                        hwloc_bitmap_isset(task->parent->coremap,task->cores[t])) 
                {
                    hwloc_bitmap_clr(task->parent->childusedmap,task->cores[t]);
                } else {
                    hwloc_bitmap_clr(ghost_thpool->busy,task->cores[t]);
                }
            }
        }
        //    task->freed = 1;

        return GHOST_SUCCESS;


    }

    /**
     * @brief Print a task and all relevant informatio to stdout.
     *
     * @param t The task
     *
     * @return GHOST_SUCCESS on success or GHOST_FAILURE on failure.
     */
    ghost_error_t ghost_task_print(ghost_task_t *t) 
    {
        ghost_printHeader("Task %p",t);
        ghost_printLine("No. of threads",NULL,"%d",t->nThreads);
        ghost_printLine("LD",NULL,"%d",t->LD);
        ghost_printFooter();

        return GHOST_SUCCESS;
    }

    /**
     * @brief Print all tasks of all queues. 
     *
     * @return GHOST_SUCCESS on success or GHOST_FAILURE on failure.
     */
    ghost_error_t ghost_taskq_print_all() 
    {
        ghost_task_t *t;

        pthread_mutex_lock(&taskq->mutex);
        ghost_printHeader("Task queue");

        t = taskq->head;
        while (t != NULL)
        {
            printf("%p ",t);
            t=t->next;
        }
        printf("\n");
        ghost_printFooter();
        pthread_mutex_unlock(&taskq->mutex);
        return GHOST_SUCCESS;
    }


    /**
     * @brief Helper function to add a task to a queue
     *
     * @param q The queue
     * @param t The task
     *
     * @return GHOST_SUCCESS on success or GHOST_FAILURE on failure.
     */
    static int taskq_additem(ghost_taskq_t *q, ghost_task_t *t)
    {

        if (q==NULL) {
            WARNING_LOG("Tried to add a task to a queue which is NULL");
            return GHOST_ERR_INVALID_ARG;
        }

        pthread_mutex_lock(&q->mutex);
        if ((q->tail == NULL) || (q->head == NULL)) {
            DEBUG_LOG(1,"Adding task %p to empty queue",t);
            q->head = t;
            q->tail = t;
            t->next = NULL;
            t->prev = NULL;
        } else {
            if (t->flags & GHOST_TASK_PRIO_HIGH) 
            {
                DEBUG_LOG(1,"Adding high-priority task %p to non-empty queue",t);
                q->head->prev = t;
                t->next = q->head;
                t->prev = NULL;
                q->head = t;

            } else
            {
                DEBUG_LOG(1,"Adding normal-priority task %p to non-empty queue",t);
                q->tail->next = t;
                t->prev = q->tail;
                t->next = NULL;
                q->tail = t;
            }
        }
        pthread_mutex_unlock(&q->mutex);


        return GHOST_SUCCESS;
    }

    /**
     * @brief Execute all outstanding threads and free the task queues' resources
     *
     * @return GHOST_SUCCESS on success or GHOST_FAILURE on failure.
     */
    ghost_error_t ghost_task_add(ghost_task_t *t)
    {
        // if a task is initialized _once_ but added several times, this has to be done each time it is added
        pthread_cond_init(t->finishedCond,NULL);
        pthread_mutex_init(t->mutex,NULL);
        *(t->state) = GHOST_TASK_INVALID;
        memset(t->cores,0,sizeof(int)*t->nThreads);

        hwloc_bitmap_zero(t->coremap);
        hwloc_bitmap_zero(t->childusedmap);
        t->parent = (ghost_task_t *)pthread_getspecific(ghost_thread_key);
        //    t->freed = 0;

        DEBUG_LOG(1,"Task %p w/ %d threads goes to queue %p (LD %d)",t,t->nThreads,taskq,t->LD);
        taskq_additem(taskq,t);
        //ghost_task_destroy(&commTask);
        *(t->state) = GHOST_TASK_ENQUEUED;

        sem_post(&taskSem);
        pthread_mutex_lock(&newTaskMutex);
        pthread_cond_broadcast(&newTaskCond);
        pthread_mutex_unlock(&newTaskMutex);

        DEBUG_LOG(1,"Task added successfully");

        return GHOST_SUCCESS;
    }

    /**
     * @brief Execute all outstanding threads and free the task queues' resources
     *
     * @return GHOST_SUCCESS on success or GHOST_FAILURE on failure.
     */
    ghost_error_t ghost_taskq_finish()
    {
        DEBUG_LOG(1,"Finishing task queue");
        if (taskq == NULL)
            return GHOST_SUCCESS;

        ghost_task_waitall(); // finish all outstanding tasks
        pthread_mutex_lock(&globalMutex);
        killed = 1;
        pthread_mutex_unlock(&globalMutex);

        DEBUG_LOG(1,"Wake up all threads");    
        if (sem_post(&taskSem)){
            WARNING_LOG("Error in sem_post: %s",strerror(errno));
            return GHOST_ERR_INTERNAL;
        }
        /*DEBUG_LOG(1,"Join all threads");
          int t; 
          for (t=0; t<ghost_thpool->nThreads; t++)
          {         
          if (pthread_join(ghost_thpool->threads[t],NULL)){
          return GHOST_FAILURE;
          }
          }*/

        DEBUG_LOG(1,"Free task queues");    
        free(taskq);

        return GHOST_SUCCESS;    
    }

    /**
     * @brief Test the task's current state
     *
     * @param t The task to test
     *
     * @return  The state of the task
     */
    ghost_task_state_t ghost_task_test(ghost_task_t * t)
    {

        if (t == NULL || t->state == NULL)
            return GHOST_TASK_INVALID;
        return *(t->state);
    }

    /**
     * @brief Wait for a task to finish
     *
     * @param t The task to wait for
     *
     * @return GHOST_SUCCESS on success or GHOST_FAILURE on failure.
     */
    ghost_error_t ghost_task_wait(ghost_task_t * task)
    {
        DEBUG_LOG(1,"Waiting @core %d for task %p whose state is %d",ghost_getCore(),task,*(task->state));


        //    ghost_task_t *parent = (ghost_task_t *)pthread_getspecific(ghost_thread_key);
        //    if (parent != NULL) {
        //    WARNING_LOG("Waiting on a task from within a task ===> free'ing the parent task's resources, idle PUs: %d",NIDLECORES);
        //    ghost_task_unpin(parent);
        //    WARNING_LOG("Now idle PUs: %d",NIDLECORES);
        //    }

        pthread_mutex_lock(task->mutex);
        if (*(task->state) != GHOST_TASK_FINISHED) {
            DEBUG_LOG(1,"Waiting for signal @ cond %p from task %p",task->finishedCond,task);
            pthread_cond_wait(task->finishedCond,task->mutex);
        } else {
            DEBUG_LOG(1,"Task %p has already finished",task);
        }

        // pin again if have been unpinned

        pthread_mutex_unlock(task->mutex);
        pthread_mutex_lock(&anyTaskFinishedMutex);
        pthread_cond_broadcast(&anyTaskFinishedCond);
        pthread_mutex_unlock(&anyTaskFinishedMutex);
        DEBUG_LOG(1,"Finished waitung for task %p!",task);

        return GHOST_SUCCESS;

    }

    /**
     * @brief Return a string representing the task's state
     *
     * @param state The task to test
     *
     * @return The state string
     */
    char *ghost_task_strstate(ghost_task_state_t state)
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

    /**
     * @brief Wait for all tasks in all queues to be finished.
     *
     * @return GHOST_SUCCESS on success or GHOST_FAILURE on failure.
     */
    ghost_error_t ghost_task_waitall()
    {
        ghost_task_t *t;

        pthread_mutex_lock(&globalMutex);
        t = taskq->head;
        pthread_mutex_unlock(&globalMutex);
        while (t != NULL)
        {
            DEBUG_LOG(1,"Waitall: Waiting for task %p",t);
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
    ghost_error_t ghost_task_waitsome(ghost_task_t ** tasks, int nt, int *index)
    {
        int t;
        int ret = 0;
        pthread_t threads[nt];

        for (t=0; t<nt; t++)
        { // look if one of the tasks is already finished
            pthread_mutex_lock(tasks[t]->mutex);
            if (*(tasks[t]->state) == GHOST_TASK_FINISHED) 
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
            if (*(tasks[t]->state) == GHOST_TASK_FINISHED) 
            {
                index[t] = 1;
            } else {
                index[t] = 0;
            }
            pthread_mutex_unlock(tasks[t]->mutex);
        }

        return GHOST_SUCCESS;
    }


    /**
     * @brief Free a task's resources.
     *
     * @param t The task to be destroyed
     *
     * @return GHOST_SUCCESS on success or GHOST_FAILURE on failure.
     */
    ghost_error_t ghost_task_destroy(ghost_task_t *t)
    {
        pthread_mutex_destroy(t->mutex);
        pthread_cond_destroy(t->finishedCond);

        free(t->cores);
        free(t->state);
        hwloc_bitmap_free(t->coremap);
        hwloc_bitmap_free(t->childusedmap);
        //free(t->ret);
        free(t->mutex);
        free(t->finishedCond);
        free(t);

        return GHOST_SUCCESS;
    }

    /**
     * @brief Initliaze a task 
     *
     * @param nThreads The number of threads which are reserved for the task
     * @param LD The index of the task queue this task should be added to
     * @param func The function the task should execute
     * @param arg The arguments to the task's function
     * @param flags The task's flags
     *
     * @return A pointer to an initialized task
     */
    ghost_error_t ghost_task_init(ghost_task_t **t, int nThreads, int LD, void *(*func)(void *), void *arg, int flags)
    {
        *t = (ghost_task_t *)ghost_malloc(sizeof(ghost_task_t));
        if (ghost_thpool == NULL) {
            WARNING_LOG("The thread pool is not initialized. Something went terribly wrong.");
            /*        int nt = ghost_getNumberOfPhysicalCores()/ghost_getNumberOfRanks(ghost_node_comm);
                      int ft = ghost_getRank(ghost_node_comm)*nt;
                      int poolThreads[] = {nt,nt};
                      int firstThread[] = {ft,ft};
                      int levels = ghost_getNumberOfHwThreads()/ghost_getNumberOfPhysicalCores();
            //DEBUG_LOG(1,"Trying to initialize a task but the thread pool has not yet been initialized. Doing the init now with %d threads!",nt*levels);
            //ghost_thpool_init(poolThreads,firstThread,levels);*/
        }
        if (taskq == NULL) {
            DEBUG_LOG(1,"Trying to initialize a task but the task queues have not yet been initialized. Doing the init now...");
            ghost_taskq_init();
        }

        if (nThreads == GHOST_TASK_FILL_LD) {
            if (LD < 0) {
                WARNING_LOG("FILL_LD does only work when the LD is given! Not adding task!");
                return GHOST_ERR_INVALID_ARG;
            }
            (*t)->nThreads = nThreadsPerLD(LD);
        } 
        else if (nThreads == GHOST_TASK_FILL_ALL) {
#ifdef GHOST_HAVE_OPENMP
            (*t)->nThreads = ghost_thpool->nThreads;
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
        (*t)->state = (int *)ghost_malloc(sizeof(int));
        *((*t)->state) = GHOST_TASK_INVALID;
        (*t)->cores = (int *)ghost_malloc(sizeof(int)*(*t)->nThreads);
        (*t)->coremap = hwloc_bitmap_alloc();
        (*t)->childusedmap = hwloc_bitmap_alloc();
        (*t)->next = NULL;
        (*t)->prev = NULL;
        (*t)->parent = NULL;
        (*t)->finishedCond = (pthread_cond_t *)ghost_malloc(sizeof(pthread_cond_t));
        (*t)->mutex = (pthread_mutex_t *)ghost_malloc(sizeof(pthread_mutex_t));

        return GHOST_SUCCESS;
    }

    /**
     * @brief Free all resources of the thread pool
     *
     * @return GHOST_SUCCESS on success or GHOST_FAILURE on failure.
     */
    ghost_error_t ghost_thpool_finish()
    {
        if (ghost_thpool == NULL)
            return GHOST_SUCCESS;

        free(ghost_thpool->threads);
        free(ghost_thpool->sem);
        free(ghost_thpool->PUs);
        free(ghost_thpool);
        hwloc_bitmap_free(ghost_thpool->cpuset);
        hwloc_bitmap_free(ghost_thpool->busy);

        return GHOST_SUCCESS;
    }
