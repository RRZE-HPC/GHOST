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

static pthread_mutex_t pinningMutex = PTHREAD_MUTEX_INITIALIZER;

/**
 * @brief The task queue created by ghost_taskq_init().
 */
ghost_taskq_t *taskq = NULL;


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

static int nrunning = 0;

static void * thread_main(void *arg);

#define PTHREAD_NULL (pthread_t)0
static pthread_cond_t * newTaskCond_by_threadcount;
static pthread_mutex_t * newTaskMutex_by_threadcount;
static int * num_shep_by_threadcount;
static int * waiting_shep_by_threadcount;
static int * num_tasks_by_threadcount;

static pthread_key_t threadcount_key;
static pthread_key_t mutex_key;


ghost_error_t ghost_taskq_create()
{
    int t;
    int npu;

    pthread_mutex_init(&newTaskMutex,NULL);
    pthread_mutex_init(&globalMutex,NULL);
    pthread_mutex_lock(&newTaskMutex);
    pthread_mutex_lock(&globalMutex);

    ghost_machine_npu(&npu,GHOST_NUMANODE_ANY);

    GHOST_CALL_RETURN(ghost_malloc((void **)&taskq,sizeof(ghost_taskq_t)));
    pthread_mutex_init(&(taskq->mutex),NULL);

    pthread_mutex_lock(&(taskq->mutex));

    GHOST_CALL_RETURN(ghost_malloc((void **)&newTaskMutex_by_threadcount,(npu+1)*sizeof(pthread_mutex_t)));
    GHOST_CALL_RETURN(ghost_malloc((void **)&newTaskCond_by_threadcount,(npu+1)*sizeof(pthread_cond_t)));
    GHOST_CALL_RETURN(ghost_malloc((void **)&num_shep_by_threadcount,(npu+1)*sizeof(int)));
    GHOST_CALL_RETURN(ghost_malloc((void **)&waiting_shep_by_threadcount,(npu+1)*sizeof(int)));
    GHOST_CALL_RETURN(ghost_malloc((void **)&num_tasks_by_threadcount,(npu+1)*sizeof(int)));
    taskq->tail = NULL;
    taskq->head = NULL;

    pthread_cond_init(&newTaskCond,NULL);
    sem_init(&taskSem, 0, 0);
    pthread_cond_init(&anyTaskFinishedCond,NULL);
    pthread_mutex_init(&anyTaskFinishedMutex,NULL);
    pthread_key_create(&threadcount_key,NULL);
    pthread_key_create(&mutex_key,NULL);


    for (t=0; t<npu+1; t++) {
        pthread_cond_init(&newTaskCond_by_threadcount[t],NULL);
        pthread_mutex_init(&newTaskMutex_by_threadcount[t],NULL);
        num_shep_by_threadcount[t] = 1;
        waiting_shep_by_threadcount[t] = 0; 
        num_tasks_by_threadcount[t] = 0; 
    }

    void *(*threadFunc)(void *);
    ghost_taskq_startroutine(&threadFunc);
    ghost_thpool_create(npu+1,threadFunc);

    pthread_mutex_unlock(&(taskq->mutex));
    pthread_mutex_unlock(&globalMutex);
    pthread_mutex_unlock(&newTaskMutex);
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
    if (q->head == NULL) {
        DEBUG_LOG(1,"Empty queue, returning NULL!");
        return NULL;
    }
    ghost_task_t *curTask = q->head;

    ghost_thpool_t *ghost_thpool = NULL;
    ghost_thpool_get(&ghost_thpool);


    while(curTask != NULL)
    {
        //if (curTask->nThreads != nthreads) {
        //    curTask = curTask->next;
        //    continue;
        // }
        int d;
        for (d=0; d<curTask->ndepends; d++) {
            pthread_mutex_lock(curTask->depends[d]->mutex);
            if (curTask->depends[d]->state != GHOST_TASK_FINISHED) {
                pthread_mutex_unlock(curTask->depends[d]->mutex);
                break;
            }
            pthread_mutex_unlock(curTask->depends[d]->mutex);
        }
        if (d<curTask->ndepends) {
            curTask = curTask->next;
            continue;
        }

        if (curTask->flags & GHOST_TASK_NOT_PIN) {
            taskq_deleteTask(q,curTask);    
            ghost_omp_nthread_set(curTask->nThreads);
#ifdef GHOST_HAVE_MKL
            mkl_set_num_threads(curTask->nThreads); 
#endif
#pragma omp parallel
            ghost_thread_unpin();
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
        hwloc_bitmap_t parentscores = hwloc_bitmap_alloc(); // TODO free
        if (curTask->parent && !(curTask->parent->flags & GHOST_TASK_NOT_ALLOW_CHILD)) {
            //char *a, *b, *c;
            //hwloc_bitmap_list_asprintf(&a,curTask->parent->coremap);
            //hwloc_bitmap_list_asprintf(&b,curTask->parent->childusedmap);

            hwloc_bitmap_andnot(parentscores,curTask->parent->coremap,curTask->parent->childusedmap);
            //hwloc_bitmap_list_asprintf(&c,parentscores);
            //WARNING_LOG("(%lu) %s = %s andnot %s (%p)",(unsigned long)pthread_self(),c,a,b,curTask->parent->childusedmap);
            if (curTask->flags & GHOST_TASK_LD_STRICT) {
                hwloc_bitmap_and(parentscores,parentscores,numanode->cpuset);
            }
            availcores += hwloc_bitmap_weight(parentscores);
        }
        if (availcores < curTask->nThreads) {
            DEBUG_LOG(1,"Skipping task %p because it needs %d threads and only %d threads are available",(void *)curTask,curTask->nThreads,availcores);
            hwloc_bitmap_free(parentscores);
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
        ghost_omp_nthread_set(curTask->nThreads);
#ifdef GHOST_HAVE_MKL
        mkl_set_num_threads(curTask->nThreads); 
#endif

        //        if (curTask->flags & GHOST_TASK_NO_PIN) {
        //            return curTask;
        //        }


        int curThread;
        ghost_pumap_t *pumap;
        ghost_pumap_get(&pumap);


        hwloc_bitmap_t mybusy = hwloc_bitmap_alloc();
        if (curTask->parent && !(curTask->parent->flags & GHOST_TASK_NOT_ALLOW_CHILD)) {
            /* char *a, *b;
               hwloc_bitmap_list_asprintf(&a,ghost_thpool->busy);
               hwloc_bitmap_list_asprintf(&b,parentscores);
               INFO_LOG("Need %d cores, available cores: %d (busy %s) + %d from parent (free in parent %s)",curTask->nThreads,NIDLECORES,a,hwloc_bitmap_weight(parentscores),b);*/
            hwloc_bitmap_andnot(mybusy,pumap->busy,parentscores);
        } else {
            hwloc_bitmap_copy(mybusy,pumap->busy);
        }

        hwloc_bitmap_t myfree = hwloc_bitmap_alloc();
        hwloc_bitmap_andnot(myfree,pumap->cpuset,mybusy);

        int idx = -1;
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


        ghost_task_flags_t flags = curTask->flags;
        //hwloc_bitmap_t coremap = curTask->coremap; 
        int nthreads = curTask->nThreads;
        int LD = curTask->LD;

        //        hwloc_bitmap_t coremap = hwloc_bitmap_alloc();

#pragma omp parallel 
        {
            pthread_mutex_lock(&pinningMutex);
            int t;
            int privreserved;
            int core;
            pthread_mutex_unlock(&pinningMutex);

#pragma omp for ordered schedule(static,1) 
            for (curThread=0; curThread<nthreads; curThread++) {
#pragma omp ordered
                {
                    pthread_mutex_lock(&pinningMutex);
                    t = hwloc_bitmap_first(myfree);

                    hwloc_obj_t pu = hwloc_get_pu_obj_by_os_index(topology,t);
                    //                for (; t<ghost_thpool->nThreads; t++) {

                    //                    hwloc_obj_t pu = pumap->PUs[LD][t];
                    core = pu->os_index;

                    //if (!hwloc_bitmap_isset(ghost_thpool->busy,core)) {
                    DEBUG_LOG(1,"Thread %d (%d): Core # %d is idle, using it",ghost_omp_threadnum(),
                            (int)pthread_self(),core);

                    //                            hwloc_bitmap_set(ghost_thpool->busy,core);
                    hwloc_bitmap_set(mybusy,core);
                    DEBUG_LOG(2,"Pinning thread %lu to core %u",(unsigned long)pthread_self(),pu->os_index);
                    ghost_thread_pin(core);
                    //curTask->cores[reservedCores] = core;
#pragma omp atomic update
                    privreserved++;
                    hwloc_bitmap_clr(myfree,t);

                    //                    if (t >= ghost_thpool->nThreads-1) {
                    //                        if (privreserved < curTask->nThreads) {
                    //                            WARNING_LOG("Too few cores reserved! %d < %d This should not have happened...",privreserved,curTask->nThreads);
                    //                        }
                    //                        break;
                    //                    }
                    pthread_mutex_unlock(&pinningMutex);
                }
                }
#ifdef GHOST_HAVE_INSTR_LIKWID
                LIKWID_MARKER_THREADINIT;
#endif
                }
                hwloc_bitmap_or(curTask->coremap,curTask->coremap,mybusy);
                //                hwloc_bitmap_or(curTask->coremap,curTask->coremap,coremap);
                //                hwloc_bitmap_free(coremap);
                if (curTask->parent && !(curTask->parent->flags & GHOST_TASK_NOT_ALLOW_CHILD)) {
                    //char *a;
                    //hwloc_bitmap_list_asprintf(&a,curTask->parent->childusedmap);
                    //WARNING_LOG("### %p %s",curTask->parent->childusedmap,a);
                    hwloc_bitmap_or(curTask->parent->childusedmap,curTask->parent->childusedmap,mybusy);
                    //hwloc_bitmap_list_asprintf(&a,curTask->parent->childusedmap);
                    //WARNING_LOG("### %p %s",curTask->parent->childusedmap,a);
                }
                ghost_pumap_setbusy(mybusy);
                //            hwloc_bitmap_or(pumap->busy,pumap->busy,mybusy);



                hwloc_bitmap_free(mybusy);
                hwloc_bitmap_free(parentscores);
                DEBUG_LOG(1,"Pinning successful, returning");

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
            //    kmp_set_blocktime(200);
            //    kmp_set_library_throughput();
            //    UNUSED(arg);
            ghost_task_t *myTask = NULL;

            pthread_key_t key;
            ghost_thpool_key(&key);
            int nthreads = (int)(intptr_t)arg;

            pthread_setspecific(key,NULL);
            pthread_setcancelstate(PTHREAD_CANCEL_ENABLE,NULL);
            pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS,NULL);

            ghost_thpool_t *ghost_thpool = NULL;
            ghost_thpool_get(&ghost_thpool);
            sem_post(ghost_thpool->sem);


            DEBUG_LOG(1,"Shepherd thread %lu in thread_main() called with %"PRIdPTR,(unsigned long)pthread_self(), (intptr_t)arg);
            while (1) // as long as there are jobs stay alive
            {
                pthread_mutex_lock(&newTaskMutex_by_threadcount[nthreads]);
                waiting_shep_by_threadcount[nthreads]++; 
                while(num_tasks_by_threadcount[nthreads] == 0) {
                    INFO_LOG("No tasks with %d threads --> waiting for them",nthreads);
                    pthread_cond_wait(&newTaskCond_by_threadcount[nthreads],&newTaskMutex_by_threadcount[nthreads]);
                    INFO_LOG("Woken up by new task with %d threads, actual number: %d",nthreads,num_tasks_by_threadcount[nthreads]);
                }
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
                    continue;
                } 
                pthread_mutex_lock(&newTaskMutex_by_threadcount[nthreads]);
                INFO_LOG("Found task with %d threads. Similar shephs waiting: %d",nthreads,waiting_shep_by_threadcount[nthreads]);
                if (waiting_shep_by_threadcount[nthreads] == 0) {
                    INFO_LOG("Adding another shepherd thread for %d-thread tasks",nthreads);
                    void *(*threadFunc)(void *);
                    ghost_taskq_startroutine(&threadFunc);
                    ghost_thpool_thread_add(threadFunc,nthreads);
                    num_shep_by_threadcount[nthreads]++;
                }
                pthread_mutex_unlock(&newTaskMutex_by_threadcount[nthreads]);


                pthread_mutex_lock(myTask->mutex);
                myTask->state = GHOST_TASK_RUNNING;    
                pthread_mutex_unlock(myTask->mutex);

                myTask->ret = myTask->func(myTask->arg);

                pthread_mutex_lock(myTask->mutex);
                myTask->state = GHOST_TASK_FINISHED;    
                ghost_task_unpin(myTask);
                pthread_cond_broadcast(myTask->finishedCond);
                pthread_mutex_unlock(myTask->mutex);

                pthread_mutex_lock(&anyTaskFinishedMutex);
                pthread_mutex_unlock(&anyTaskFinishedMutex);
                pthread_cond_broadcast(&anyTaskFinishedCond);

                pthread_mutex_lock(&newTaskMutex_by_threadcount[nthreads]);
                num_tasks_by_threadcount[nthreads]--;
                pthread_mutex_unlock(&newTaskMutex_by_threadcount[nthreads]);

#if 0
                // TODO wait for condition when unpinned or new task

                //if (sem_wait(&taskSem)) // TODO wait for a signal in order to avoid entering the loop when nothing has changed
                // {
                //     if (errno == EINTR) {
                //        continue;
                //    }
                //    ERROR_LOG("Waiting for tasks failed: %s. Will try again.",strerror(errno));
                //    continue;
                //}
                //            INFO_LOG("%d CONDWAIT killed==%d",(int)pthread_self(),killed);

                pthread_mutex_lock(&globalMutex);
                if (killed) { // check if killed while waiting for newTaskMutex
                    pthread_mutex_unlock(&globalMutex);
                    break;
                }


                //                uintptr_t key_nthr = (uintptr_t)pthread_getspecific(threadcount_key);
                //                INFO_LOG("Shepherd thread %d [%d] handled a task with %"PRIuPTR" threads once.",(int)pthread_self(),(intptr_t)arg,key_nthr);
                //if (key_nthr != 0) {
                //    INFO_LOG("Shepherd thread %d waiting for pinned tasks with %"PRIuPTR" threads on condition %p",(int)pthread_self(),key_nthr,&newTaskCond_by_threadcount[key_nthr]);
                //    INFO_LOG("Shepherd thread %d finished waiting",(int)pthread_self());
                //}


                pthread_mutex_unlock(&globalMutex);

                if (pthread_getspecific(key)) { // if this shepherd thread is a virgin, look for new tasks immediately!
                    pthread_mutex_t *mutex = &newTaskMutex_by_threadcount[(intptr_t)arg];
                    // (pthread_mutex_t *)pthread_getspecific(mutex_key);
                    //                    WARNING_LOG("%d %p",(intptr_t)arg,mutex);
                    pthread_mutex_lock(&globalMutex);
                    waiting_shep_by_threadcount[(intptr_t)arg]++;
                    INFO_LOG("Shepherd thread %d [%"PRIiPTR"] waiting for tasks. Total waiting: %d",(int)pthread_self(),(intptr_t)arg,waiting_shep_by_threadcount[(intptr_t)arg]);
                    pthread_mutex_unlock(&globalMutex);
                    pthread_mutex_lock(mutex);
                    //while (!&newTaskCond_by_threadcount[(intptr_t)arg])
                    pthread_cond_wait(&newTaskCond_by_threadcount[(intptr_t)arg],mutex);
                    pthread_mutex_unlock(mutex);
                    pthread_mutex_lock(&globalMutex);
                    waiting_shep_by_threadcount[(intptr_t)arg]--;
                    INFO_LOG("The number of waiting shepherd threads for %d-thread tasks gets diminished to %d",(intptr_t)arg,waiting_shep_by_threadcount[(intptr_t)arg]);
                    pthread_mutex_unlock(&globalMutex);
                } else {
                    INFO_LOG("Shepherd thread %d [%"PRIiPTR"] will skip the wait. Total waiting: %d",(int)pthread_self(),(intptr_t)arg,waiting_shep_by_threadcount[(intptr_t)arg]);
                }


                pthread_mutex_lock(&globalMutex);

                if (pthread_getspecific(key) && waiting_shep_by_threadcount[(intptr_t)arg] <= 1) {
                    INFO_LOG("Adding another shepherd thread for %d-thread tasks",(intptr_t)arg);
                    void *(*threadFunc)(void *);
                    ghost_taskq_startroutine(&threadFunc);
                    ghost_thpool_thread_add(threadFunc,(intptr_t)arg);
                    num_shep_by_threadcount[(intptr_t)arg]++;
                }

                //                if (key_nthr == 0) {
                //                    INFO_LOG("Shepherd thread %d will continue",(int)pthread_self());
                //                    pthread_mutex_unlock(&newTaskMutex);
                //                    continue;
                //                }

                //pthread_cond_wait(&newTaskCond,&newTaskMutex);

                if (killed) { // check if killed while waiting for newTaskCond
                    pthread_mutex_unlock(&globalMutex);
                    //                    pthread_mutex_unlock(&newTaskMutex);
                    break;
                }
                //            INFO_LOG("%d AFTER CONDWAIT killed==%d",(int)pthread_self(),killed);

                //pthread_mutex_lock(&globalMutex);
                //if (killed) // thread has been woken by the finish() function
                //{
                //    pthread_mutex_unlock(&globalMutex);
                //    DEBUG_LOG(2,"Thread %d: Not executing any further tasks",(int)pthread_self());
                //pthread_cond_signal(&newTaskCond);
                //sem_post(&taskSem); // wake up another thread
                //    break;
                //}
                //pthread_mutex_unlock(&globalMutex);

                //    WARNING_LOG("1 %d : %d",(intptr_t)arg,kmp_get_blocktime());
                //    kmp_set_blocktime((intptr_t)arg);
                //    WARNING_LOG("2 %d : %d",(intptr_t)arg,kmp_get_blocktime());

                pthread_mutex_unlock(&globalMutex);
                INFO_LOG("HERE1");
                pthread_mutex_lock(&taskq->mutex);
                INFO_LOG("HERE2");
                myTask = taskq_findDeleteAndPinTask(taskq);
                pthread_mutex_unlock(&taskq->mutex);
                if (!myTask) {
                    //                    nrunning++;

                    // resize thread pool
                    //if (nrunning == ghost_thpool->nThreads) {
                    //    void *(*threadFunc)(void *);
                    //    ghost_taskq_startroutine(&threadFunc);
                    //    ghost_thpool_create(ghost_thpool->nThreads*2,threadFunc);
                    //}

                    DEBUG_LOG(1,"Thread %d: Could not find a suited task in any queue",(int)pthread_self());
                    INFO_LOG("Coud not find %d-thread task",(intptr_t)arg);
                    //pthread_cond_wait(&newTaskCond,&newTaskMutex);
                    //                    pthread_mutex_unlock(&newTaskMutex);
                    pthread_setspecific(key,0xbeef);
                    //pthread_cond_signal(&newTaskCond_by_threadcount[(intptr_t)arg]);
                    //sem_post(&taskSem);
                    continue;
                }
                arg = myTask->nThreads;

                //pthread_mutex_unlock(&newTaskMutex);

                pthread_mutex_lock(myTask->mutex);
                myTask->state = GHOST_TASK_RUNNING;    
                pthread_mutex_unlock(myTask->mutex);

                DEBUG_LOG(1,"Thread %d: Finally executing task %p",(int)pthread_self(),(void *)myTask);

                pthread_setspecific(key,myTask);

#ifdef __INTEL_COMPILER
                //kmp_set_blocktime(0);
#endif
                INFO_LOG("Doing %d-thread task",(intptr_t)arg);
                myTask->ret = myTask->func(myTask->arg);
                INFO_LOG("Finished %d-thread task",(intptr_t)arg);
                //    WARNING_LOG("2 %d : %d",(intptr_t)arg,kmp_get_blocktime());
#ifdef __INTEL_COMPILER
                //kmp_set_blocktime(200);
#endif

                //pthread_setspecific(key,NULL);
                //if (!(myTask->flags & GHOST_TASK_NOT_PIN)) {
                //    INFO_LOG("Reserving shepherd thread %d for later pinned tasks with %d threads",(int)pthread_self(),myTask->nThreads);
                //    pthread_setspecific(threadcount_key,(const void *)(uintptr_t)myTask->nThreads);
                //                } else {
                //                    if (shep_threads_by_threadcount[myTask->nThreads] == PTHREAD_NULL) {
                //                        shep_threads_by_threadcount[myTask->nThreads] = pthread_self();
                //                    }
                // }

                DEBUG_LOG(1,"Thread %lu: Finished executing task: %p. Free'ing resources and waking up another thread"
                        ,(unsigned long)pthread_self(),(void *)myTask);

                pthread_mutex_lock(&globalMutex);
                ghost_task_unpin(myTask);
                nrunning--;
                pthread_mutex_unlock(&globalMutex);

                // try to run any task that has been blocked because of this one
                //pthread_mutex_lock(&newTaskMutex);
                //pthread_cond_signal(&newTaskCond_by_threadcount[(intptr_t)arg]);
                //pthread_cond_signal(&newTaskCond_by_threadcount[]);
                if (taskq->head) {
                    INFO_LOG("Try to kick off the head of the queue which has %d threads",taskq->head->nThreads);
                    pthread_cond_signal(&newTaskCond_by_threadcount[taskq->head->nThreads]);
                }
                //                pthread_cond_signal(&newTaskCond);
                //pthread_mutex_unlock(&newTaskMutex);

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
#endif
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

            //            pthread_mutex_lock(&newTaskMutex_by_threadcount[t->nThreads]);
            if (taskq==NULL) {
                WARNING_LOG("Tried to add a task to a queue which is NULL");
                return GHOST_ERR_INVALID_ARG;
            }

            //pthread_mutex_lock(&taskq->mutex);
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

            //sem_post(&taskSem);
            //pthread_mutex_lock(&newTaskMutex);
            //if (!(t->flags & GHOST_TASK_NOT_PIN)) {
            //
            pthread_mutex_lock(&newTaskMutex_by_threadcount[t->nThreads]);
            INFO_LOG("CONDSIGNAL %d",t->nThreads);
            num_tasks_by_threadcount[t->nThreads]++;
            pthread_cond_signal(&newTaskCond_by_threadcount[t->nThreads]);
            //}
            //usleep(1e6); // sleep 1 sec
            //pthread_cond_signal(&newTaskCond);

            pthread_mutex_unlock(&newTaskMutex_by_threadcount[t->nThreads]);
            //pthread_mutex_unlock(&newTaskMutex);

            //pthread_mutex_unlock(&taskq->mutex);
            return GHOST_SUCCESS;
        }

        /**
         * @brief Execute all outstanding threads and free the task queues' resources
         *
         * @return GHOST_SUCCESS on success or GHOST_FAILURE on failure.
         */
        ghost_error_t ghost_taskq_destroy()
        {
            int npu;
            ghost_machine_npu(&npu,GHOST_NUMANODE_ANY);
            if (taskq == NULL) {
                return GHOST_SUCCESS;
            }


            pthread_mutex_lock(&globalMutex);
            killed = 1;
            pthread_mutex_unlock(&globalMutex);

            DEBUG_LOG(1,"Wake up all threads");    
            //pthread_cond_broadcast(&newTaskCond);


            int t,n;
            for (t=0; t<(npu+1); t++) {
                for (n=0; n<num_shep_by_threadcount[t]; n++) {
                    pthread_mutex_lock(&newTaskMutex_by_threadcount[t]);
                    num_tasks_by_threadcount[t]=1;                    
                    pthread_mutex_unlock(&newTaskMutex_by_threadcount[t]);
                    pthread_cond_signal(&newTaskCond_by_threadcount[t]);
                }
            }
            //        if (sem_post(&taskSem)){
            //            WARNING_LOG("Error in sem_post: %s",strerror(errno));
            //            return GHOST_ERR_UNKNOWN;
            //        }

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

            ghost_task_t *t;

            pthread_mutex_lock(&taskq->mutex);
            t = taskq->head;
            pthread_mutex_unlock(&taskq->mutex);
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
            // pthread_t threads[nt];

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

            INFO_LOG("None of the tasks has already finished. Waiting for (at least) one of them...");


            //for (t=0; t<nt; t++)
            //{
            //    pthread_create(&threads[t],NULL,(void *(*)(void *))&ghost_task_wait,tasks[t]);
            //}

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
