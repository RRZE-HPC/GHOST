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
#include <omp.h>


#include "ghost_taskq.h"
#include "ghost_util.h"

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
//int corestate[8]; //TODO length constant

static void * thread_main(void *arg);
static int ghost_task_unpin(ghost_task_t *task);

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

static int nIdleCoresAtLD(int ld);
static int nThreadsPerLD(int ld);
static int firstThreadOfLD(int ld);

/**
 * @brief Initializes a thread pool with a given number of threads.
 * @param nThreads
 * @return GHOST_SUCCESS on success or GHOST_FAILURE on failure.
 * 
 * A number of pthreads will be created and each one will have thread_main() as start routine.
 * In order to make sure that each thread has entered the infinite loop, a wait on a semaphore is
 * performed before this function returns.
 */
int ghost_thpool_init(int *nThreads, int *firstThread, int levels)
{
	int t,q,i,l,p;
	int totalThreads = 0;
	hwloc_obj_t obj;

	int ncores = hwloc_get_nbobjs_by_type(topology,HWLOC_OBJ_CORE);
	int nthreads = hwloc_get_nbobjs_by_type(topology,HWLOC_OBJ_PU);
	int smt = nthreads/ncores;

	for (l=0; l<levels; l++) {
		DEBUG_LOG(1,"Required %d threads @ SMT level %d, starting from thread %d",nThreads[l],l,firstThread[l]);
		totalThreads += nThreads[l];
	}

	ghost_thpool = (ghost_thpool_t*)ghost_malloc(sizeof(ghost_thpool_t));
	ghost_thpool->PUs = (hwloc_obj_t *)ghost_malloc(totalThreads*sizeof(hwloc_obj_t));
	ghost_thpool->nThreads = totalThreads;
	ghost_thpool->threads = (pthread_t *)ghost_malloc(ghost_thpool->nThreads*sizeof(pthread_t));
	ghost_thpool->sem = (sem_t*)ghost_malloc(sizeof(sem_t));
	sem_init(ghost_thpool->sem, 0, 0);
	sem_init(&taskSem, 0, 0);
	pthread_mutex_init(&globalMutex,NULL);

	pthread_key_create(&ghost_thread_key,NULL);

	ghost_thpool->cpuset = hwloc_bitmap_alloc();
	ghost_thpool->busy = hwloc_bitmap_alloc();

	int puidx;
	hwloc_obj_t runner;
	for (l=0; l<levels; l++) {
		for (p=firstThread[l]; p<firstThread[l]+nThreads[l]; p++) {
			i = p*smt+l;
			puidx = (i-smt*firstThread[l])/(smt-levels+1);
			obj = hwloc_get_obj_by_type(topology,HWLOC_OBJ_PU,i);
			hwloc_bitmap_or(ghost_thpool->cpuset,ghost_thpool->cpuset,obj->cpuset);
			ghost_thpool->PUs[puidx] = obj;
		}
	}

	int nodes[totalThreads];
	for (i=0; i<totalThreads; i++) {
		obj = ghost_thpool->PUs[i];
		for (runner=obj; runner; runner=runner->parent) {
			if (runner->type == HWLOC_OBJ_NODE) {
				nodes[i] = runner->logical_index;
				break;
			}
		}
		DEBUG_LOG(1,"Thread # %3d running @ PU %3u (OS: %3u), SMT level %2d, NUMA node %u",i,obj->logical_index,obj->os_index,i%smt,runner->logical_index);
	}
	
	qsort(nodes,totalThreads,sizeof(int),intcomp);
	ghost_thpool->nLDs = 1;
	
	for (i=1; i<totalThreads; i++) {
		if (nodes[i] != nodes[i-1]) {
			ghost_thpool->nLDs++;
		}
	}
	coreidx = (int **)ghost_malloc(sizeof(int *)*ghost_thpool->nLDs);

	for (q=0; q<ghost_thpool->nLDs; q++) {
		int localthreads = nThreadsPerLD(q);
		coreidx[q] = (int *)ghost_malloc(sizeof(int)*ghost_thpool->nThreads);

		for (t=0; t<localthreads; t++) { // my own threads
			coreidx[q][t] = firstThreadOfLD(q)+t;
		}
		for (; t-localthreads<firstThreadOfLD(q); t++) {
			coreidx[q][t] = t-firstThreadOfLD(q);
		}	
		for (; t<ghost_thpool->nThreads; t++) {
			coreidx[q][t] = t;
		}	
	}

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
			if (runner->type == HWLOC_OBJ_NODE) {
				if (runner->logical_index == ld) {
					n++;
				}
			}
		}
	}

	return n;
}

static int nIdleCoresAtLD(int ld)
{
	int begin = firstThreadOfLD(ld), end = begin+nThreadsPerLD(ld);
	hwloc_bitmap_t LDBM = hwloc_bitmap_alloc();

	hwloc_bitmap_set_range(LDBM,begin,end);
	hwloc_bitmap_t busy = hwloc_bitmap_alloc();
   	hwloc_bitmap_andnot(busy,LDBM,ghost_thpool->busy);

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
			if (runner->type == HWLOC_OBJ_NODE) {
				if (runner->logical_index == ld) {
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
int ghost_taskq_init()
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
		if ((curTask->flags & GHOST_TASK_LD_STRICT) && (nIdleCoresAtLD(curTask->LD) < curTask->nThreads)) {
			DEBUG_LOG(1,"Skipping task %p because there are not enough idle cores at its strict LD %d: %d < %d",curTask,curTask->LD,nIdleCoresAtLD(curTask->LD),curTask->nThreads);
			curTask = curTask->next;
			continue;
		}
		if ((ghost_thpool->nThreads-hwloc_bitmap_weight(ghost_thpool->busy)) < curTask->nThreads) {
			DEBUG_LOG(1,"Skipping task %p because it needs %d threads and only %d threads are idle",curTask,curTask->nThreads,(ghost_thpool->nThreads-hwloc_bitmap_weight(ghost_thpool->busy)));
			curTask = curTask->next;
			continue;
		}

		DEBUG_LOG(1,"Thread %d: Found a suiting task: %p! task->nThreads=%d, nIdleCores[LD%d]=%d, nIdleCores=%d",(int)pthread_self(),curTask,curTask->nThreads,curTask->LD,nIdleCoresAtLD(curTask->LD),(ghost_thpool->nThreads-hwloc_bitmap_weight(ghost_thpool->busy)));

		DEBUG_LOG(1,"Deleting task itself");
		taskq_deleteTask(q,curTask);	
		DEBUG_LOG(1,"Pinning the task's threads");
		ghost_ompSetNumThreads(curTask->nThreads);

		int reservedCores = 0;

		int t = 0;
		int curThread;

		for (curThread=0; curThread<curTask->nThreads; curThread++)
		{
#pragma omp parallel
			{
				if (ghost_ompGetThreadNum() == curThread)
				{ // this block will be entered by one thread at a time in ascending order

					for (; t<ghost_thpool->nThreads; t++) {
						int core = coreidx[curTask->LD][t];
						if (!hwloc_bitmap_isset(ghost_thpool->busy,core)) {
							DEBUG_LOG(1,"Thread %d: Core # %d is idle, using it",
									(int)pthread_self(),core);

							hwloc_bitmap_set(ghost_thpool->busy,core);
							ghost_setCore(ghost_thpool->PUs[core]->os_index);
							curTask->cores[reservedCores] = core;
							reservedCores++;
							break;
						}
					}
				}
			}
		}

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
static void * thread_main(void *arg)
{
	UNUSED(arg);
	ghost_task_t *myTask;

	int sval = 1;
	sem_post(ghost_thpool->sem);

	while (1) // as long as there are jobs stay alive
	{
		if (sem_wait(&taskSem)) 
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


		pthread_mutex_lock(&globalMutex);
		myTask = taskq_findDeleteAndPinTask(taskq);
		pthread_mutex_unlock(&globalMutex);

		if (myTask  == NULL) // no suiting task found
		{
			DEBUG_LOG(1,"Thread %d: Could not find a suited task in any queue",(int)pthread_self());
			sem_post(&taskSem);
			continue;
		}

		pthread_mutex_lock(myTask->mutex);
		*(myTask->state) = GHOST_TASK_RUNNING;	
		pthread_mutex_unlock(myTask->mutex);


		DEBUG_LOG(1,"Thread %d: Finally executing task at core %d: %p",(int)pthread_self(),ghost_getCore(),myTask);

		pthread_setspecific(ghost_thread_key,myTask);
		myTask->ret = myTask->func(myTask->arg);
		pthread_setspecific(ghost_thread_key,NULL);

		//	ghost_unsetCore();

		DEBUG_LOG(1,"Thread %d: Finished executing task: %p",(int)pthread_self(),myTask);

		if (!(myTask->freed))
			ghost_task_unpin(myTask);

		pthread_mutex_lock(myTask->mutex); 
		DEBUG_LOG(1,"Thread %d: Finished with task %p. Setting state to finished...",(int)pthread_self(),myTask);
		*(myTask->state) = GHOST_TASK_FINISHED;
		pthread_cond_broadcast(myTask->finishedCond);
		pthread_mutex_unlock(myTask->mutex);
		DEBUG_LOG(1,"Thread %d: Finished with task %p. Sending signal to all waiters (cond: %p).",(int)pthread_self(),myTask,myTask->finishedCond);
		if (killed) {
			break;
		}
	}
	sem_getvalue(&taskSem,&sval);
	DEBUG_LOG(1,"Thread %d: Exited infinite loop (%d)",(int)pthread_self(),sval);
	return NULL;
}


static int ghost_task_unpin(ghost_task_t *task)
{
	for (int t=0; t<task->nThreads; t++) {
		hwloc_bitmap_clr(ghost_thpool->busy,task->cores[t]);
	}
	task->freed = 1;

	return GHOST_SUCCESS;


}

/**
 * @brief Print a task and all relevant informatio to stdout.
 *
 * @param t The task
 *
 * @return GHOST_SUCCESS on success or GHOST_FAILURE on failure.
 */
int ghost_task_print(ghost_task_t *t) 
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
int ghost_taskq_print_all() 
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
		return GHOST_FAILURE;
	}

	pthread_mutex_lock(&q->mutex);
	if ((q->head == q->tail) && (q->head == NULL)) {
		DEBUG_LOG(1,"Adding task to empty queue");
		q->head = t;
		q->tail = t;
		t->next = NULL;
		t->prev = NULL;
	} else {
		if (t->flags & GHOST_TASK_PRIO_HIGH) 
		{
			DEBUG_LOG(1,"Adding high-priority task to non-empty queue");
			q->head->prev = t;
			t->next = q->head;
			t->prev = NULL;
			q->head = t;

		} else
		{
			DEBUG_LOG(1,"Adding normal-priority task to non-empty queue");
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
int ghost_task_add(ghost_task_t *t)
{
	// if a task is initialized _once_ but added several times, this has to be done each time it is added
	pthread_cond_init(t->finishedCond,NULL);
	pthread_mutex_init(t->mutex,NULL);
	*(t->state) = GHOST_TASK_INVALID;
	memset(t->cores,0,sizeof(int)*t->nThreads);

	t->parent = (ghost_task_t *)pthread_getspecific(ghost_thread_key);
	t->freed = 0;

	DEBUG_LOG(1,"Task %p w/ %d threads goes to queue %p (LD %d)",t,t->nThreads,taskq,t->LD);
	taskq_additem(taskq,t);
	*(t->state) = GHOST_TASK_ENQUEUED;

	sem_post(&taskSem);

	DEBUG_LOG(1,"Task added successfully");

	return GHOST_SUCCESS;
}

/**
 * @brief Execute all outstanding threads and free the task queues' resources
 *
 * @return GHOST_SUCCESS on success or GHOST_FAILURE on failure.
 */
int ghost_taskq_finish()
{
	DEBUG_LOG(1,"Finishing task queue");
	int t;
	if (taskq == NULL)
		return GHOST_SUCCESS;

	ghost_task_waitall(); // finish all outstanding tasks
	pthread_mutex_lock(&globalMutex);
	killed = 1;
	pthread_mutex_unlock(&globalMutex);

	DEBUG_LOG(1,"Wake up all threads");	
	if (sem_post(&taskSem)){
		WARNING_LOG("Error in sem_post: %s",strerror(errno));
		return GHOST_FAILURE;
	}
	DEBUG_LOG(1,"Join all threads");	
	for (t=0; t<ghost_thpool->nThreads; t++)
	{ 		
		if (pthread_join(ghost_thpool->threads[t],NULL)){
			return GHOST_FAILURE;
		}
	}

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
int ghost_task_test(ghost_task_t * t)
{
	if (t->state == NULL)
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
int ghost_task_wait(ghost_task_t * task)
{
	DEBUG_LOG(1,"Waiting @core %d for task %p whose state is %d",ghost_getCore(),task,*(task->state));


	ghost_task_t *parent = (ghost_task_t *)pthread_getspecific(ghost_thread_key);
	if (parent != NULL) {
		ghost_task_unpin(parent);
	}

	pthread_mutex_lock(task->mutex);
	if (*(task->state) != GHOST_TASK_FINISHED) {
		DEBUG_LOG(1,"Waiting for signal @ cond %p from task %p",task->finishedCond,task);
		pthread_cond_wait(task->finishedCond,task->mutex);
	} else {
		DEBUG_LOG(1,"Task %p has already finished",task);
	}

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
char *ghost_task_strstate(int state)
{
	switch (state) {
		case 0: 
			return "Invalid";
			break;
		case 1: 
			return "Enqueued";
			break;
		case 2: 
			return "Running";
			break;
		case 3: 
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
int ghost_task_waitall()
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
int ghost_task_waitsome(ghost_task_t ** tasks, int nt, int *index)
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
int ghost_task_destroy(ghost_task_t *t)
{
	pthread_mutex_destroy(t->mutex);
	pthread_cond_destroy(t->finishedCond);
	
	free(t->cores);
	free(t->state);
	free(t->ret);
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
ghost_task_t * ghost_task_init(int nThreads, int LD, void *(*func)(void *), void *arg, int flags)
{
	ghost_task_t *t = (ghost_task_t *)ghost_malloc(sizeof(ghost_task_t));
	if (ghost_thpool == NULL) {
		int nt = ghost_getNumberOfPhysicalCores()/ghost_getNumberOfRanksOnNode();
		int ft = ghost_getLocalRank()*nt;
		int poolThreads[] = {nt,nt};
		int firstThread[] = {ft,ft};
		int levels = ghost_getNumberOfHwThreads()/ghost_getNumberOfPhysicalCores();
		int totalthreads = nt*levels;
		WARNING_LOG("Trying to initialize a task but the thread pool has not yet been initialized. Doing the init now with %d threads!",totalthreads);
		ghost_thpool_init(poolThreads,firstThread,levels);
	}
	if (taskq == NULL) {
		DEBUG_LOG(1,"Trying to initialize a task but the task queues have not yet been initialized. Doing the init now...");
		ghost_taskq_init();
	}

	if (nThreads == GHOST_TASK_FILL_LD) {
		if (LD < 0) {
			WARNING_LOG("FILL_LD does only work when the LD is given! Not adding task!");
			return NULL;
		}
		t->nThreads = nThreadsPerLD(LD);
	} 
	else if (nThreads == GHOST_TASK_FILL_ALL) {
#ifdef GHOST_OPENMP
		t->nThreads = ghost_thpool->nThreads;
#else
		t->nThreads = 1; //TODO is this the correct behavior?
#endif
	} 
	else {
		t->nThreads = nThreads;
	}

	t->LD = LD;
	t->func = func;
	t->arg = arg;
	t->flags = flags;

	t->freed = 0;
	t->state = (int *)ghost_malloc(sizeof(int));
	*(t->state) = GHOST_TASK_INVALID;
	t->cores = (int *)ghost_malloc(sizeof(int)*t->nThreads);
	t->next = NULL;
	t->prev = NULL;
	t->parent = NULL;
	t->finishedCond = (pthread_cond_t *)ghost_malloc(sizeof(pthread_cond_t));
	t->mutex = (pthread_mutex_t *)ghost_malloc(sizeof(pthread_mutex_t));

	return t;
}

/**
 * @brief Free all resources of the thread pool
 *
 * @return GHOST_SUCCESS on success or GHOST_FAILURE on failure.
 */
int ghost_thpool_finish()
{
	if (ghost_thpool == NULL)
		return GHOST_SUCCESS;

	free(ghost_thpool->threads);
	free(ghost_thpool->sem);
	free(ghost_thpool->PUs);
	free(ghost_thpool);

	return GHOST_SUCCESS;
}
