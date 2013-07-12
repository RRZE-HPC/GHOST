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
#include "ghost_cpuid.h"

#define CLR_BIT(field,bit) (field[bit/(sizeof(int)*8)] &= ~(1 << (bit%(sizeof(int)*8))))
#define SET_BIT(field,bit) (field[bit/(sizeof(int)*8)] |=  (1 << (bit%(sizeof(int)*8))))
#define TGL_BIT(field,bit) (field[bit/(sizeof(int)*8)] ^=  (1 << (bit%(sizeof(int)*8))))
#define CHK_BIT(field,bit) (field[bit/(sizeof(int)*8)]  &  (1 << (bit%(sizeof(int)*8))))

static ghost_taskq_t **taskqs;
static ghost_thpool_t *thpool;
static int nTasks = 0;
static sem_t taskSem;
static int killed = 0;
static int threadstate[GHOST_MAX_THREADS/sizeof(int)/8] = {0};
static pthread_mutex_t globalMutex;

static void * thread_do(void *arg);

static void printThreadstate()
{
	int b;

	for(b=0; b<GHOST_MAX_THREADS; b++)
		printf(CHK_BIT(threadstate,b)?"1":"0");

	printf("\n");

}
static int intcomp(const void *x, const void *y) 
{
	return (*(int *)x - *(int *)y);
}


int ghost_thpool_init(int nThreads){
	int t;


	if ((uint32_t)nThreads > ghost_cpuid_topology.numHWThreads) {
		WARNING_LOG("Trying to create more threads than there are hardware threads. Setting no. of threads to %u",ghost_cpuid_topology.numHWThreads);
		nThreads = ghost_cpuid_topology.numHWThreads;
	}

	if (nThreads < 1) {
		WARNING_LOG("Invalid number of threads given for thread pool (%d), setting to 1",nThreads);
		nThreads=1;
	}

	int oldLDs[nThreads];

	thpool = (ghost_thpool_t*)ghost_malloc(sizeof(ghost_thpool_t));
	thpool->nLDs = 2; // TODO
	thpool->threads = (pthread_t *)ghost_malloc(nThreads*sizeof(pthread_t));
	thpool->taskDoneConds = (pthread_cond_t *)ghost_malloc(nThreads*sizeof(pthread_cond_t));
	thpool->nThreadsPerLD = (int *)ghost_malloc(thpool->nLDs*sizeof(int));
	thpool->firstThreadOfLD = (int *)ghost_malloc((thpool->nLDs+1)*sizeof(int));
	thpool->LDs = (int *)ghost_malloc(nThreads*sizeof(int));
	thpool->nThreads = nThreads;
	thpool->idleThreads = nThreads;
	thpool->sem = (sem_t*)ghost_malloc(sizeof(sem_t));
	sem_init(thpool->sem, 0, 0);
	sem_init(&taskSem, 0, 0);
	pthread_mutex_init(&globalMutex,NULL);

	for (t=0; t<thpool->nLDs; t++)
		thpool->nThreadsPerLD[t] = 0;

	DEBUG_LOG(1,"Creating thread pool with %d threads on %d LDs", nThreads, thpool->nLDs);
	// sort and normalize LDs (avoid errors when there are, e.g. only sockets 0 and 3 on a system)	
	qsort(oldLDs,thpool->nThreads,sizeof(int),intcomp);

	thpool->firstThreadOfLD[0] = 0;
	int curLD = 0;
	for (t=0; t<nThreads; t++){
		oldLDs[t] = ghost_cpuid_topology.threadPool[t].packageId;
		if ((t > 0) && (oldLDs[t] != oldLDs[t-1])) // change in LD
		{
			curLD++;
			thpool->firstThreadOfLD[curLD] = t;
		}
		thpool->LDs[t] = curLD;
		thpool->nThreadsPerLD[curLD]++;
		DEBUG_LOG(1,"Thread %d @ LD %d, # threads @ LD: %d",t,thpool->LDs[t],thpool->nThreadsPerLD[curLD]);
	}
	thpool->firstThreadOfLD[thpool->nLDs] = nThreads;
	
	for (t=0; t<nThreads; t++){
		pthread_cond_init(&thpool->taskDoneConds[t],NULL);

		pthread_create(&(thpool->threads[t]), NULL, thread_do, (void *)(intptr_t)t);

	}
	for (t=0; t<nThreads; t++){
		sem_wait(thpool->sem);
	}
	DEBUG_LOG(1,"All threads are initialized and waiting for tasks");



	return GHOST_SUCCESS;
}

int ghost_taskq_init(int nqs)
{
	int q;
	DEBUG_LOG(1,"There will be %d task queues",nqs);
	thpool->nLDs = nqs;

	taskqs=(ghost_taskq_t**)ghost_malloc(thpool->nLDs*sizeof(ghost_taskq_t *));

	for (q=0; q<thpool->nLDs; q++) {
		taskqs[q] = (ghost_taskq_t *)ghost_malloc(sizeof(ghost_taskq_t));
		DEBUG_LOG(1,"taskq # %d: %p",q,taskqs[q]);
		taskqs[q]->tail = NULL;
		taskqs[q]->head = NULL;
		taskqs[q]->nTasks = 0;
		taskqs[q]->sem = (sem_t*)malloc(sizeof(sem_t));
		sem_init(taskqs[q]->sem, 0, 0);
		pthread_mutex_init(&(taskqs[q]->mutex),NULL);
	}


	return GHOST_SUCCESS;
}

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
			q->head->next = NULL;
	}

	if (t->prev != NULL)
		t->prev->next = t->next;

	if (t->next != NULL)
		t->next->prev = t->prev;


	return GHOST_SUCCESS;
}


static ghost_task_t * taskq_findAndDeleteTask(ghost_taskq_t *q)
{
	DEBUG_LOG(1,"Trying to find a suited task in q %p",q);
	if (q == NULL) {
		WARNING_LOG("Tried to find a job but the queue is NULL");
		return NULL;
	}
	if (q->head == NULL) {
		WARNING_LOG("Empty queue, returning NULL!");
		return NULL;
	}


	ghost_task_t *curTask = q->head;
	ghost_task_t *prevTask;

	while(curTask != NULL)
	{
		prevTask = curTask->prev;

		if (thpool->idleThreads >= curTask->nThreads)
		{
			DEBUG_LOG(1,"Found a suiting task!");

			// if task has siblings, delete them here
			if (curTask->siblings != NULL)
			{
				DEBUG_LOG(1,"Deleting siblings of task");
			}

			DEBUG_LOG(1,"Deleting task itself");
			taskq_deleteTask(q,curTask);	
			return curTask;
		} else {
			DEBUG_LOG(1,"Skipping task %p because it needs %d threads and only %d threads are idle",curTask,curTask->nThreads,thpool->idleThreads);
		}

		curTask = curTask->next;

	}

	DEBUG_LOG(1,"Could not find and delete a task, returning NULL");
	return NULL;


}


static void * thread_do(void *arg)
{
	ghost_task_t *myTask;
	ghost_taskq_t *curQueue;
	intptr_t core = (intptr_t)arg;
	int t,q;
	void*(*tFunc)(void* arg);
	void*  tArg;

	ghost_setCore(core);
	int myCore = ghost_getCore();
	int myLD = thpool->LDs[core];
	DEBUG_LOG(1,"Thread %d: Pinning to core %d (%d) on LD %d",(int)pthread_self(),myCore,(int)core,myLD);
	
	sem_post(thpool->sem);
	
	while ((!killed) || (nTasks > 0)) // as long as there are jobs stay alive
	{

		int sval;
		sem_getvalue(&taskSem,&sval);
		WARNING_LOG("Thread %d: Waiting for tasks (taskSem: %d)...",(int)pthread_self(),sval);
		
		if (sem_wait(&taskSem)) 
			ABORT("Waiting for tasks failed: %s",strerror(errno));

		sem_getvalue(&taskSem,&sval);

		WARNING_LOG("Thread %d: Finished waiting for a task, now there are %d tasks (taskSem: %d) available and killed=%d",(int)pthread_self(),nTasks,sval,killed);
		if (killed && (nTasks <= 0)) {
			DEBUG_LOG(1,"Thread %d: Not executing any further tasks",(int)pthread_self());
			//pthread_mutex_unlock(&globalMutex);
			sem_post(&taskSem);
			break;
		}
		pthread_mutex_lock(&globalMutex);
		nTasks--;	
		pthread_mutex_unlock(&globalMutex);	

		curQueue = taskqs[myLD];	
		DEBUG_LOG(1,"Thread %d: Trying to find task in queue %d: %p",(int)pthread_self(),myLD,curQueue);	

		pthread_mutex_lock(&curQueue->mutex);
		myTask = taskq_findAndDeleteTask(curQueue);
		pthread_mutex_unlock(&curQueue->mutex);

		if (myTask  == NULL) { // steal work

			DEBUG_LOG(1,"Thread %d: Trying to steal work from other queues",(int)pthread_self());	

			for (q=0; q<thpool->nLDs && q!=myLD; q++) {
				curQueue = taskqs[q];	
				DEBUG_LOG(1,"Thread %d: Trying to find task in queue %d: %p",(int)pthread_self(),q,curQueue);	


				pthread_mutex_lock(&curQueue->mutex);
				myTask = taskq_findAndDeleteTask(curQueue);
				pthread_mutex_unlock(&curQueue->mutex);

			}
			if (myTask  == NULL) // no suiting task found
			{
				DEBUG_LOG(1,"Thread %d: Could not find a suited task in any queue",(int)pthread_self());
				//pthread_mutex_unlock(&globalMutex);	
				pthread_mutex_lock(&globalMutex);
				nTasks++;	
				pthread_mutex_unlock(&globalMutex);	
				sem_post(&taskSem);
				continue;
			}
		}
		
		tFunc = myTask->func;
		tArg = myTask->arg;

		//pthread_mutex_unlock(&globalMutex);	

		DEBUG_LOG(1,"Thread %d: Got a task: %p",(int)pthread_self(),myTask);

		omp_set_num_threads(myTask->nThreads);
		int reservedCores = 0;
		t = thpool->firstThreadOfLD[myLD];
#pragma omp parallel
#pragma omp critical
		{
			pthread_mutex_lock(&globalMutex);
			DEBUG_LOG(1,"Thread %d: OMP thread %d Looking for idle cores between %d and %d",
					(int)pthread_self(),omp_get_thread_num(),t,thpool->firstThreadOfLD[myLD+1]-1);

			for (; (t<thpool->firstThreadOfLD[myLD+1]) && (reservedCores < myTask->nThreads); t++) {
				if (!(CHK_BIT(threadstate,t))) {
					DEBUG_LOG(1,"Thread %d: Core # %d is idle, using it",(int)pthread_self(),t);
					ghost_setCore(t);
					reservedCores++;
					SET_BIT(threadstate,t);
					SET_BIT(myTask->cores,t);
					t++; // start from next core for next OMP thread
					break;
				}
			}
			pthread_mutex_unlock(&globalMutex);	
		}

		myTask->executingThreadNo == myCore;
		myTask->state = GHOST_TASK_RUNNING;	
		
		myTask->ret = tFunc(tArg);
		
		myTask->state = GHOST_TASK_FINISHED;
		pthread_cond_broadcast(&thpool->taskDoneConds[myCore]);
			
		pthread_mutex_lock(&globalMutex);
		for (t=thpool->firstThreadOfLD[myLD]; t<thpool->firstThreadOfLD[myLD+1]; t++) {
			if (CHK_BIT(myTask->cores,t)) {
				DEBUG_LOG(1,"Thread %d: Setting core # %d to idle again",(int)pthread_self(),t);
				CLR_BIT(threadstate,t);
			}
		}
		pthread_mutex_unlock(&globalMutex);	
	}
	DEBUG_LOG(1,"Thread %d: Exited infinite loop",(int)pthread_self());
	return NULL;
}

int ghost_task_print(ghost_task_t *t) {
	ghost_printHeader("Task %p",t);
	ghost_printLine("No. of threads",NULL,"%d",t->nThreads);
	ghost_printLine("LD",NULL,"%d",t->LD);
	if (t->siblings != NULL) {
		ghost_printLine("First sibling",NULL,"%p",t->siblings[0]);
		ghost_printLine("Last sibling",NULL,"%p",t->siblings[thpool->nLDs-1]);
	}
	ghost_printFooter();

	return GHOST_SUCCESS;
}


static int taskq_additem(ghost_taskq_t *q, ghost_task_t *t)
{

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

int ghost_task_add(ghost_task_t *t)
{
	int q;

	t->cores = (int *)ghost_malloc(GHOST_MAX_THREADS/8);
	memset(t->cores,0,GHOST_MAX_THREADS/8);

	if (t->LD == GHOST_TASK_LD_ANY) // add to all queues
	{
		DEBUG_LOG(1,"This task goes to all queues");
		t->siblings = (ghost_task_t **)ghost_malloc(thpool->nLDs*sizeof(ghost_task_t *));

		DEBUG_LOG(1,"Cloning task...");
		for (q=0; q<thpool->nLDs; q++) // create siblings
		{
			t->siblings[q] = (ghost_task_t *)ghost_malloc(sizeof(ghost_task_t));
		}
		// TODO fuse loops?	
		for (q=0; q<thpool->nLDs; q++) // clone 
		{
			memcpy(t->siblings[q],t,sizeof(ghost_task_t));
			t->siblings[q]->LD = q;
			taskq_additem(taskqs[q],t);
		}
	} 
	else 
	{
		DEBUG_LOG(1,"This task goes to LD %d",t->LD);
		if (t->LD > thpool->nLDs) {
			WARNING_LOG("Task shall go to LD %d but there are only %d LDs. Setting LD to zero...", t->LD, thpool->nLDs);
			t->LD = 0;
		}
		t->siblings = NULL;
		taskq_additem(taskqs[t->LD],t);
	}

	pthread_mutex_lock(&globalMutex);
	nTasks++;	
	pthread_mutex_unlock(&globalMutex);	
	sem_post(&taskSem);

	t->state = GHOST_TASK_ENQUEUED;

	DEBUG_LOG(1,"Task added successfully");

	return GHOST_SUCCESS;
}

int ghost_taskq_finish()
{
	int t;

	killed = 1;

	DEBUG_LOG(0,"Wake up all tasks");	
//	for (t=0; t<thpool->nThreads; t++) // wake up waiting threads, i.e., post a task for each thread
//	{ 	
//		if (CHK_BIT(threadstate,t)) {
//			DEBUG_LOG(1,"Waiting for thread %d to be finished",t);
//			pthread_mutex_t mutex;
//			pthread_mutex_lock(&mutex);
//			pthread_cond_wait(&thpool->taskDoneConds[t],&mutex);
//		}
		if (sem_post(&taskSem)){
			WARNING_LOG("Error in sem_post: %s",strerror(errno));
			return GHOST_FAILURE;
		}
//	}
	DEBUG_LOG(0,"Join all threads");	
	for (t=0; t<thpool->nThreads; t++)
	{ 		
		if (pthread_join(thpool->threads[t],NULL)){
			return GHOST_FAILURE;
		}
	}

	return GHOST_SUCCESS;	


}

int ghost_task_test(ghost_task_t * t)
{
	return t->state;
}

int ghost_task_wait(ghost_task_t * t)
{
	if (t->state == GHOST_TASK_RUNNING) {
		pthread_mutex_t mutex;

		pthread_mutex_lock(&mutex);
		pthread_cond_wait(&thpool->taskDoneConds[t->executingThreadNo],&mutex);

	}

	return GHOST_SUCCESS;

}
