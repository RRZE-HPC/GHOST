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
#include "cpuid.h"

#define CLR_BIT(field,bit) (field[(bit)/(sizeof(int)*8)] &= ~(1 << ((bit)%(sizeof(int)*8))))
#define SET_BIT(field,bit) (field[(bit)/(sizeof(int)*8)] |=  (1 << ((bit)%(sizeof(int)*8))))
#define TGL_BIT(field,bit) (field[(bit)/(sizeof(int)*8)] ^=  (1 << ((bit)%(sizeof(int)*8))))
#define CHK_BIT(field,bit) (field[(bit)/(sizeof(int)*8)]  &  (1 << ((bit)%(sizeof(int)*8))))

static ghost_taskq_t **taskqs; // the task queues (one per LD)
ghost_thpool_t *ghost_thpool; // the thread pool
static int nTasks = 0; // the total number of tasks in all queues 
static sem_t taskSem;  // a semaphore for the total number of tasks in all queues
static int killed = 0; // this will be set to 1 if the queues should be killed
static pthread_mutex_t globalMutex; // protect accesses to global variables
static pthread_cond_t anyTaskFinishedCond; // will be broadcasted if a task has finished

static void * thread_main(void *arg);

static void printThreadstate(int *coreState)
{
	UNUSED(printThreadstate);
	int b;

	for(b=0; b<GHOST_MAX_THREADS; b++)
		printf(CHK_BIT(coreState,b)?"1":"0");

	printf("\n");

}

static int intcomp(const void *x, const void *y) 
{
	return (*(int *)x - *(int *)y);
}


int ghost_thpool_init(int nThreads)
{
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

	ghost_thpool = (ghost_thpool_t*)ghost_malloc(sizeof(ghost_thpool_t));
	ghost_thpool->nLDs = 2; // TODO
	ghost_thpool->threads = (pthread_t *)ghost_malloc(nThreads*sizeof(pthread_t));
	ghost_thpool->firstThreadOfLD = (int *)ghost_malloc((ghost_thpool->nLDs+1)*sizeof(int));
	ghost_thpool->LDs = (int *)ghost_malloc(nThreads*sizeof(int));
	ghost_thpool->nThreads = nThreads;
	ghost_thpool->nIdleCores = nThreads;
	ghost_thpool->sem = (sem_t*)ghost_malloc(sizeof(sem_t));
	sem_init(ghost_thpool->sem, 0, 0);
	sem_init(&taskSem, 0, 0);
	pthread_mutex_init(&globalMutex,NULL);


	DEBUG_LOG(1,"Creating thread pool with %d threads on %d LDs", nThreads, ghost_thpool->nLDs);
	// sort and normalize LDs (avoid errors when there are, e.g. only sockets 0 and 3 on a system)	
	for (t=0; t<nThreads; t++){
		oldLDs[t] = ghost_cpuid_topology.threadPool[t].packageId;
	}
	qsort(oldLDs,ghost_thpool->nThreads,sizeof(int),intcomp);

	ghost_thpool->firstThreadOfLD[0] = 0;
	int curLD = 0;
	for (t=0; t<nThreads; t++){
		if ((t > 0) && (oldLDs[t] != oldLDs[t-1])) // change in LD
		{
			curLD++;
			ghost_thpool->firstThreadOfLD[curLD] = t;
		}
		ghost_thpool->LDs[t] = curLD;
		//	ghost_thpool->idleThreadsPerLD[curLD]++;
		DEBUG_LOG(1,"Thread %d @ LD %d",t,ghost_thpool->LDs[t]);
	}
	ghost_thpool->firstThreadOfLD[ghost_thpool->nLDs] = nThreads;

	for (t=0; t<nThreads; t++){
		pthread_create(&(ghost_thpool->threads[t]), NULL, thread_main, (void *)(intptr_t)t);
	}
	for (t=0; t<nThreads; t++){
		sem_wait(ghost_thpool->sem);
	}
	DEBUG_LOG(1,"All threads are initialized and waiting for tasks");



	return GHOST_SUCCESS;
}

int ghost_taskq_init(int nqs)
{
	int q;
	DEBUG_LOG(1,"There will be %d task queues",nqs);
	ghost_thpool->nLDs = nqs;

	taskqs=(ghost_taskq_t**)ghost_malloc(ghost_thpool->nLDs*sizeof(ghost_taskq_t *));

	for (q=0; q<ghost_thpool->nLDs; q++) {
		taskqs[q] = (ghost_taskq_t *)ghost_malloc(sizeof(ghost_taskq_t));
		DEBUG_LOG(1,"taskq # %d: %p",q,taskqs[q]);
		taskqs[q]->tail = NULL;
		taskqs[q]->head = NULL;
		taskqs[q]->LD = q;
		taskqs[q]->nIdleCores = ghost_thpool->firstThreadOfLD[q+1]-ghost_thpool->firstThreadOfLD[q];
		taskqs[q]->coreState = (int *)ghost_malloc(GHOST_MAX_THREADS/8);
		memset(taskqs[q]->coreState,0,GHOST_MAX_THREADS/8);
		pthread_mutex_init(&(taskqs[q]->mutex),NULL);
	}

	pthread_cond_init(&anyTaskFinishedCond,NULL);

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



static ghost_task_t * taskq_findDeleteAndPinTask(ghost_taskq_t *q)
{
	//DEBUG_LOG(1,"Trying to find a suited task in q %p",q);
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
		if ((curTask->flags & GHOST_TASK_LD_STRICT) && (q->LD != curTask->LD)) {
			DEBUG_LOG(1,"Skipping task %p because I'm looking in LD%d but the task has to be executed exclusively in LD%d",curTask,q->LD,curTask->LD);
			curTask = curTask->next;
			continue;
		}
		if (ghost_thpool->nIdleCores < curTask->nThreads) {
			DEBUG_LOG(1,"Skipping task %p because it needs %d threads and only %d threads are idle",curTask,curTask->nThreads,ghost_thpool->nIdleCores);
			curTask = curTask->next;
			continue;
		}
		if ((curTask->flags & GHOST_TASK_LD_STRICT) && (q->nIdleCores < curTask->nThreads)) {
			DEBUG_LOG(1,"Skipping task %p because it has to be executed exclusively in LD%d and there are only %d cores available (the task needs %d)",curTask,curTask->LD,q->nIdleCores,curTask->nThreads);
			curTask = curTask->next;
			continue;
		}


		DEBUG_LOG(1,"Found a suiting task: %p! task->nThreads=%d, nIdleCores[LD%d]=%d, nIdleCores=%d",curTask,curTask->nThreads,q->LD,q->nIdleCores,ghost_thpool->nIdleCores);

		// if task has siblings, delete them here
		if (curTask->siblings != NULL)
		{
			DEBUG_LOG(1,"The task is an LD_ANY-task. Deleting siblings of task");
			int s;
			for (s=0; s<ghost_thpool->nLDs; s++) {
				if (s!=q->LD) {
					pthread_mutex_lock(&taskqs[s]->mutex);
				}
				taskq_deleteTask(taskqs[s],curTask->siblings[s]);
				if (s!=q->LD) {
					pthread_mutex_unlock(&taskqs[s]->mutex);
				}
			}
		} else {
			DEBUG_LOG(1,"Deleting task itself");
			taskq_deleteTask(q,curTask);	
		}
		DEBUG_LOG(1,"Pinning the task's threads");
		omp_set_num_threads(curTask->nThreads);

		ghost_thpool->nIdleCores -= curTask->nThreads;
		int reservedCores = 0;

		int self;
		int t;
		int foundCore;
		int curThread;
		int curLD = q->LD; // setting this value here ensures that no thread searches in already fully occupied LDs

		for (curThread=0; curThread<curTask->nThreads; curThread++)
		{
#pragma omp parallel
			{
				if (omp_get_thread_num() == curThread)
				{ // this block will be entered by one thread at a time in ascending order

					foundCore = 0;
					self = 0;

					for (; (curLD<ghost_thpool->nLDs) && (reservedCores < curTask->nThreads); curLD++)
					{
						if ((self == 1) && (curLD == q->LD))
							continue;

						if (curLD != q->LD) // the own queue is already locked
							pthread_mutex_lock(&taskqs[curLD]->mutex);

						DEBUG_LOG(1,"Thread %d looking for an empty core @ LD%d",curThread,curLD);

						for (t = 0; t < (ghost_thpool->firstThreadOfLD[curLD+1]-ghost_thpool->firstThreadOfLD[curLD]); t++) 
						{
							if (!(CHK_BIT(taskqs[curLD]->coreState,t))) 
							{
								DEBUG_LOG(1,"Thread %d: Core # %d is idle, using it, idle cores @ LD%d: %d",
										(int)pthread_self(),curTask->cores[reservedCores],curLD,taskqs[curLD]->nIdleCores);

								taskqs[curLD]->nIdleCores --;
								ghost_setCore(ghost_thpool->firstThreadOfLD[curLD]+t);
								curTask->cores[reservedCores] = ghost_thpool->firstThreadOfLD[curLD]+t;
								SET_BIT(taskqs[curLD]->coreState,t);
								reservedCores++;
								foundCore = 1;
								break;
							}
						}

						if (curLD != q->LD)
							pthread_mutex_unlock(&taskqs[curLD]->mutex);

						if (foundCore)
						{ // found a core for this thread, proceed to next thread
							break;
						}



						if (self == 0) // own LD has been looked in, start from zero now
						{
							curLD = -1; // will be 0 when re-entering the for-loop
							self = 1;
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

static void * thread_main(void *arg)
{
	ghost_task_t *myTask;
	ghost_taskq_t *curQueue;
	intptr_t core = (intptr_t)arg;
	int t,q;
	void*(*tFunc)(void* arg);
	void*  tArg;

	ghost_setCore(core);
	int myCore = ghost_getCore();
	int myLD = ghost_thpool->LDs[core];
	int curLD = 0;
	DEBUG_LOG(1,"Thread %d: Pinning to core %d (%d) on LD %d",(int)pthread_self(),myCore,(int)core,myLD);

	sem_post(ghost_thpool->sem);

	while ((!killed) || (nTasks > 0)) // as long as there are jobs stay alive
	{

		int sval;
		sem_getvalue(&taskSem,&sval);
		DEBUG_LOG(1,"Thread %d: Waiting for tasks (taskSem: %d)...",(int)pthread_self(),sval);

		if (sem_wait(&taskSem)) 
		{
			if (errno == EINTR)
				continue;
			ABORT("Waiting for tasks failed: %s",strerror(errno));
		}

		sem_getvalue(&taskSem,&sval);

		DEBUG_LOG(1,"Thread %d: Finished waiting for a task, now there are %d tasks (taskSem: %d) available and killed=%d",(int)pthread_self(),nTasks,sval,killed);
		if (killed && (nTasks <= 0)) {
			DEBUG_LOG(1,"Thread %d: Not executing any further tasks",(int)pthread_self());
			sem_post(&taskSem);
			break;
		}
		pthread_mutex_lock(&globalMutex);
		nTasks--;	
		pthread_mutex_unlock(&globalMutex);	


		int self = 0;
		q = myLD;
		for (; q<ghost_thpool->nLDs; q++) 
		{ // search for task in any queue, starting from the own queue
			if ((self == 1) && (q == myLD)) // skip own queue if it has been looked in
				continue;

			curQueue = taskqs[q];
			DEBUG_LOG(1,"Thread %d: Trying to find task in queue %d: %p",(int)pthread_self(),myLD,curQueue);	
			
			pthread_mutex_lock(&curQueue->mutex);
			myTask = taskq_findDeleteAndPinTask(curQueue);
			pthread_mutex_unlock(&curQueue->mutex);
			if (myTask != NULL) {
				break;
			}

			if (self == 0) // own LD has been looked in, start from zero now
			{
				q = -1;
				self = 1; // after the first iteration, the own LD has been dealt with
			}
		}
		if (myTask  == NULL) // no suiting task found
		{
			DEBUG_LOG(1,"Thread %d: Could not find a suited task in any queue",(int)pthread_self());
			pthread_mutex_lock(&globalMutex);
			nTasks++;	
			pthread_mutex_unlock(&globalMutex);	
			sem_post(&taskSem);
		//	usleep(1); // give other threads a chance
			continue;
		}
			
		*(myTask->executingThreadNo) = myCore;
		*(myTask->state) = GHOST_TASK_RUNNING;	
		
		tFunc = myTask->func;
		tArg = myTask->arg;


		DEBUG_LOG(1,"Thread %d: Finally executing task: %p",(int)pthread_self(),myTask);

		myTask->ret = tFunc(tArg);
		
		DEBUG_LOG(1,"Thread %d: Finished executing task: %p",(int)pthread_self(),myTask);
		
		ghost_thpool->nIdleCores += myTask->nThreads;
		for (t=0; t<myTask->nThreads; t++) {
			for (curLD = 0; curLD < ghost_thpool->nLDs; curLD++) {
				if ((myTask->cores[t]>=ghost_thpool->firstThreadOfLD[curLD]) && (myTask->cores[t]<ghost_thpool->firstThreadOfLD[curLD+1])) {
					//pthread_mutex_lock(&globalMutex);
					pthread_mutex_lock(&taskqs[curLD]->mutex);
					DEBUG_LOG(1,"%d-th thread of task resided @ LD%d",t,curLD);
					taskqs[curLD]->nIdleCores++;
					DEBUG_LOG(1,"Thread %d: Setting core # %d (loc: %d) to idle again, idle cores @ LD%d: %d",(int)pthread_self(),myTask->cores[t],myTask->cores[t]-ghost_thpool->firstThreadOfLD[curLD],curLD,taskqs[curLD]->nIdleCores);
					CLR_BIT(taskqs[curLD]->coreState,myTask->cores[t]-ghost_thpool->firstThreadOfLD[curLD]);
					//pthread_mutex_unlock(&globalMutex);	
					pthread_mutex_unlock(&taskqs[curLD]->mutex);
					break;
				}
			}
		}
		pthread_mutex_lock(myTask->mutex); // TODO will this cause race conditions?
		DEBUG_LOG(1,"Thread %d: Finished with task %p. Setting state to finished...",(int)pthread_self(),myTask);
		*(myTask->state) = GHOST_TASK_FINISHED;
		pthread_mutex_unlock(myTask->mutex);
		DEBUG_LOG(1,"Thread %d: Finished ith task %p. Sending signal to all waiters (cond: %p)...",(int)pthread_self(),myTask,myTask->finishedCond);
		pthread_cond_broadcast(myTask->finishedCond);
	}
	DEBUG_LOG(1,"Thread %d: Exited infinite loop",(int)pthread_self());
	return NULL;
}

int ghost_task_print(ghost_task_t *t) 
{
	ghost_printHeader("Task %p",t);
	ghost_printLine("No. of threads",NULL,"%d",t->nThreads);
	ghost_printLine("LD",NULL,"%d",t->LD);
	if (t->siblings != NULL) {
		ghost_printLine("First sibling",NULL,"%p",t->siblings[0]);
		ghost_printLine("Last sibling",NULL,"%p",t->siblings[ghost_thpool->nLDs-1]);
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

	// if a task is initialized _once_ but added several times, this has to be done each time it is added
	pthread_cond_init(t->finishedCond,NULL);
	pthread_mutex_init(t->mutex,NULL);
	*(t->state) = GHOST_TASK_INVALID;
	memset(t->cores,0,sizeof(int)*t->nThreads);

	if (t->LD == GHOST_TASK_LD_ANY) // add to all queues
	{
		DEBUG_LOG(1,"Task %p goes to all queues",t);
		t->siblings = (ghost_task_t **)ghost_malloc(ghost_thpool->nLDs*sizeof(ghost_task_t *));

		DEBUG_LOG(1,"Cloning task...");
		for (q=0; q<ghost_thpool->nLDs; q++)
		{
			t->siblings[q] = (ghost_task_t *)ghost_malloc(sizeof(ghost_task_t));
			t->siblings[q]->nThreads = t->nThreads;
			t->siblings[q]->LD = q;
			t->siblings[q]->flags = t->flags;
			t->siblings[q]->func = t->func;
			t->siblings[q]->arg = t->arg;
			t->siblings[q]->state = t->state;
			t->siblings[q]->executingThreadNo = t->executingThreadNo;
			t->siblings[q]->cores = t->cores;
			t->siblings[q]->ret = t->ret;
			t->siblings[q]->siblings = t->siblings;
			t->siblings[q]->finishedCond = t->finishedCond;
			t->siblings[q]->mutex = t->mutex;
			DEBUG_LOG(1,"Adding sibling to queue %p (LD %d)",taskqs[q],q);
			taskq_additem(taskqs[q],t->siblings[q]);
		}
	} 
	else 
	{
		DEBUG_LOG(1,"This task goes to  queue %p (LD %d)",taskqs[t->LD],t->LD);
		if (t->LD > ghost_thpool->nLDs) {
			WARNING_LOG("Task shall go to LD %d but there are only %d LDs. Setting LD to zero...", t->LD, ghost_thpool->nLDs);
			t->LD = 0;
		}
		t->siblings = NULL;
		taskq_additem(taskqs[t->LD],t);
	}
	*(t->state) = GHOST_TASK_ENQUEUED;

	pthread_mutex_lock(&globalMutex);
	nTasks++;	
	pthread_mutex_unlock(&globalMutex);	
	sem_post(&taskSem);


	DEBUG_LOG(1,"Task added successfully");

	return GHOST_SUCCESS;
}

int ghost_taskq_finish()
{
	int t;

	killed = 1;

	DEBUG_LOG(1,"Wake up all tasks");	
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

	return GHOST_SUCCESS;	


}

int ghost_task_test(ghost_task_t * t)
{
	if (t->state == NULL)
		return GHOST_TASK_INVALID;
	return *(t->state);
}

int ghost_task_wait(ghost_task_t * t)
{
	DEBUG_LOG(1,"Waiting for task %p which is managed by thread %d and whose state is %d",t,*(t->executingThreadNo),*(t->state));

		pthread_mutex_lock(t->mutex);
	if (*(t->state) != GHOST_TASK_FINISHED) {
		DEBUG_LOG(1,"Waiting for signal @ cond %p from task %p which is managed by thread %d",t->finishedCond,t,*(t->executingThreadNo));
		pthread_cond_wait(t->finishedCond,t->mutex);
	} else {
		DEBUG_LOG(1,"Task %p has already finished",t);
	}

	pthread_mutex_unlock(t->mutex);
	pthread_cond_broadcast(&anyTaskFinishedCond);
	DEBUG_LOG(1,"Finished waitung for task %p!",t);

	return GHOST_SUCCESS;

}

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

int ghost_task_waitall()
{
	int q;
	ghost_task_t *t;
	
	for (q=0; q<ghost_thpool->nLDs; q++)
	{
		t = taskqs[q]->head;
		while (t != NULL)
		{
			ghost_task_wait(t);
			t = t->next;
		}
	}

	return GHOST_SUCCESS;
}


int ghost_task_waitsome(ghost_task_t ** tasks, int nt, int *index)
{
	int t;
	int ret = 0;
	pthread_t threads[nt];
	pthread_mutex_t mutex;
	pthread_mutex_init(&mutex,NULL);

	for (t=0; t<nt; t++)
	{ // look if one of the tasks is already finished
		if (*(tasks[t]->state) == GHOST_TASK_FINISHED) 
		{ // one of the tasks is already finished
			DEBUG_LOG(1,"One of the tasks has already finished");
			ret = 1;
			index[t] = 1;
		} else {
			index[t] = 0;
		}
	}
	if (ret)
		return GHOST_SUCCESS;

	DEBUG_LOG(1,"None of the tasks has already finished. Waiting for (at least) one of them...");

	
	for (t=0; t<nt; t++)
	{
		pthread_create(&threads[t],NULL,(void *(*)(void *))&ghost_task_wait,tasks[t]);
	}

	pthread_mutex_lock(&mutex);
	pthread_cond_wait(&anyTaskFinishedCond,&mutex);

	for (t=0; t<nt; t++)
	{ // again look which tasks are finished
		if (*(tasks[t]->state) == GHOST_TASK_FINISHED) 
		{
			index[t] = 1;
		} else {
			index[t] = 0;
		}
	}

	return GHOST_SUCCESS;
}


int ghost_task_destroy(ghost_task_t *t)
{
	free(t->cores);
	free(t->siblings); // TODO free each sibling recursively
	free(t->state);
	free(t->executingThreadNo);
	free(t->ret);

	pthread_mutex_destroy(t->mutex);
	pthread_cond_destroy(t->finishedCond);

	return GHOST_SUCCESS;
}

ghost_task_t * ghost_task_init(int nThreads, int LD, void *(*func)(void *), void *arg, int flags)
{
	ghost_task_t *t = (ghost_task_t *)ghost_malloc(sizeof(ghost_task_t));
	
	if (nThreads == GHOST_TASK_FILL_LD) {
		if (LD < 0) {
			WARNING_LOG("FILL_LD does only work when the LD is given! Not adding task!");
			return NULL;
		}
		t->nThreads = ghost_thpool->firstThreadOfLD[LD+1]-ghost_thpool->firstThreadOfLD[LD];
	} 
	else if (nThreads == GHOST_TASK_FILL_ALL) {
	/*	if (LD < 0) {
			WARNING_LOG("FILL_ALL does only work when the LD is given! Not adding task!");
			return NULL;
		}*/
		t->nThreads = ghost_thpool->nThreads;
	} 
	else {
		t->nThreads = nThreads;
	}

	t->LD = LD;
	t->func = func;
	t->arg = arg;
	t->flags = flags;

	t->state = (int *)ghost_malloc(sizeof(int));
	t->executingThreadNo = (int *)ghost_malloc(sizeof(int));
	t->cores = (int *)ghost_malloc(sizeof(int)*t->nThreads);
	t->next = NULL;
	t->prev = NULL;
	t->siblings = NULL;
	t->finishedCond = (pthread_cond_t *)ghost_malloc(sizeof(pthread_cond_t));
	t->mutex = (pthread_mutex_t *)ghost_malloc(sizeof(pthread_mutex_t));

	//pthread_cond_init(t->finishedCond,NULL);
	//pthread_mutex_init(t->mutex,NULL);
	//*(t->state) = GHOST_TASK_INVALID;
	//memset(t->cores,0,sizeof(int)*t->nThreads);

	return t;
}
