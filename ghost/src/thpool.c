/* ********************************
 * 
 * Author:  Johan Hanssen Seferidis
 * Date:    12/08/2011
 * Update:  01/11/2011
 * License: LGPL
 * 
 * 
 *//** @file thpool.h *//*
 ********************************/

/* Library providing a threading pool where you can add work. For an example on 
 * usage you refer to the main file found in the same package */

/* 
 * Fast reminders:
 * 
 * tp           = threadpool 
 * thpool       = threadpool
 * thpool_t     = threadpool type
 * tp_p         = threadpool pointer
 * sem          = semaphore
 * xN           = x can be any string. N stands for amount
 * 
 * */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <errno.h>

#include "thpool.h"      /* here you can also find the interface to each function */
#include "ghost_util.h"

#define THREAD_IDLE 0
#define THREAD_BUSY 1

static int thpool_keepalive=1;

/* Create mutex variable */
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER; /* used to serialize queue access */

static int *threadstate;
static int idleThreads;


/* Initialise thread pool */
thpool_t* thpool_init(int threadsN){
	thpool_t* tp_p;
	threadstate = (int *)ghost_malloc(sizeof(int)*threadsN);
	idleThreads = threadsN;

	if (!threadsN || threadsN<1) threadsN=1;

	/* Make new thread pool */
	tp_p=(thpool_t*)malloc(sizeof(thpool_t));                              /* MALLOC thread pool */
	if (tp_p==NULL){
		fprintf(stderr, "thpool_init(): Could not allocate memory for thread pool\n");
		return NULL;
	}
	tp_p->threads=(pthread_t*)malloc(threadsN*sizeof(pthread_t));          /* MALLOC thread IDs */
	if (tp_p->threads==NULL){
		fprintf(stderr, "thpool_init(): Could not allocate memory for thread IDs\n");
		return NULL;
	}
	tp_p->threadsN=threadsN;

	/* Initialise the job queue */
	if (thpool_jobqueue_init(tp_p)==-1){
		fprintf(stderr, "thpool_init(): Could not allocate memory for job queue\n");
		return NULL;
	}

	/* Initialise semaphore*/
	tp_p->jobqueue->queueSem=(sem_t*)malloc(sizeof(sem_t));                 /* MALLOC job queue semaphore */
	tp_p->thSem=(sem_t*)malloc(sizeof(sem_t));                 /* MALLOC job queue semaphore */
	sem_init(tp_p->jobqueue->queueSem, 0, 0); /* no shared, initial value */
	sem_init(tp_p->thSem, 0, threadsN); /* no shared, initial value */

	/* Make threads in pool */
	int t;
	for (t=0; t<threadsN; t++){
		printf("Created thread %d in pool \n", t);
		threadstate[t] = THREAD_IDLE; 
		pthread_create(&(tp_p->threads[t]), NULL, thpool_thread_do, (void *)tp_p); /* MALLOCS INSIDE PTHREAD HERE */
	}

	return tp_p;
}


/* What each individual thread is doing 
 * */
/* There are two scenarios here. One is everything works as it should and second if
 * the thpool is to be killed. In that manner we try to BYPASS sem_wait and end each thread. */
void * thpool_thread_do(void *arg){

	//		int sval;
	int t;	
	int enoughThreads;
	//	int hasJob = 0;	
	thpool_t *tp_p = (thpool_t *)arg;
	while(thpool_keepalive || (tp_p->jobqueue->jobsN > 0)){
		//		printf("a\n");

		DEBUG_LOG(0,"%d: Waiting for work, jobsN = %d",(int)pthread_self(),tp_p->jobqueue->jobsN);
		if (sem_wait(tp_p->jobqueue->queueSem)) {/* WAITING until there is work in the queue */
			perror("thpool_thread_do(): Waiting for semaphore");
			exit(1);
		}

		//	if (thpool_keepalive || (tp_p->jobqueue->jobsN > 0)){

		/* Read job from queue and execute it */
		void*(*func_buff)(void* arg);
		void*  arg_buff;
		thpool_job_t* job_p;
		//	sem_getvalue(tp_p->jobqueue->queueSem,&sval);
		//DEBUG_LOG(0,"2: There is work (jobs in queue: %d/%d)",tp_p->jobqueue->jobsN,sval);

		pthread_mutex_lock(&mutex);                  /* LOCK */
		//	if (!hasJob)
		job_p = thpool_jobqueue_peek(tp_p);

		if (job_p == NULL) {
			WARNING_LOG("%d: Peeked NULL",(int)pthread_self());
			pthread_mutex_unlock(&mutex);                  /* LOCK */
			continue;
			//	ABORT("This should not happen. I peeked NULL but there are %d jobs left",tp_p->jobqueue->jobsN);
		}
		(tp_p->jobqueue->jobsN)--;
		//	hasJob = 1;
		//	pthread_mutex_unlock(&mutex);                /* UNLOCK */

		//		sem_getvalue(tp_p->thSem,&sval);
		//		DEBUG_LOG(0,"Before: There are %d threads available",sval);
		/*for (t=0; t<job_p->nThreads; t++) {
		  DEBUG_LOG(0,"Waiting for enough threads to be idle...");
		//				printf("%d \n",sem_trywait(tp_p->thSem));
		if (sem_trywait(tp_p->thSem)) {// WAITING until there are threads available
		if (errno == EAGAIN) {
		DEBUG_LOG(0,"Not enough threads for this job now");
		pthread_mutex_unlock(&mutex);                // LOCK 
		enoughThreads=0;
		break;

		} else {
		ABORT("Error in sem_wait for the thread semaphore");
		}
		/////			DEBUG_LOG(0,"Not enough threads available!");
		//			perror("thpool_thread_do(): Waiting for semaphore");
		//			exit(1);
		}
		enoughThreads = 1;
		}
		if (!enoughThreads) {
		pthread_mutex_unlock(&mutex);                  // LOCK 
		continue;
		}*/
		DEBUG_LOG(0,"%d: %d threads needed, %d threads idle",(int)pthread_self(),job_p->nThreads,idleThreads);
		/*if (idleThreads < job_p->nThreads) {
		  pthread_mutex_unlock(&mutex);                // LOCK 
		  continue;
		  }*/
		for (t=0; t<job_p->nThreads; t++) {
			if (sem_wait(tp_p->thSem)) {
					perror("thpool_thread_do(): Waiting for semaphore");
					exit(1);
			}

			/*if (sem_trywait(tp_p->thSem)) {// WAITING until there are threads available
				if (errno==EAGAIN) {
					pthread_mutex_unlock(&mutex);                  // UNLOCK 
					if (sem_wait(tp_p->thSem)) {// WAITING until there are threads available
						perror("thpool_thread_do(): Waiting for semaphore");
						exit(1);
					}
					pthread_mutex_lock(&mutex);                  // LOCK 
				} else {
					perror("thpool_thread_do(): Waiting for semaphore");
					exit(1);
				}

			} */

		}


		idleThreads -= job_p->nThreads;
		//		sem_getvalue(tp_p->thSem,&sval);
		//		DEBUG_LOG(0,"Before: Now there are %d threads available",sval);

		//	pthread_mutex_lock(&mutex);                  /* LOCK */



		DEBUG_LOG(0,"%d: Doing job %p",(int)pthread_self(),job_p);
		func_buff=job_p->function;
		arg_buff =job_p->arg;
		//thpool_jobqueue_removelast(tp_p);

		//	DEBUG_LOG(0,"There are %d idle threads",idleThreads);
		//	idleThreads -= job_p->nThreads;

		pthread_mutex_unlock(&mutex);                /* UNLOCK */


		func_buff(arg_buff);               			 /* run function */

		DEBUG_LOG(2,"Making threads available again");	
		for (t=0; t<job_p->nThreads; t++) { // "free" the threads
			if (sem_post(tp_p->thSem)) {
				ABORT("Error in sem_post for the thread semaphore");
			}
		}
		free(job_p);                                                       /* DEALLOC job */
		pthread_mutex_lock(&mutex);                  // LOCK 
		idleThreads += job_p->nThreads;
		//	hasJob = 0;
		//		sem_getvalue(tp_p->thSem,&sval);
		//		DEBUG_LOG(0,"After: There are %d threads available",sval);
		pthread_mutex_unlock(&mutex);                // UNLOCK 
		//	}
		//	else
		//	{
		//		return NULL; /* EXIT thread*/
		//	}
	}
	return NULL;
}


/* Add work to the thread pool */
int thpool_add_work(thpool_t* tp_p, void *(*function_p)(void*), void* arg_p, int nThreads){
	thpool_job_t* newJob;

	newJob=(thpool_job_t*)malloc(sizeof(thpool_job_t));                        /* MALLOC job */
	if (newJob==NULL){
		fprintf(stderr, "thpool_add_work(): Could not allocate memory for new job\n");
		exit(1);
	}

	/* add function and argument */
	newJob->function=function_p;
	newJob->arg=arg_p;
	newJob->nThreads=nThreads;

	/* add job to queue */
	pthread_mutex_lock(&mutex);                  /* LOCK */
	thpool_jobqueue_add(tp_p, newJob);
	pthread_mutex_unlock(&mutex);                /* UNLOCK */
	//	int sval;
	//	sem_getvalue(tp_p->jobqueue->queueSem, &sval);
	//	DEBUG_LOG(0,"There are now %d jobs in the queue",sval);

	return 0;
}


/* Destroy the threadpool */
void thpool_destroy(thpool_t* tp_p){
	int t;

	/* End each thread's infinite loop */
	thpool_keepalive=0; 

	/* Awake idle threads waiting at semaphore */
	for (t=0; t<(tp_p->threadsN); t++){
		if (sem_post(tp_p->jobqueue->queueSem)){
			fprintf(stderr, "thpool_destroy(): Could not bypass sem_wait()\n");
		}
	}
	DEBUG_LOG(1,"Thread pool about to be destroyed but %d jobs still in queue",tp_p->jobqueue->jobsN);


	/* Kill semaphore */
	//	if (sem_destroy(tp_p->thSem)!=0){
	//		fprintf(stderr, "thpool_destroy(): Could not destroy semaphore\n");
	//	}

	/* Wait for threads to finish */
	for (t=0; t<(tp_p->threadsN); t++){
		pthread_join(tp_p->threads[t], NULL);
	}
	DEBUG_LOG(1,"All work is done, ready to destroy the thread pool");
	if (sem_destroy(tp_p->jobqueue->queueSem)!=0){
		fprintf(stderr, "thpool_destroy(): Could not destroy semaphore\n");
	}

	thpool_jobqueue_empty(tp_p);

	/* Dealloc */
	free(tp_p->threads);                                                   /* DEALLOC threads             */
	free(tp_p->jobqueue->queueSem);                                        /* DEALLOC job queue semaphore */
	free(tp_p->jobqueue);                                                  /* DEALLOC job queue           */
	free(tp_p);                                                            /* DEALLOC thread pool         */
}



/* =================== JOB QUEUE OPERATIONS ===================== */



/* Initialise queue */
int thpool_jobqueue_init(thpool_t* tp_p){
	tp_p->jobqueue=(thpool_jobqueue*)malloc(sizeof(thpool_jobqueue));      /* MALLOC job queue */
	if (tp_p->jobqueue==NULL) return -1;
	tp_p->jobqueue->tail=NULL;
	tp_p->jobqueue->head=NULL;
	tp_p->jobqueue->jobsN=0;
	return 0;
}


/* Add job to queue */
void thpool_jobqueue_add(thpool_t* tp_p, thpool_job_t* newjob_p){ /* remember that job prev and next point to NULL */

	newjob_p->next=NULL;
	newjob_p->prev=NULL;

	if (newjob_p->nThreads > tp_p->threadsN) {
		WARNING_LOG("Cannot add job because it requests too many threads");
		return;
	}

	thpool_job_t *oldFirstJob;
	oldFirstJob = tp_p->jobqueue->head;

	/* fix jobs' pointers */
	switch(tp_p->jobqueue->jobsN){

		case 0:     /* if there are no jobs in queue */
			tp_p->jobqueue->tail=newjob_p;
			tp_p->jobqueue->head=newjob_p;
			break;

		default: 	/* if there are already jobs in queue */
			oldFirstJob->prev=newjob_p;
			newjob_p->next=oldFirstJob;
			tp_p->jobqueue->head=newjob_p;

	}

	(tp_p->jobqueue->jobsN)++;     /* increment amount of jobs in queue */
	sem_post(tp_p->jobqueue->queueSem);

	int sval;
	sem_getvalue(tp_p->jobqueue->queueSem, &sval);
	DEBUG_LOG(0,"Job %p added, now %d jobs in queue",newjob_p,tp_p->jobqueue->jobsN);
}


/* Remove job from queue */
int thpool_jobqueue_removelast(thpool_t* tp_p){
	thpool_job_t *oldLastJob;
	oldLastJob = tp_p->jobqueue->tail;

	/* fix jobs' pointers */
	switch(tp_p->jobqueue->jobsN){

		case 0:     /* if there are no jobs in queue */
			return -1;
			break;

		case 1:     /* if there is only one job in queue */
			tp_p->jobqueue->tail=NULL;
			tp_p->jobqueue->head=NULL;
			break;

		default: 	/* if there are more than one jobs in queue */
			oldLastJob->prev->next=NULL;               /* the almost last item */
			tp_p->jobqueue->tail=oldLastJob->prev;

	}

	(tp_p->jobqueue->jobsN)--;

	int sval;
	sem_getvalue(tp_p->jobqueue->queueSem, &sval);
	return 0;
}


/* Get first element from queue */
thpool_job_t* thpool_jobqueue_peek(thpool_t* tp_p){
	switch(tp_p->jobqueue->jobsN){

		case 0:     /* if there are no jobs in queue */
			DEBUG_LOG(0,"There are no jobs in the queue, returning NULL");
			return NULL;
			break;

		case 1:     /* if there is only one job in queue */
			{
				thpool_job_t *this = tp_p->jobqueue->tail;
				tp_p->jobqueue->tail=NULL;
				tp_p->jobqueue->head=NULL;
				DEBUG_LOG(0,"There is only one job in the queue");
				return this;
				break;
			}

		default: 	/* if there are more than one jobs in queue */
			{
				thpool_job_t *this = tp_p->jobqueue->tail;
				thpool_job_t *next = this->next;
				int idx = 0;


				while (this!=NULL) {
					if (this->nThreads <= idleThreads) { // remove this
						if (this->next == NULL) { // this was tail
							tp_p->jobqueue->tail = this->prev;
						} else {
							next->prev = this->prev;
						}
						if (this->prev == NULL) { // this was head
							tp_p->jobqueue->head = next;
						} else {
							this->prev->next = next;
						}
						DEBUG_LOG(0,"Returned the %d-th job (counted from end), jobs left: %d",idx,tp_p->jobqueue->jobsN-1);
						return this;
					}
					next = this;
					this = this->prev;
					idx++;
				}
				this = tp_p->jobqueue->tail;

				this->prev->next = NULL;
				tp_p->jobqueue->tail = this->prev;

				DEBUG_LOG(0,"There is no job which fits the number of available threads, returning the first one added");
				return this; // there is no job which fits our return first job
			}



	}


	//return tp_p->jobqueue->tail;
}
static thpool_job_t* thpool_jobqueue_peek_dumb(thpool_t* tp_p){
	return tp_p->jobqueue->tail;
}

/* Remove and deallocate all jobs in queue */
void thpool_jobqueue_empty(thpool_t* tp_p){

	thpool_job_t* curjob;
	curjob=tp_p->jobqueue->tail;

	while(tp_p->jobqueue->jobsN){
		tp_p->jobqueue->tail=curjob->prev;
		free(curjob);
		curjob=tp_p->jobqueue->tail;
		tp_p->jobqueue->jobsN--;
	}

	/* Fix head and tail */
	tp_p->jobqueue->tail=NULL;
	tp_p->jobqueue->head=NULL;
}
