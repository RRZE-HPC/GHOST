#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <pthread.h>
#include <time.h>

#include "ghost.h"
#include "ghost_taskq.h"

//#define SYNC_WAIT
#define DONT_PIN_CONTROL_THREADS

//                                                              
// this example driver performs the following algorithm.        
// given a number of columns of integer vectors, fill each      
// vector with the sequence                                     
//                                                              
// V[n+1]=random() if n=-1 or V[n]=0 or V[n]=1                  
// V[n+1] = V[n]+1 if V[n] is odd                               
// V[n+1] = V[n]/2 if n is even                                 
//                                                              
// Each vector is treated by a different 'control thread', the  
// actual operations (rndX(), incX and divX) are put into a     
// task buffer and executed in bulks whenever each control      
// thread has announced the type of operation it needs.         
//                                                              

////////////////////////////////////////////////////////////////////////////////////////
// some constants for the example                                                     //
////////////////////////////////////////////////////////////////////////////////////////

// a general implementation of this task buffer model in phist or ghost is achieved when
// all these macros are only needed in the main program and model implementation, but not
// in the taskBuffer implementation.

// no operation
#define JOBTYPE_NOOP -1
// x=random()
#define JOBTYPE_RNDX 0
// x=x+1
#define JOBTYPE_INCX 1
// x=x/2
#define JOBTYPE_DIVX 2

// random numbers are picked between 0 and RND_MAX
#define RND_MAX 10000
// number of columns to be filled (=number of control threads)
#define NUM_TASKS 4
// we perform three different operations, rndX, incX and divX
#define NUM_JOBTYPES 3 
// local length of each column
#define NDIM 40

#define OUT stderr

////////////////////////////////////////////////////////////////////////////////////////
// algorithm-specific functionality                                                   //
////////////////////////////////////////////////////////////////////////////////////////

// argument for rndX, incX and divX functions operating on an array of int pointers.
typedef struct
{
	int nThreads;
	int n;
	int **val;
} argList_t;

void argList_create(argList_t** args, int ntasks)
{
	*args = (argList_t*)malloc(sizeof(argList_t));
	(*args)->nThreads=1;
	(*args)->n=0;
	(*args)->val=(int**)malloc(ntasks*sizeof(int*));
}

void argList_delete(argList_t* args)
{
	free(args->val);
	free(args);
}

// fill  all entries in an array with random integers
static void *rndX(void* v_arg)
{
	int i;
	argList_t* arg = (argList_t*)v_arg;

	//  fprintf(OUT,"NUM THREADS: %d\n",arg->nThreads);
	omp_set_num_threads(arg->nThreads);

#pragma omp parallel
	{
#pragma omp single
		fprintf(OUT,"rndX executed by %d threads on %d values\n",omp_get_num_threads(),arg->n);  
#pragma omp for
		for (i=0;i<arg->n;i++)
			*(arg->val[i])=(int)((rand()/(double)RAND_MAX)*RND_MAX);
	}
	return NULL;
}

// increment all entries in an array by one
static void *incX(void* v_arg)
{
	int i;
	argList_t* arg = (argList_t*)v_arg;

	//  fprintf(OUT,"NUM THREADS: %d\n",arg->nThreads);
	omp_set_num_threads(arg->nThreads);

#pragma omp parallel
	{
#pragma omp single
		fprintf(OUT,"incX executed by %d threads on %d values\n",
				omp_get_num_threads(),arg->n);  
#pragma omp for
		for (i=0;i<arg->n;i++)
			*(arg->val[i])+=1;
	}
	return NULL;
}


// divide all entries in an array by two
static void *divX(void* v_arg)
{
	int i;
	argList_t* arg = (argList_t*)v_arg;

	//  fprintf(OUT,"NUM THREADS: %d\n",arg->nThreads);
	omp_set_num_threads(arg->nThreads);

#pragma omp parallel
	{
#pragma omp single
		fprintf(OUT,"divX executed by %d threads on %d values\n", 
				omp_get_num_threads(),arg->n);  
#pragma omp for
		for (i=0;i<arg->n;i++)
			*(arg->val[i])/=2;
	}
	return NULL;
}

////////////////////////////////////////////////////////////////////////////////////////
// task buffer prototype                                                              //
////////////////////////////////////////////////////////////////////////////////////////

// a buffer object, when each control task has set its jobType flag,
// the rndX, incX and divX jobs are bundled and enqueued in the ghost queue.
typedef struct {

	int njobs; // total number of jobs (operating on a single int value)
	int njobTypes; // number of different tasks (e.g. incX, divX, rndX) that the buffer can handle
	int countdown; // when this reaches 0, the jobs are launched
	pthread_mutex_t lock_mx; // for controlling access to the shared object and the condition 
	// variable finished_cv.
	pthread_cond_t finished_cv; // for waiting for the buffer to be emptied

	int *jobType; // dimension njobs
	int **jobArg; // dimension njobs

	ghost_task_t **ghost_task; // dimension njobTypes
	argList_t **ghost_args;    // dimension njobTypes

} taskBuf_t;

//! create a new task buffer for handling certain basic operations in a blocked way
void taskBuf_create(taskBuf_t** buf, int num_tasks, int num_workers, 
		int num_jobTypes, int* ierr);
//! delete the task buffer
void taskBuf_delete(taskBuf_t* buf, int* ierr);
//! put a job request into the task buffer and wait until it is enqueued
void taskBuf_add(taskBuf_t* buf, int* arg, int task_id, int taskFlag);
//! flush the task buffer, e.g. group the tasks together and put them in
//! the ghost queue, then signal any waiting jobs.
void taskBuf_flush(taskBuf_t* buf);
//! wait for the task that you put in the buffer to be finished
void taskBuf_wait(taskBuf_t* buf, int task_id);

// create a new task buffer object
void taskBuf_create(taskBuf_t** buf, int num_tasks, 
		int num_workers, int num_jobTypes, int* ierr)
{
	int i;
	*ierr=0;
	*buf = (taskBuf_t*)malloc(sizeof(taskBuf_t));

	(*buf)->njobs=num_tasks;
	(*buf)->njobTypes=num_jobTypes;
	(*buf)->countdown=num_tasks;
	(*buf)->jobType=(int*)malloc(num_tasks*sizeof(int));
	(*buf)->jobArg=(int**)malloc(num_tasks*sizeof(int*));

	pthread_mutex_init(&((*buf)->lock_mx),NULL);
	pthread_cond_init(&(*buf)->finished_cv,NULL);

	(*buf)->ghost_args=(argList_t**)malloc(num_jobTypes*sizeof(argList_t*));
	(*buf)->ghost_task=(ghost_task_t**)malloc(num_jobTypes*sizeof(ghost_task_t*));

	for (i=0;i<num_jobTypes;i++)
	{
		argList_create(&(*buf)->ghost_args[i], num_tasks);
		(*buf)->ghost_args[i]->nThreads = num_workers;
		(*buf)->ghost_args[i]->nThreads = num_workers;
	}
#ifdef DONT_PIN_CONTROL_THREADS
	(*buf)->ghost_task[JOBTYPE_RNDX] = ghost_task_init(GHOST_TASK_FILL_ALL, GHOST_TASK_LD_ANY, &rndX, 
			(void*)((*buf)->ghost_args[JOBTYPE_RNDX]),GHOST_TASK_DEFAULT);

	(*buf)->ghost_task[JOBTYPE_INCX] = ghost_task_init(GHOST_TASK_FILL_ALL, GHOST_TASK_LD_ANY, &incX, 
			(void*)((*buf)->ghost_args[JOBTYPE_INCX]),GHOST_TASK_DEFAULT);

	(*buf)->ghost_task[JOBTYPE_DIVX] = ghost_task_init(GHOST_TASK_FILL_ALL, GHOST_TASK_LD_ANY, &divX, 
			(void*)((*buf)->ghost_args[JOBTYPE_DIVX]),GHOST_TASK_DEFAULT);
#else
	(*buf)->ghost_task[JOBTYPE_RNDX] = ghost_task_init(num_workers, 0, &rndX, 
			(void*)((*buf)->ghost_args[JOBTYPE_RNDX]),GHOST_TASK_DEFAULT);

	(*buf)->ghost_task[JOBTYPE_INCX] = ghost_task_init(num_workers, 0, &incX, 
			(void*)((*buf)->ghost_args[JOBTYPE_INCX]),GHOST_TASK_DEFAULT);

	(*buf)->ghost_task[JOBTYPE_DIVX] = ghost_task_init(num_workers, 0, &divX, 
			(void*)((*buf)->ghost_args[JOBTYPE_DIVX]),GHOST_TASK_DEFAULT);
#endif
	return;
}

// create a new task buffer object
void taskBuf_delete(taskBuf_t* buf, int* ierr)
{
	int i;
	*ierr=0;
	//TODO - maybe we should force execution/wait here
	free(buf->jobType);
	free(buf->jobArg);
	for (i=0;i<buf->njobTypes;i++)
	{
		ghost_task_destroy(buf->ghost_task[i]);
		argList_delete(buf->ghost_args[i]);
	}
	pthread_mutex_destroy(&buf->lock_mx);
	pthread_cond_destroy(&buf->finished_cv);
	free(buf->ghost_task);
	free(buf->ghost_args);
	free(buf);
	return;
}

// a job is put into the task buffer and executed once the
// buffer is full. This function returns only when the task 
// is finished.
void taskBuf_add(taskBuf_t* buf, int* arg, int task_id, int taskFlag)
{
	// lock the buffer while putting in jobs and executing them
	pthread_mutex_lock(&buf->lock_mx);
	buf->jobType[task_id]=taskFlag;
	buf->jobArg[task_id]=arg;
	buf->countdown--;
	fprintf(OUT,"control thread %lu request job %d on column %d, countdown=%d\n",
			pthread_self(), taskFlag, task_id, buf->countdown);
	if (buf->countdown==0)
	{
		taskBuf_flush(buf);
		fprintf(OUT,"Thread %lu sending signal @ cond %p \n",pthread_self(),buf->finished_cv);
		pthread_cond_broadcast(&buf->finished_cv);
	}
	else
	{
		fprintf(OUT,"Thread %lu wait @ cond %p \n",pthread_self(),buf->finished_cv);
		// this sends the thread to sleep and releases the mutex. When the signal is
		// received, the mutex is locked again
		pthread_cond_wait(&buf->finished_cv,&buf->lock_mx);
	}
	// the bundled tasks have been put in the ghost queue.
	// Release the mutex so the buffer can be used for the next iteration.
	// The control thread is responsible for waiting for the task to finish
	// before putting in a new one, otherwise a race condition is created.
	pthread_mutex_unlock(&buf->lock_mx);
}

// once the buffer is full, group the tasks into blocks and enqueue them.
// This function must not be called by multiple threads at a time, which 
// is why we encapsulate it in an openMP lock in taskBuf_add above.
void taskBuf_flush(taskBuf_t* buf)
{
	int i,pos1,pos2;

	fprintf(OUT,"control thread %lu starting jobs\n",pthread_self());
	fprintf(OUT,"job types: ");
	for (i=0;i<buf->njobs;i++)
	{
		fprintf(OUT," %d",buf->jobType[i]);
	}
	fprintf(OUT,"\n");

	for (i=0;i<buf->njobTypes;i++)
	{
		buf->ghost_args[i]->n=0;
	}
	for (i=0;i<buf->njobs;i++)
	{
		pos1=buf->jobType[i]; // which op is requested? (rndX, incX or divX)
		pos2=buf->ghost_args[pos1]->n; // how many of this op have been requested already?
		buf->ghost_args[pos1]->val[pos2]=buf->jobArg[i];
		buf->ghost_args[pos1]->n++;
	}
	for (i=0;i<buf->njobTypes;i++)
	{
		if (buf->ghost_args[i]->n>0)
		{
			fprintf(OUT,"enqueue job type %d on %d workers for %d values\n",
					i,buf->ghost_args[i]->nThreads,buf->ghost_args[i]->n);
			ghost_task_add(buf->ghost_task[i]);
		}
	}
	buf->countdown=buf->njobs;
#ifdef SYNC_WAIT
	//TODO
	// this should be removed, waiting should be left to 
	// each of the control threads using taskBuf_wait().
	for (i=0;i<buf->njobTypes;i++)
	{
		if (buf->ghost_args[i]->n>0)
		{
			fprintf(OUT,"Thread %lu waiting for job type %d\n",pthread_self(),i);
			ghost_task_wait(buf->ghost_task[i]);
		}
	}
#endif
	return;
}

void taskBuf_wait(taskBuf_t* buf, int task_id)
{
	int jt = buf->jobType[task_id];
	if (buf->ghost_args[jt]->n>0)
	{
		fprintf(OUT,"Thread %lu waiting for job type %d\n",pthread_self(),jt);
		ghost_task_wait(buf->ghost_task[jt]);
		fprintf(OUT,"Thread %lu, job type %d finished.\n",pthread_self(),jt);
	}
	return;
}

////////////////////////////////////////////////////////////////////////////////////////////
// implementation of the algorithm                                                        //
////////////////////////////////////////////////////////////////////////////////////////////

typedef struct {
	int n;
	int col;
	int *v;
	taskBuf_t *taskBuf;
} mainArg_t;

// this is a long-running job (an "algorithm") executed by a control thread.
// Inside, work that has to be done is put in a task buffer and than bundled
// between control threads and put into the ghost queue.
void* fill_vector(void* v_mainArg)
{
	int i,n,col;
	int* v;
	taskBuf_t* taskBuf;
	mainArg_t *mainArg = (mainArg_t*)v_mainArg;

	n = mainArg->n;
	col = mainArg->col;
	v = mainArg->v;
	taskBuf=mainArg->taskBuf;

	// these are all blocking function calls right now
	v[0]=0;// next one will be randomized
	for (i=1; i<n;i++)
	{
		v[i]=v[i-1];
		if (v[i-1]==0 || v[i-1]==1)
		{
			taskBuf_add(taskBuf,&v[i],col,JOBTYPE_RNDX);
		}
		else if (v[i-1]%2==0)
		{
			taskBuf_add(taskBuf,&v[i],col,JOBTYPE_DIVX);
		}
		else
		{
			taskBuf_add(taskBuf,&v[i],col,JOBTYPE_INCX);
		}
		//    TODO - we currently wait in the taskBuf_flush function
#ifndef SYNC_WAIT    
		taskBuf_wait(taskBuf,col);    
#endif
	}
	return NULL;
}


int main(int argc, char** argv)
{
	int rank, num_proc;
	int ierr;
	int i,j;
	int nworkers;
	taskBuf_t *taskBuf;
	mainArg_t mainArg[NUM_TASKS];
#ifdef DONT_PIN_CONTROL_THREADS
	pthread_t controlThread[NUM_TASKS];
#else
	ghost_task_t *controlTask[NUM_TASKS];
#endif

	int* vectors = (int*)malloc(NUM_TASKS*NDIM*sizeof(int));

	// initialize ghost queue
	ghost_init(argc,argv);

	// initialize C random number generator
	srand(time(NULL));

#ifdef DONT_PIN_CONTROL_THREADS

	nworkers=ghost_thpool->nThreads;

#else
	// we would actually like to run the control threads without pinning
	// and allow having all cores for doing the work, but if we do that 
	// in the current ghost implementation, the jobs in the ghost queue 
	// never get executed because NUM_TASKS cores are reserved for the     
	// control threads.
	nworkers=ghost_thpool->nThreads - NUM_TASKS;

#endif

	taskBuf_create(&taskBuf,NUM_TASKS,nworkers,NUM_JOBTYPES,&ierr);


	for (i=0;i<NUM_TASKS;i++)
	{
		mainArg[i].n=NDIM;
		mainArg[i].col=i;
		mainArg[i].v=vectors+i*NDIM;
		mainArg[i].taskBuf=taskBuf;
#ifndef DONT_PIN_CONTROL_THREADS
		controlTask[i] = ghost_task_init(1, 0, &fill_vector, 
				(void*)(&mainArg[i]),GHOST_TASK_DEFAULT);
		ghost_task_add(controlTask[i]);
#else
		pthread_create(&controlThread[i], NULL, &fill_vector, (void *)&mainArg[i]);
#endif
	}

#ifndef DONT_PIN_CONTROL_THREADS
	ghost_task_waitall();
#else
	for (i=0;i<NUM_TASKS;i++)
	{
		pthread_join(controlThread[i],NULL);
	}
#endif
	// single thread prints the results
	for (i=0;i<NDIM;i++)
	{
		for (j=0;j<NUM_TASKS;j++)
		{
			fprintf(OUT,"\t%d",vectors[j*NDIM+i]);
		}
		fprintf(OUT,"\n");
	}

	// finalize ghost queue
	ghost_finish();
}
