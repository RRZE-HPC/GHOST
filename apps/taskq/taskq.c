#define _XOPEN_SOURCE 500
#include <stdio.h>
#include <unistd.h>
#include <omp.h>

#include "ghost.h"
#include "ghost_util.h"
#include "ghost_taskq.h"

static void *accuFunc(void *arg) 
{
	int *ret = (int *)ghost_malloc(sizeof(int));
	*ret = (*(int *)arg)+1;
	
#pragma omp parallel
	{
#pragma omp single
	printf("    ######### accuFunc: numThreads: %d\n",omp_get_num_threads());
	printf("    ######### accuFunc: thread %d running @ core %d\n",omp_get_thread_num(), ghost_getCore());
	}

	return ret;
}

static void *shortRunningFunc(void *arg) 
{

	usleep(5e5); // sleep 500 ms
	if (arg != NULL)
		printf("    ######### shortRunningFunc arg: %s\n",(char *)arg);

#pragma omp parallel
	{
#pragma omp single
	printf("    ######### shortRunningFunc: numThreads: %d\n",omp_get_num_threads());
	printf("    ######### shortRunningFunc: thread %d running @ core %d\n",omp_get_thread_num(), ghost_getCore());
	}

	return NULL;
}

static void *longRunningFunc(void *arg) 
{
	usleep(1e6); // sleep 1 sec
	if (arg != NULL)
		printf("    ######### longRunningFunc arg: %s\n",(char *)arg);

#pragma omp parallel
	{
#pragma omp single
	printf("    ######### longRunningFunc: numThreads: %d\n",omp_get_num_threads());
	printf("    ######### longRunningFunc: thread %d running @ core %d\n",omp_get_thread_num(), ghost_getCore());
	}

	return NULL;
}

int main(int argc, char ** argv)
{
	int foo = 42;

	ghost_init(argc,argv);
	ghost_thpool_init(ghost_getNumberOfPhysicalCores());

	printf("The thread pool consists of %d threads in %d locality domains\n",ghost_thpool->nThreads,ghost_thpool->nLDs);

	ghost_task_t *accuTask;
	ghost_task_t *lrTask;
	ghost_task_t *srTask;

	accuTask = ghost_task_init(1, 0, &accuFunc, &foo, GHOST_TASK_DEFAULT);

	printf("checking for correct task state and whether accuTask returns %d\n",foo+1);
	printf("state should be invalid: %s\n",ghost_task_strstate(ghost_task_test(accuTask)));

	ghost_task_add(accuTask);
	printf("state should be enqueued or running: %s\n",ghost_task_strstate(ghost_task_test(accuTask)));
	ghost_task_wait(accuTask);
	printf("state should be finished: %s\n",ghost_task_strstate(ghost_task_test(accuTask)));
	printf("accuTask returned %d\n",*(int *)accuTask->ret);
	ghost_task_destroy(accuTask);

	printf("\nstarting long running task @ LD0 and check if another strict LD0 task is blocking\n");
	lrTask = ghost_task_init(GHOST_TASK_FILL_LD, 0, &longRunningFunc, NULL, GHOST_TASK_DEFAULT);
	srTask = ghost_task_init(GHOST_TASK_FILL_LD, 0, &shortRunningFunc, NULL, GHOST_TASK_LD_STRICT);
	ghost_task_add(lrTask);
	ghost_task_add(srTask);

	ghost_task_waitall();
	ghost_task_destroy(lrTask);
	ghost_task_destroy(srTask);

	printf("\nstarting long running task @ LD0 and check if another non-strict LD0 task is running @ LD1\n");
	lrTask = ghost_task_init(GHOST_TASK_FILL_LD, 0, &longRunningFunc, NULL, GHOST_TASK_DEFAULT);
	srTask = ghost_task_init(GHOST_TASK_FILL_LD, 0, &shortRunningFunc, NULL, GHOST_TASK_DEFAULT);
	ghost_task_add(lrTask);
	ghost_task_add(srTask);
	
	ghost_task_waitall();
	ghost_task_destroy(lrTask);
	ghost_task_destroy(srTask);

	printf("\nwaiting for any out of ten tasks where 9 are long- and 1 is short-running\n");
	srand((int)time(NULL));
	int nTasks = 4, t, shortIdx = (int)(((double)rand()/RAND_MAX)*(double)(nTasks-1));
	//ghost_task_t *tasks[nTasks];
	ghost_task_t **tasks = (ghost_task_t **)ghost_malloc(nTasks*sizeof(ghost_task_t *));

	for (t=0; t<nTasks; t++) {
		tasks[t] = ghost_task_init(1, t%ghost_thpool->nLDs, t==shortIdx?&shortRunningFunc:&longRunningFunc, NULL, GHOST_TASK_DEFAULT);
		ghost_task_add(tasks[t]);
	}

	int status[nTasks];
	
	usleep(1e3);
	ghost_task_waitsome(tasks,nTasks,status);
	printf("one should be finished: ");
	for (t=0; t<nTasks; t++) {
		printf("%d",status[t]);
	}
	printf("\n");
	
	usleep(1e6);
	ghost_task_waitsome(tasks,nTasks,status);
	printf("all should be finished: ");
	for (t=0; t<nTasks; t++) {
		printf("%d",status[t]);
	}
	printf("\n");

	ghost_task_waitall();

	printf("\nchecking if wait() works even when the task is not yet running\n");
	lrTask = ghost_task_init(GHOST_TASK_FILL_ALL, GHOST_TASK_LD_ANY, &longRunningFunc, NULL, GHOST_TASK_DEFAULT);
	srTask = ghost_task_init(1, 0, &shortRunningFunc, NULL, GHOST_TASK_DEFAULT);

	ghost_task_add(lrTask);
	ghost_task_add(srTask);

	ghost_task_wait(srTask);
	printf("short running task finished!\n");

	ghost_task_waitall();
	ghost_task_destroy(lrTask);
//	ghost_task_destroy(srTask);

	ghost_finish();
	return 0;
}
