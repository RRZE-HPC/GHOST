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

static void *controller1(void *arg)
{
	UNUSED(arg);
	

	ghost_task_t *lrTask;
	ghost_task_t *srTask;

	lrTask = ghost_task_init(GHOST_TASK_FILL_ALL, 0, &longRunningFunc, NULL, GHOST_TASK_DEFAULT);
	srTask = ghost_task_init(GHOST_TASK_FILL_ALL, 0, &shortRunningFunc, NULL, GHOST_TASK_DEFAULT);
	ghost_task_add(lrTask);
	ghost_task_add(srTask);

	return NULL;
}

static void *controller2(void *arg)
{
	UNUSED(arg);
	
	ghost_task_t *lrTask;
	ghost_task_t *srTask;

	lrTask = ghost_task_init(GHOST_TASK_FILL_ALL, 0, &longRunningFunc, NULL, GHOST_TASK_DEFAULT);
	srTask = ghost_task_init(GHOST_TASK_FILL_ALL, 0, &shortRunningFunc, NULL, GHOST_TASK_DEFAULT);
	ghost_task_add(lrTask);
	ghost_task_add(srTask);

	return NULL;
}

int main(int argc, char ** argv)
{
	ghost_init(argc,argv);
	pthread_t ct1, ct2;
	pthread_create(&ct1,NULL,controller1,NULL);
	pthread_create(&ct2,NULL,controller2,NULL);

	printf(">>> started controllers\n");	
	pthread_join(ct1,NULL);
	pthread_join(ct2,NULL);
//	ghost_task_waitall();

	ghost_finish();
	return 0;
}
