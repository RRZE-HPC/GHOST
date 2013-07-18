#include <stdio.h>
#include <unistd.h>
#include <omp.h>

#include "thpool.h"
#include "ghost.h"
#include "ghost_util.h"
#include "ghost_cpuid.h"

static void *task1(void *arg) {
#pragma omp parallel
#pragma omp single
	printf("task1: has %d threads\n",omp_get_num_threads());

	omp_set_num_threads(2);
#pragma omp parallel
	ghost_setCore(omp_get_thread_num());

#pragma omp parallel
	printf("task1: thread %d running @ core %d\n",omp_get_thread_num(), ghost_getCore());

	return NULL;
}


static void *task2(void *arg) {
#pragma omp parallel
#pragma omp single
	printf("task2: has %d threads\n",omp_get_num_threads());

	omp_set_num_threads(2);
#pragma omp parallel
	ghost_setCore(omp_get_thread_num()+2);

#pragma omp parallel
	printf("task2: thread %d running @ core %d\n",omp_get_thread_num(), ghost_getCore());

	return NULL;
}

int main(){

	pthread_t *t1, *t2;

	t1 = (pthread_t *)ghost_malloc(sizeof(pthread_t));
	t2 = (pthread_t *)ghost_malloc(sizeof(pthread_t));

	
	pthread_create(t1,NULL,task1,NULL);
	pthread_create(t2,NULL,task2,NULL);

	pthread_join(*t1,NULL);
	pthread_join(*t2,NULL);
	
	return 0;
}
