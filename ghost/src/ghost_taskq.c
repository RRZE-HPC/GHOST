#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <errno.h>
#include <omp.h>

#include "ghost_taskq.h"
#include "ghost_util.h"
#include "ghost_cpuid.h"

#define CLR_BIT(field,bit) field[bit/(sizeof(int)*8)] &= ~(1 << (bit%(sizeof(int)*8)))
#define SET_BIT(field,bit) field[bit/(sizeof(int)*8)] |=  (1 << (bit%(sizeof(int)*8)))
#define TGL_BIT(field,bit) field[bit/(sizeof(int)*8)] ^=  (1 << (bit%(sizeof(int)*8)))
#define CHK_BIT(field,bit) field[bit/(sizeof(int)*8)]  &  (1 << (bit%(sizeof(int)*8)))

static ghost_taskq_t **taskqs;
static ghost_thpool_t *thpool;
static int nTaskqs = 0;
static int nJobs = 0;
static int killed = 0;
static int threadstate[GHOST_MAX_THREADS/sizeof(int)/8] = {0};
static int idleThreads;

static void * thread_do(void *arg);

int ghost_thpool_init(int nThreads){
	int t;

	idleThreads = nThreads;
	nTaskqs = 2;//TODO ghost_cpuid_topology.numSockets;

	if ((uint32_t)nThreads > ghost_cpuid_topology.numHWThreads) {
		WARNING_LOG("Trying to create more threads than there are hardware threads. Setting no. of threads to %u",ghost_cpuid_topology.numHWThreads);
		nThreads = ghost_cpuid_topology.numHWThreads;
	}

	if (nThreads < 1) {
		WARNING_LOG("Invalid number of threads given for thread pool (%d), setting to 1",nThreads);
		nThreads=1;
	}

	thpool = (ghost_thpool_t*)ghost_malloc(sizeof(ghost_thpool_t));
	thpool->threads = (pthread_t*)ghost_malloc(nThreads*sizeof(pthread_t));
	thpool->nThreads = nThreads;
	thpool->sem = (sem_t*)ghost_malloc(sizeof(sem_t));
	sem_init(thpool->sem, 0, nThreads);

	for (t=0; t<threadsN; t++){
		pthread_create(&(thpool->threads[t]), NULL, thread_do, NULL); 
		printf("Created thread %d in pool \n", t);
	}

	return GHOST_SUCCESS;
}

int ghost_taskq_init(int nqs){
	int q;
	DEBUG_LOG(0,"There will be %d job queues",nqs);
	nTaskqs = nqs

	taskqs=(ghost_taskq_t**)ghost_malloc(nTaskqs*sizeof(ghost_taskq_t *));

	for (q=0; q<nTaskqs; q++) {
		taskqs[q] = (ghost_taskq_t *)ghost_malloc(sizeof(ghost_taskq_t));
		taskqs[q]->tail = NULL;
		taskqs[q]->head = NULL;
		taskqs[q]->nTasks = 0;
		taskqs[q]->mutex = PTHREAD_MUTEX_INITIALIZER;
		taskqs[q]->sem = (sem_t*)malloc(sizeof(sem_t));
		sem_init(taskqs[q]->sem, 0, 0);
	}
	return GHOST_SUCCESS;
}

static void * thread_do(void *arg){


	return NULL;
}
