#ifndef __GHOST_TASKQ_H__
#define __GHOST_TASKQ_H__

#define GHOST_TASK_LD_UNDEFINED -1
#define GHOST_TASK_LD_ANY -2
#define GHOST_TASK_LD_ALL -3

#define GHOST_TASK_DEFAULT 0
#define GHOST_TASK_PRIO_NORMAL 1
#define GHOST_TASK_PRIO_HIGH 2
#define GHOST_TASK_LD_STRICT 4

#define GHOST_TASK_INVALID 0
#define GHOST_TASK_ENQUEUED 1
#define GHOST_TASK_RUNNING 2
#define GHOST_TASK_FINISHED 3

#define GHOST_MAX_THREADS 8192

#include <pthread.h>
#include <semaphore.h>

typedef struct ghost_task_t {
	// user defined
	int nThreads, LD, flags, state;
	void *(*func)(void *);
	void *args;

	// set by the library
	int cores[128];
	void *ret;
	struct ghost_task_t *next, *prev;
	struct ghost_task_t *siblings; //there are either zero or nQueues siblings
} ghost_task_t;


typedef struct ghost_taskq_t {
	ghost_task_t *tail;
	ghost_task_t *head;
	int nTasks;
	sem_t *sem;
	pthread_mutex_t *mutex;
} ghost_taskq_t;

typedef struct ghost_thpool_t {
	pthread_t* threads;
	int nThreads;
	sem_t *sem;
} ghost_thpool_t;


int ghost_task_add(ghost_task_t *);
int ghost_task_wait(ghost_task_t *);
int ghost_task_test(ghost_task_t *);


#endif //__GHOST_TASKQ_H__
