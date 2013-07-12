#ifndef __GHOST_TASKQ_H__
#define __GHOST_TASKQ_H__

#include <pthread.h>
#include <semaphore.h>

#define GHOST_TASK_LD_UNDEFINED -1
#define GHOST_TASK_LD_ANY -2 // execute task on any LD
#define GHOST_TASK_LD_ALL -3 // span across all LDs

#define GHOST_TASK_DEFAULT 0
#define GHOST_TASK_PRIO_NORMAL 1
#define GHOST_TASK_PRIO_HIGH 2
#define GHOST_TASK_LD_STRICT 4 // task _must_ be executed on the defined LD

#define GHOST_TASK_INVALID 0
#define GHOST_TASK_ENQUEUED 1
#define GHOST_TASK_RUNNING 2
#define GHOST_TASK_FINISHED 3

#define GHOST_MAX_THREADS 8192

#define GHOST_TASK_INIT(...) { .nThreads = 0, .LD = GHOST_TASK_LD_UNDEFINED, .flags = GHOST_TASK_DEFAULT, .func = NULL, .arg = NULL, .state = GHOST_TASK_INVALID, ## __VA_ARGS__ }


typedef struct ghost_task_t {
	// user defined
	int nThreads, LD, flags;
	void *(*func)(void *);
	void *arg;

	// set by the library
	int state, executingThreadNo;
	int *cores;
	void *ret;
	struct ghost_task_t *next, *prev;
	struct ghost_task_t **siblings; //there are either zero or nQueues siblings
	pthread_cond_t finishedCond;
} ghost_task_t;


typedef struct ghost_taskq_t {
	ghost_task_t *tail;
	ghost_task_t *head;
	int nTasks;
	sem_t *sem;
	pthread_mutex_t mutex;
	int LD;
} ghost_taskq_t;

typedef struct ghost_thpool_t {
	pthread_t *threads;
	int *LDs;
	int *nThreadsPerLD;
	int *firstThreadOfLD;
	int nLDs;
	int nThreads;
	int idleThreads;
	sem_t *sem;
} ghost_thpool_t;


int ghost_thpool_init(int nThreads);
int ghost_taskq_init(int nqs);
int ghost_taskq_finish();

int ghost_task_add(ghost_task_t *);
int ghost_task_wait(ghost_task_t *);
int ghost_task_test(ghost_task_t *);
int ghost_task_free(ghost_task_t *); // care for free'ing siblings
int ghost_task_print(ghost_task_t *t);


#endif //__GHOST_TASKQ_H__
