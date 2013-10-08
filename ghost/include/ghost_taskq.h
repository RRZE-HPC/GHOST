#ifndef __GHOST_TASKQ_H__
#define __GHOST_TASKQ_H__

#include <pthread.h>
#include <semaphore.h>
#include <hwloc.h>

#define GHOST_TASK_LD_UNDEFINED -1 // initializer
#define GHOST_TASK_LD_ANY 0 // execute task on any LD

#define GHOST_TASK_DEFAULT 0
#define GHOST_TASK_PRIO_HIGH 1 // task will be added to the head of the queue
#define GHOST_TASK_LD_STRICT 2 // task _must_ be executed on the defined LD
#define GHOST_TASK_USE_PARENTS 4 // task can use the parent's resources if added from within a task 
#define GHOST_TASK_NO_PIN 8 
#define GHOST_TASK_ONLY_HYPERTHREADS 16
#define GHOST_TASK_NO_HYPERTHREADS 32

#define GHOST_TASK_INVALID 0 // task has not been enqueued
#define GHOST_TASK_ENQUEUED 1 // task has been enqueued
#define GHOST_TASK_RUNNING 2 // task is currently running
#define GHOST_TASK_FINISHED 3 // task has finished

#define GHOST_TASK_FILL_LD -1 // use all threads of the given LD
#define GHOST_TASK_FILL_ALL -2 // use all threads of all LDs

#define GHOST_THPOOL_NTHREADS_FULLNODE ((int*)(0xBEEF))
#define GHOST_THPOOL_FTHREAD_DEFAULT ((int *)(0xBEEF))
#define GHOST_THPOOL_LEVELS_FULLSMT 0


#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief This structure represents a GHOST task.
 *
 * This data structure holds all necessary information for
 * a task. The members #nThreads, #LD, #flags, #func and #arg have to be set by
 * the user in ghost_task_init(). All other members are set by the library at
 * some point.
 */
typedef struct ghost_task_t {
	/**
	 * @brief The number of threads the task should use. (user-defined)
	 */
	int nThreads;
	/**
	 * @brief The index of the queue in which the task should be present and
	 * (preferrably) running. (user-defined)
	 */
	int LD;
	/**
	 * @brief Optional flags for the task. (user-defined)
	 */
	int flags;
	/**
	 * @brief The function to be executed by the task. (user-defined)
	 */
	void *(*func)(void *);
	/**
	 * @brief The arguments to the task's function. (user-defined)
	 */
	void *arg;

	// set by the library
	/**
	 * @brief The current state of the task. (set by the library)
	 */
	int *state;
	/**
	 * @brief The list of cores where the task's threads are running. (set by the library)
	 */
	int *cores;
	hwloc_bitmap_t coremap; // map of cores this task is using
	hwloc_bitmap_t childusedmap; // map of cores a child of this task is using
	/**
	 * @brief The return value of the task's funtion. (set by the library)
	 */
	void *ret;
	/**
	 * @brief Pointer to the next task in the queue. (set by the library)
	 */
	struct ghost_task_t *next; 
	/**
	 * @brief Pointer to the previous task in the queue. (set by the library)
	 */
	struct ghost_task_t *prev;
	/**
	 * @brief The adding task if the task has been added from within a task.
	 * (set by the library)
	 */
	struct ghost_task_t *parent;
	/**
	 * @brief Indicator that the task is finished. (set by the library)
	 */
	pthread_cond_t *finishedCond;
	/**
	 * @brief Protect accesses to the task's members. (set by the library)
	 */
	pthread_mutex_t *mutex;
	/**
	 * @brief Set to one as soon as the task's resources have been free'd.
	 * This can be the case when the task waits for a child-task to finish or
	 * when the task itself is finished.
	 */
	int freed;
} ghost_task_t;


/**
 * @brief This struct represents the task queue.
 */
typedef struct ghost_taskq_t {
	/**
	 * @brief The first (= highest priority) task in the queue
	 */
	ghost_task_t *head;
	/**
	 * @brief The last (= lowest priority) task in the queue
	 */
	ghost_task_t *tail;
	/**
	 * @brief Serialize access to the queue
	 */
	pthread_mutex_t mutex;
} ghost_taskq_t;

/**
 * @brief The thread pool consisting of all threads that will ever do some
 * tasking-related work.
 */
typedef struct ghost_thpool_t {
	/**
	 * @brief The pthread of each GHOST thread.
	 */
	pthread_t *threads;
	/**
	 * @brief The PU (Processing Unit) of each GHOST thread.
	 */
	hwloc_obj_t *PUs;
	hwloc_bitmap_t cpuset;
	/**
	 * @brief A bitmap with one bit per PU where 1 means that a PU is busy and 0 means that it is
	 * idle.
	 */
   	hwloc_bitmap_t busy;
	/**
	 * @brief The total number of threads in the thread pool
	 */
	int nThreads;
	/**
	 * @brief The number of LDs covered by the pool's threads.
	 */
	int nLDs;
	/**
	 * @brief Counts the number of readily  
	 */
	sem_t *sem; // counts the number of initialized threads
} ghost_thpool_t;

#define ghost_tasking_init(_nt,_ft,_l) \
	omp_set_nested(1); \
	ghost_thpool_init(_nt,_ft,_l); \
	_Pragma("omp parallel num_threads(2)") \
	{ \
		if (omp_get_thread_num() == 0) { \
			ghost_taskq_init(); \
			int __th; \
			for (__th=0; __th<ghost_thpool->nThreads; __th++) \
			{ \
				sem_wait(ghost_thpool->sem); \
			} \

#define ghost_tasking_finish(_nt,_ft,_l) \
			ghost_taskq_finish();\
		} else { \
			_Pragma("omp parallel num_threads(ghost_thpool->nThreads)") \
			{ \
				thread_main(NULL); \
			} \
		} \
	} \
	ghost_thpool_finish(); \

int ghost_thpool_init(int *nThreads, int *firstThread, int levels);
int ghost_taskq_init();
int ghost_taskq_finish();
int ghost_thpool_finish();

ghost_task_t * ghost_task_init(int nThreads, int LD, void *(*func)(void *), void *arg, int flags);
int ghost_task_add(ghost_task_t *);
int ghost_task_wait(ghost_task_t *);
int ghost_task_waitall();
int ghost_task_waitsome(ghost_task_t **, int, int*);
int ghost_task_test(ghost_task_t *);
int ghost_task_destroy(ghost_task_t *); // care for free'ing siblings
int ghost_task_print(ghost_task_t *t);
int ghost_taskq_print_all(); 
void * thread_main(void *arg);

char *ghost_task_strstate(int state);

extern ghost_thpool_t *ghost_thpool; // the thread pool
extern pthread_key_t ghost_thread_key; 

#ifdef __cplusplus
}// extern "C"
#endif

#endif //__GHOST_TASKQ_H__
