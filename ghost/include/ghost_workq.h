#include <pthread.h>

typedef struct ghost_workq_ele_t {
	struct ghost_workq_ele_t *next;
	void *data;
} ghost_workq_ele_t;

typedef struct ghost_workq_t {
	pthread_mutex_t mutex;
	pthread_cond_t cv;
	pthread_attr_t attr;
	ghost_workq_ele_t *first, *last;
	int valid;
	int quit;
	int parallelism;
	int counter;
	int idle;
	void (*engine)(void *arg);
} ghost_workq_t;

#define GHOST_WORKQ_VALID 1

int ghost_workq_init(ghost_workq_t *wq, int threads, void (*engine)(void *));
int ghost_workq_destroy(ghost_workq_t *wq);
int ghost_workq_add(ghost_workq_t *wq, void *data);
