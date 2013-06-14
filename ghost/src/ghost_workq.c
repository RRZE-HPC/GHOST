#define _POSIX_C_SOURCE 199309L
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <errno.h>
#include <ghost.h>
#include <ghost_util.h>
#include <ghost_workq.h>

#define WAIT_BEFORE_FINISH 1

static void *ghost_workq_server(void *arg); 

int ghost_workq_init(ghost_workq_t *wq, int threads, void (*engine)(void *))
{
	int status;

	status = pthread_attr_init(&wq->attr);
	if (status != 0)
		return status;

	status = pthread_attr_setdetachstate(&wq->attr,PTHREAD_CREATE_DETACHED);
	if (status != 0) {
		pthread_attr_destroy(&wq->attr);
		return status;
	}

	status = pthread_mutex_init(&wq->mutex, NULL);
	if (status != 0) {
		pthread_attr_destroy(&wq->attr);
		return status;
	}

	status = pthread_cond_init(&wq->cv, NULL);
	if (status != 0) {
		pthread_mutex_destroy(&wq->mutex);
		pthread_attr_destroy(&wq->attr);
		return status;
	}

	wq->quit = 0;
	wq->first = wq->last = NULL;
	wq->parallelism = threads;
	wq->counter = 0;
	wq->idle = 0;
	wq->engine = engine;
	wq->valid = GHOST_WORKQ_VALID;

	return 0;
}

int ghost_workq_destroy(ghost_workq_t *wq)
{
	int status, status1, status2;

	if (wq->valid != GHOST_WORKQ_VALID)
		return -1;

	status = pthread_mutex_lock(&wq->mutex);
	if (status != 0)
		return status;
	wq->valid = 0;	

	if (wq->counter > 0) {
		wq->quit = 1;
		if (wq->idle > 0) {
			status = pthread_cond_broadcast(&wq->cv);
			if (status != 0) {
				pthread_mutex_unlock(&wq->mutex);
				return status;
			}
		}

		while (wq->counter > 0) {
			status = pthread_cond_wait(&wq->cv, &wq->mutex);
			if (status != 0) {
				pthread_mutex_unlock(&wq->mutex);
				return status;
			}
		}
	}

	status = pthread_mutex_unlock(&wq->mutex);
	if (status != 0)
		return status;

	status = pthread_mutex_destroy(&wq->mutex);
	status1 = pthread_cond_destroy(&wq->cv);
	status2 = pthread_attr_destroy(&wq->attr);

	return (status?status:(status1?status1:status2));
}

int ghost_workq_add(ghost_workq_t *wq, void * element)
{
	ghost_workq_ele_t *item;
	pthread_t id;
	int status;

	if (wq->valid != GHOST_WORKQ_VALID)
		return -1;

	item = (ghost_workq_ele_t *)ghost_malloc(sizeof(ghost_workq_ele_t));

	item->data = element;
	item->next = NULL;
	status = pthread_mutex_lock(&wq->mutex);
	if (status != 0) {
		free(item);
		return status;
	}

	if (wq->first == NULL)
		wq->first = item;
	else
		wq->last->next = item;
	wq->last = item;

	if (wq->idle > 0) {
		status = pthread_cond_signal(&wq->cv);
		if (status != 0) {
			pthread_mutex_unlock(&wq->mutex);
			return status;
		} 
	} else if (wq->counter < wq->parallelism) {
		DEBUG_LOG(0,"Creating new worker");
		status = pthread_create(&id, &wq->attr, ghost_workq_server, (void *)wq);
		if (status != 0) {
			pthread_mutex_unlock(&wq->mutex);
			return status;
		}
		wq->counter++;
	}
	pthread_mutex_unlock(&wq->mutex);
	return 0;
}


static void *ghost_workq_server(void *arg) 
{
	struct timespec timeout;
	ghost_workq_t *wq = (ghost_workq_t *)arg;
	ghost_workq_ele_t *we;
	int status, timedout;

	DEBUG_LOG(0,"A worker is starting");
	status = pthread_mutex_lock(&wq->mutex);
	if (status != 0)
		return NULL;

	while(1) {
		timedout = 0;
		DEBUG_LOG(0,"Worker waiting for work");
		clock_gettime(CLOCK_REALTIME,&timeout);
		timeout.tv_sec += WAIT_BEFORE_FINISH;

		while (wq->first == NULL && !wq->quit) {
			status = pthread_cond_timedwait(&wq->cv, &wq->mutex, &timeout);
			if (status == ETIMEDOUT) {
				DEBUG_LOG(0,"Worker wait timed out");
				timedout = 1;
				break;
			} else if (status != 0) {
				DEBUG_LOG(0,"Worker wait failed: %d (%s)",status,strerror(status));
				wq->counter--;
				pthread_mutex_unlock(&wq->mutex);
				return NULL;
			}
		}

		DEBUG_LOG(0,"Work queue: 0x%p, quit: %d",wq->first,wq->quit);
		we = wq->first;

		if (we != NULL) {
			wq->first = we->next;
			if (wq->last == we)
				wq->last = NULL;
			status = pthread_mutex_unlock(&wq->mutex);
			if (status != 0)
				return NULL;
			DEBUG_LOG(0,"Worker calling engine");
			wq->engine(we->data);
			free(we);
			status = pthread_mutex_lock(&wq->mutex);
			if (status != 0)
				return NULL;
		}

		if (wq->first == NULL && wq->quit) {
			DEBUG_LOG(0,"Worker shutting down");
			wq->counter--;

			if (wq->counter == 0)
				pthread_cond_broadcast(&wq->cv);
			pthread_mutex_unlock(&wq->mutex);
			return NULL;
		}
		if (wq->first == NULL && timedout) {
			DEBUG_LOG(0,"Engine terminating due to timeout.");
			wq->counter--;
			break;
		}
	}

	pthread_mutex_unlock(&wq->mutex);
	DEBUG_LOG(0,"Worker exiting");
	return NULL;
}
	




