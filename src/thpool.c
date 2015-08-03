#include "ghost/config.h"
#include "ghost/thpool.h"
#include "ghost/util.h"
#include "ghost/machine.h"

/**
 * @brief Each of the threads in the thread pool gets assigned the task it executes via pthread_setspecific. 
 This is the key to this specific data. It is exported in ghost_taskq.h
 */
static pthread_key_t ghost_thread_key = 0;

/**
 * @brief The thread pool created by ghost_thpool_create(). 
 */
static ghost_thpool_t *ghost_thpool = NULL;



ghost_error_t ghost_thpool_create(int nThreads, void *(func)(void *))
{
    int t;
    int oldthreads = 0; 

    if (!ghost_thpool) {
        GHOST_CALL_RETURN(ghost_malloc((void **)&ghost_thpool,sizeof(ghost_thpool_t)));
        ghost_thpool->nThreads = nThreads;
        GHOST_CALL_RETURN(ghost_malloc((void **)&ghost_thpool->threads,ghost_thpool->nThreads*sizeof(pthread_t)));
        GHOST_CALL_RETURN(ghost_malloc((void **)&ghost_thpool->sem,sizeof(sem_t)));

        pthread_key_create(&ghost_thread_key,NULL);

        DEBUG_LOG(1,"All threads are initialized and waiting for tasks");
    } else {
        DEBUG_LOG(1,"Resizing the thread pool");
       
        oldthreads = ghost_thpool->nThreads; 
        ghost_thpool->nThreads = nThreads;
        sem_init(ghost_thpool->sem, 0, 0);

        ghost_thpool->threads = realloc(ghost_thpool->threads,ghost_thpool->nThreads*sizeof(pthread_t));
        
    }
        
    sem_init(ghost_thpool->sem, 0, 0);
    for (t=oldthreads; t<ghost_thpool->nThreads; t++){
        pthread_create(&(ghost_thpool->threads[t]), NULL, func, (void *)(intptr_t)t);
    }
    for (t=oldthreads; t<ghost_thpool->nThreads; t++){
        sem_wait(ghost_thpool->sem);
    }
        
    return GHOST_SUCCESS;
}

ghost_error_t ghost_thpool_thread_add(void *(func)(void *), intptr_t arg)
{
    ghost_thpool->nThreads++;

    ghost_thpool->threads = realloc(ghost_thpool->threads,ghost_thpool->nThreads*sizeof(pthread_t));
    pthread_create(&(ghost_thpool->threads[ghost_thpool->nThreads-1]), NULL, func, (void *)arg);
    sem_wait(ghost_thpool->sem);
    
    return GHOST_SUCCESS;


}

ghost_error_t ghost_thpool_destroy()
{
    if (ghost_thpool == NULL)
        return GHOST_SUCCESS;

    DEBUG_LOG(1,"Join all threads");
    int t; 
    for (t=0; t<ghost_thpool->nThreads; t++)
    {   
//        pthread_cancel(ghost_thpool->threads[t]);
//        printf("cancelled %d %d\n",t,(int)ghost_thpool->threads[t]); 
        if (pthread_join(ghost_thpool->threads[t],NULL)){
            ERROR_LOG("pthread_join failed: %s",strerror(errno));
            return GHOST_ERR_UNKNOWN;
        }
    }

    free(ghost_thpool->threads); ghost_thpool->threads = NULL;
    free(ghost_thpool->sem); ghost_thpool->sem = NULL;
//    free(ghost_thpool->PUs); ghost_thpool->PUs = NULL;
//    hwloc_bitmap_free(ghost_thpool->cpuset); ghost_thpool->cpuset = NULL;
//    hwloc_bitmap_free(ghost_thpool->busy); ghost_thpool->busy = NULL;
    free(ghost_thpool); ghost_thpool = NULL;
        

    return GHOST_SUCCESS;
}

ghost_error_t ghost_thpool_get(ghost_thpool_t **thpool)
{
    if (!thpool) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }
    if (!ghost_thpool) {
        ERROR_LOG("Thread pool not present");
        return GHOST_ERR_UNKNOWN;
    }

    *thpool = ghost_thpool;

    return GHOST_SUCCESS;
}

ghost_error_t ghost_thpool_key(pthread_key_t *key)
{
    if (!key) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }

    *key = ghost_thread_key;

    return GHOST_SUCCESS;
}
