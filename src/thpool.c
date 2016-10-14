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
static ghost_thpool *thpool = NULL;



ghost_error ghost_thpool_create(int nThreads, void *(func)(void *))
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_TASKING|GHOST_FUNCTYPE_SETUP); 
    
    int t;
    int oldthreads = 0; 

    if (!thpool) {
        GHOST_CALL_RETURN(ghost_malloc((void **)&thpool,sizeof(ghost_thpool)));
        thpool->nThreads = nThreads;
        GHOST_CALL_RETURN(ghost_malloc((void **)&thpool->threads,thpool->nThreads*sizeof(pthread_t)));
        GHOST_CALL_RETURN(ghost_malloc((void **)&thpool->sem,sizeof(sem_t)));

        pthread_key_create(&ghost_thread_key,NULL);

        DEBUG_LOG(1,"All threads are initialized and waiting for tasks");
    } else {
        DEBUG_LOG(1,"Resizing the thread pool");
       
        oldthreads = thpool->nThreads; 
        thpool->nThreads = nThreads;
        sem_init(thpool->sem, 0, 0);

        thpool->threads = realloc(thpool->threads,thpool->nThreads*sizeof(pthread_t));
        
    }
        
    sem_init(thpool->sem, 0, 0);
    for (t=oldthreads; t<thpool->nThreads; t++){
        pthread_create(&(thpool->threads[t]), NULL, func, (void *)(intptr_t)t);
    }
    for (t=oldthreads; t<thpool->nThreads; t++){
        sem_wait(thpool->sem);
    }
        
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_TASKING|GHOST_FUNCTYPE_SETUP); 
    return GHOST_SUCCESS;
}

ghost_error ghost_thpoolhread_add(void *(func)(void *), intptr_t arg)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_TASKING); 
    
    thpool->nThreads++;

    thpool->threads = realloc(thpool->threads,thpool->nThreads*sizeof(pthread_t));
    pthread_create(&(thpool->threads[thpool->nThreads-1]), NULL, func, (void *)arg);
    sem_wait(thpool->sem);
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_TASKING); 
    return GHOST_SUCCESS;
}

ghost_error ghost_thpool_destroy()
{
    if (thpool == NULL)
        return GHOST_SUCCESS;

    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_TASKING|GHOST_FUNCTYPE_TEARDOWN); 

    DEBUG_LOG(1,"Join all threads");
    int t; 
    for (t=0; t<thpool->nThreads; t++)
    {   
//        pthread_cancel(ghost_thpool->threads[t]);
//        printf("cancelled %d %d\n",t,(int)ghost_thpool->threads[t]); 
        if (pthread_join(thpool->threads[t],NULL)){
            ERROR_LOG("pthread_join failed: %s",strerror(errno));
            return GHOST_ERR_UNKNOWN;
        }
    }

    free(thpool->threads); thpool->threads = NULL;
    free(thpool->sem); thpool->sem = NULL;
//    free(ghost_thpool->PUs); ghost_thpool->PUs = NULL;
//    hwloc_bitmap_free(ghost_thpool->cpuset); ghost_thpool->cpuset = NULL;
//    hwloc_bitmap_free(ghost_thpool->busy); ghost_thpool->busy = NULL;
    free(thpool); thpool = NULL;
        
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_TASKING|GHOST_FUNCTYPE_TEARDOWN); 
    return GHOST_SUCCESS;
}

ghost_error ghost_thpool_get(ghost_thpool **thp)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_TASKING); 
    
    *thp = thpool;

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_TASKING); 
    return GHOST_SUCCESS;
}

ghost_error ghost_thpool_key(pthread_key_t *key)
{
    if (!key) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_TASKING); 

    // only valid if thpool was created!
    if (!thpool) {
      ERROR_LOG("ghost_thpool not initialized, yet!");
      return GHOST_ERR_UNKNOWN;
    }
    *key = ghost_thread_key;

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_TASKING); 
    return GHOST_SUCCESS;
}
