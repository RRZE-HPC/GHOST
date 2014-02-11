#include "ghost/thpool.h"
#include "ghost/util.h"
#include "ghost/machine.h"

/**
 * @brief The thread pool created by ghost_thpool_create(). 
 */
static ghost_thpool_t *ghost_thpool = NULL;


ghost_error_t ghost_thpool_create(int nThreads, void *(func)(void *))
{
    int t;
    int totalThreads;

    if (ghost_thpool) {
        ERROR_LOG("The thread pool has already been initialized.");
        return GHOST_ERR_UNKNOWN;
    }

    totalThreads = nThreads;

    GHOST_CALL_RETURN(ghost_malloc((void **)&ghost_thpool,sizeof(ghost_thpool_t)));
    ghost_thpool->nThreads = nThreads;
//    GHOST_CALL_RETURN(ghost_malloc((void **)&ghost_thpool->PUs,totalThreads*sizeof(hwloc_obj_t)));
    GHOST_CALL_RETURN(ghost_malloc((void **)&ghost_thpool->threads,ghost_thpool->nThreads*sizeof(pthread_t)));
    GHOST_CALL_RETURN(ghost_malloc((void **)&ghost_thpool->sem,sizeof(sem_t)));
    ghost_thpool->nThreads = totalThreads;
    sem_init(ghost_thpool->sem, 0, 0);


    //ghost_thpool->cpuset = hwloc_bitmap_alloc();
    //hwloc_bitmap_copy(ghost_thpool->cpuset,cpuset);
    //ghost_thpool->busy = hwloc_bitmap_alloc();


    //    WARNING_LOG("Creating %d threads for the thread pool",ghost_thpool->nThreads);
    for (t=0; t<ghost_thpool->nThreads; t++){
        pthread_create(&(ghost_thpool->threads[t]), NULL, func, (void *)(intptr_t)t);
    }
    for (t=0; t<ghost_thpool->nThreads; t++){
        sem_wait(ghost_thpool->sem);
    }
    DEBUG_LOG(1,"All threads are initialized and waiting for tasks");

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

