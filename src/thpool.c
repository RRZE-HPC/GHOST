#include "ghost/thpool.h"
#include "ghost/util.h"
#include "ghost/machine.h"

/**
 * @brief The thread pool created by ghost_thpool_init(). This variable is exported in ghost_taskq.h
 */
static ghost_thpool_t *ghost_thpool = NULL;

static int** coreidx;
static int firstThreadOfLD(int ld);

static int intcomp(const void *x, const void *y) 
{
    return (*(int *)x - *(int *)y);
}

int coreIdx(int LD, int t)
{

    return coreidx[LD][t];
}
/**
 * @brief Initializes a thread pool with a given number of threads.
 * @param nThreads
 * @return GHOST_SUCCESS on success or GHOST_FAILURE on failure.
 * 
 * A number of pthreads will be created and each one will have thread_main() as start routine.
 * In order to make sure that each thread has entered the infinite loop, a wait on a semaphore is
 * performed before this function returns.
 */
ghost_error_t ghost_thpool_init(hwloc_cpuset_t cpuset, void *(func)(void *))
{
    static int initialized = 0;
    int t,q,i;
    int totalThreads;
    hwloc_obj_t obj;

    if (initialized) {
        WARNING_LOG("The thread pool has already been initialized.");
        return GHOST_ERR_UNKNOWN;
    }
    if (!cpuset) {
        WARNING_LOG("The thread pool's cpuset is NULL.");
        return GHOST_ERR_INVALID_ARG;
    }
    initialized=1;

    totalThreads = hwloc_bitmap_weight(cpuset);

    GHOST_CALL_RETURN(ghost_malloc((void **)&ghost_thpool,sizeof(ghost_thpool_t)));
    GHOST_CALL_RETURN(ghost_malloc((void **)&ghost_thpool->PUs,totalThreads*sizeof(hwloc_obj_t)));
    GHOST_CALL_RETURN(ghost_malloc((void **)&ghost_thpool->threads,ghost_thpool->nThreads*sizeof(pthread_t)));
    GHOST_CALL_RETURN(ghost_malloc((void **)&ghost_thpool->sem,sizeof(sem_t)));
    ghost_thpool->nThreads = totalThreads;
    sem_init(ghost_thpool->sem, 0, 0);


    ghost_thpool->cpuset = hwloc_bitmap_alloc();
    hwloc_bitmap_copy(ghost_thpool->cpuset,cpuset);
    ghost_thpool->busy = hwloc_bitmap_alloc();

    hwloc_obj_t runner;

    hwloc_topology_t topology;
    ghost_getTopology(&topology);

    int npus = hwloc_get_nbobjs_inside_cpuset_by_type(topology,ghost_thpool->cpuset,HWLOC_OBJ_PU);

    for (i=0; i<npus; i++) {
        ghost_thpool->PUs[i] = hwloc_get_obj_inside_cpuset_by_type(topology,ghost_thpool->cpuset,HWLOC_OBJ_PU,i);
    }

    int nodes[totalThreads];
    for (i=0; i<totalThreads; i++) {
        nodes[i] = 0;
        obj = ghost_thpool->PUs[i];
        for (runner=obj; runner; runner=runner->parent) {
            if (!hwloc_compare_types(runner->type,HWLOC_OBJ_NODE)) {
                nodes[i] = runner->logical_index;
                break;
            }
        }
        //INFO_LOG("Thread # %3d running @ PU %3u (OS: %3u), SMT level %2d, NUMA node %u",i,obj->logical_index,obj->os_index,obj->sibling_rank,nodes[i]);
    }

    qsort(nodes,totalThreads,sizeof(int),intcomp);
    ghost_thpool->nLDs = 1;

    for (i=1; i<totalThreads; i++) {
        if (nodes[i] != nodes[i-1]) {
            ghost_thpool->nLDs++;
        }
    }

    // holds core indices sorted for each locality domain
    GHOST_CALL_RETURN(ghost_malloc((void **)&coreidx,sizeof(int *)*ghost_thpool->nLDs));

    // TODO error goto
    int li;
    // sort the cores according to the locality domain
    for (q=0; q<ghost_thpool->nLDs; q++) {
        int localthreads = nThreadsPerLD(q);
        GHOST_CALL_RETURN(ghost_malloc((void **)&coreidx[q],sizeof(int)*ghost_thpool->nThreads));

        for (t=0; t<localthreads; t++) { // my own threads
            coreidx[q][t] = firstThreadOfLD(q)+t;
            //WARNING_LOG("1 coreidx[%d][%d] = %d",q,t,coreidx[q][t]);
        }
        for (li=0; t-localthreads<firstThreadOfLD(q); t++, li++) { // earlier LDs
            coreidx[q][t] = li;
            //WARNING_LOG("2 coreidx[%d][%d] = %d",q,t,coreidx[q][t]);
        }    
        for (; t<ghost_thpool->nThreads; t++) { // later LDs
            coreidx[q][t] = t;
            //WARNING_LOG("3 coreidx[%d][%d] = %d",q,t,coreidx[q][t]);
        }    
    }

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

    /**
     * @brief Free all resources of the thread pool
     *
     * @return GHOST_SUCCESS on success or GHOST_FAILURE on failure.
     */
    ghost_error_t ghost_thpool_finish()
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
        free(ghost_thpool->PUs); ghost_thpool->PUs = NULL;
        hwloc_bitmap_free(ghost_thpool->cpuset); ghost_thpool->cpuset = NULL;
        hwloc_bitmap_free(ghost_thpool->busy); ghost_thpool->busy = NULL;
        free(ghost_thpool); ghost_thpool = NULL;

        return GHOST_SUCCESS;
    }

ghost_error_t ghost_getThreadpool(ghost_thpool_t **thpool)
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

int nThreadsPerLD(int ld)
{
    int i,n = 0;
    hwloc_obj_t obj,runner;
    for (i=0; i<ghost_thpool->nThreads; i++) {    
        obj = ghost_thpool->PUs[i];
        for (runner=obj; runner; runner=runner->parent) {
            if (runner->type <= HWLOC_OBJ_NODE) {
                if ((int)runner->logical_index == ld) {
                    n++;
                }
                break;
            }
        }
    }

    return n;
}

int nIdleCoresAtLD(int ld)
{
    int begin = firstThreadOfLD(ld), end = begin+nThreadsPerLD(ld);
    hwloc_bitmap_t LDBM = hwloc_bitmap_alloc();

    hwloc_bitmap_set_range(LDBM,begin,end-1);
    hwloc_bitmap_t idle = hwloc_bitmap_alloc();
    hwloc_bitmap_andnot(idle,LDBM,ghost_thpool->busy);

    int w = hwloc_bitmap_weight(idle);

    hwloc_bitmap_free(LDBM);
    hwloc_bitmap_free(idle);
    return w;
}

int nIdleCores()
{
    return ghost_thpool->nThreads-hwloc_bitmap_weight(ghost_thpool->busy);
}


int nBusyCoresAtLD(hwloc_bitmap_t bm, int ld)
{
    int begin = firstThreadOfLD(ld), end = begin+nThreadsPerLD(ld);
    hwloc_bitmap_t LDBM = hwloc_bitmap_alloc();

    hwloc_bitmap_set_range(LDBM,begin,end-1);
    hwloc_bitmap_t busy = hwloc_bitmap_alloc();
    hwloc_bitmap_and(busy,LDBM,bm);

    int w = hwloc_bitmap_weight(busy);

    hwloc_bitmap_free(LDBM);
    hwloc_bitmap_free(busy);
    return w;
}

static int firstThreadOfLD(int ld)
{
    int i;
    hwloc_obj_t obj,runner;

    for (i=0; i<ghost_thpool->nThreads; i++) {    
        obj = ghost_thpool->PUs[i];
        for (runner=obj; runner; runner=runner->parent) {
            if (runner->type <= HWLOC_OBJ_NODE) {
                if ((int)runner->logical_index == ld) {
                    return i;
                }
            }
        }
    }

    return -1;
}

