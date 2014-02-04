#define _GNU_SOURCE
#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/constants.h"
#include "ghost/util.h"
#include "ghost/affinity.h"
#include "ghost/log.h"
#include "ghost/machine.h"
#include "ghost/error.h"

#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

const ghost_hw_config_t GHOST_HW_CONFIG_INITIALIZER = {.maxCores = GHOST_HW_CONFIG_INVALID, .smtLevel = GHOST_HW_CONFIG_INVALID};
static ghost_hw_config_t ghost_hw_config = {.maxCores = GHOST_HW_CONFIG_INVALID, .smtLevel = GHOST_HW_CONFIG_INVALID};

static int stringcmp(const void *x, const void *y)
{
    return (strcmp((char *)x, (char *)y));
}

void ghost_pinThreads(int options, char *procList)
{
    if (procList != NULL) {
        char *list = strdup(procList);
        DEBUG_LOG(1,"Setting number of threads and pinning them to cores %s",list);

        const char delim[] = ",";
        char *coreStr;
        int *cores = NULL;
        int nCores = 0;

        coreStr = strtok(list,delim);
        while(coreStr != NULL) 
        {
            nCores++;
            cores = (int *)realloc(cores,nCores*sizeof(int));
            cores[nCores-1] = atoi(coreStr);
            coreStr = strtok(NULL,delim);
        }

        DEBUG_LOG(1,"Adjusting number of threads to %d",nCores);
        ghost_ompSetNumThreads(nCores);

        if (cores != NULL) {
#pragma omp parallel
            ghost_setCore(cores[ghost_ompGetThreadNum()]);
        }

        free(list);
        free(cores);
    } else {
        DEBUG_LOG(1,"Trying to automatically pin threads");

        hwloc_topology_t topology;
        ghost_getTopology(&topology);

        int nranks = ghost_getNumberOfRanks(ghost_node_comm);
        int npus = hwloc_get_nbobjs_by_type(topology,HWLOC_OBJ_PU);
        int ncores = hwloc_get_nbobjs_by_type(topology,HWLOC_OBJ_CORE);
        int nthreads;
        if (options & GHOST_PIN_SMT) {
            nthreads = npus/nranks;
        } else {
            nthreads = ncores/nranks;
        }
    
        ghost_ompSetNumThreads(nthreads);    
        int t;
        hwloc_obj_t pu = hwloc_get_obj_by_type(topology,HWLOC_OBJ_PU,0);

#pragma omp parallel for ordered schedule(static,1)
        for (t=0; t<nthreads; t++) {
#pragma omp ordered
            for (; pu != NULL; pu=pu->next_cousin) {
                if ((options & GHOST_PIN_PHYS) && (pu->sibling_rank != 0)) {
                    continue;
                }
                ghost_setCore(pu->os_index);

                pu = pu->next_cousin;
                break;
            }
        }
    }
}

ghost_error_t ghost_setCore(int coreNumber)
{
    IF_DEBUG(2) {
        int core;
        GHOST_CALL_RETURN(ghost_getCore(&core));
        DEBUG_LOG(2,"Pinning OpenMP thread %d to core %d",ghost_ompGetThreadNum(),core);
    }
    hwloc_topology_t topology;
    ghost_getTopology(&topology);
    
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    if (!cpuset) {
        ERROR_LOG("Could not allocate bitmap");
        return GHOST_ERR_HWLOC;
    }

    hwloc_bitmap_set(cpuset,coreNumber);
    if (hwloc_set_cpubind(topology,cpuset,HWLOC_CPUBIND_THREAD) == -1) {
        ERROR_LOG("Pinning failed: %s",strerror(errno));
        hwloc_bitmap_free(cpuset);
        return GHOST_ERR_HWLOC;
    }
    hwloc_bitmap_free(cpuset);

    return GHOST_SUCCESS;
}

ghost_error_t ghost_unsetCore()
{
    IF_DEBUG(2) {
        int core;
        GHOST_CALL_RETURN(ghost_getCore(&core));
        DEBUG_LOG(2,"Unpinning OpenMP thread %d from core %d",ghost_ompGetThreadNum(),core);
    }
    hwloc_topology_t topology;
    ghost_getTopology(&topology);
   
    hwloc_const_cpuset_t cpuset = hwloc_topology_get_allowed_cpuset(topology);
    if (!cpuset) {
        ERROR_LOG("Can not get allowed CPU set of entire topology");
        return GHOST_ERR_HWLOC;
    }

    hwloc_set_cpubind(topology,cpuset,HWLOC_CPUBIND_THREAD);

    return GHOST_SUCCESS;
}

ghost_error_t ghost_getCore(int *core)
{
    hwloc_topology_t topology;
    ghost_getTopology(&topology);
    
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    hwloc_get_cpubind(topology,cpuset,HWLOC_CPUBIND_THREAD);

    if (hwloc_bitmap_weight(cpuset) == 0) {
        ERROR_LOG("No CPU is set");
        hwloc_bitmap_free(cpuset);
        return GHOST_ERR_HWLOC;
    }

    *core = hwloc_bitmap_first(cpuset);
    hwloc_bitmap_free(cpuset);
    
    return GHOST_SUCCESS;
}

int ghost_getRank(ghost_mpi_comm_t comm) 
{
#ifdef GHOST_HAVE_MPI
    int rank;
    MPI_safecall(MPI_Comm_rank ( comm, &rank ));
    return rank;
#else
    UNUSED(comm);
    return 0;
#endif
}

int ghost_getNumberOfNodes() 
{
#ifndef GHOST_HAVE_MPI
    UNUSED(stringcmp);
    return 1;
#else

    int nameLen,me,size,i,distinctNames = 1;
    char name[MPI_MAX_PROCESSOR_NAME] = "";
    char *names = NULL;

    MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD,&me));
    MPI_safecall(MPI_Comm_size(MPI_COMM_WORLD,&size));
    MPI_safecall(MPI_Get_processor_name(name,&nameLen));


    if (me==0) {
        names = ghost_malloc(size*MPI_MAX_PROCESSOR_NAME*sizeof(char));
    }


    MPI_safecall(MPI_Gather(name,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,names,
                MPI_MAX_PROCESSOR_NAME,MPI_CHAR,0,MPI_COMM_WORLD));

    if (me==0) {
        qsort(names,size,MPI_MAX_PROCESSOR_NAME*sizeof(char),stringcmp);
        for (i=1; i<size; i++) {
            if (strcmp(names+(i-1)*MPI_MAX_PROCESSOR_NAME,names+
                        i*MPI_MAX_PROCESSOR_NAME)) {
                distinctNames++;
            }
        }
        free(names);
    }

    MPI_safecall(MPI_Bcast(&distinctNames,1,MPI_INT,0,MPI_COMM_WORLD));

    return distinctNames;
#endif
}

int ghost_getNumberOfRanks(ghost_mpi_comm_t comm)
{
#ifdef GHOST_HAVE_MPI
    int nnodes;
    MPI_safecall(MPI_Comm_size(comm, &nnodes));
    return nnodes;
#else
    UNUSED(comm);
    return 1;
#endif

}

ghost_error_t ghost_setHwConfig(ghost_hw_config_t a)
{
   ghost_hw_config = a;
   return GHOST_SUCCESS; 
}

ghost_error_t ghost_getHwConfig(ghost_hw_config_t * hwconfig)
{
    hwconfig->maxCores = ghost_hw_config.maxCores;
    hwconfig->smtLevel = ghost_hw_config.smtLevel;
    
    return GHOST_SUCCESS;
}
