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
static ghost_hybridmode_t ghost_hybridmode = GHOST_HYBRIDMODE_INVALID;

static int stringcmp(const void *x, const void *y)
{
    return (strcmp((char *)x, (char *)y));
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

ghost_error_t ghost_getRank(ghost_mpi_comm_t comm, int *rank) 
{
#ifdef GHOST_HAVE_MPI
    MPI_CALL_RETURN(MPI_Comm_rank(comm,rank));
#else
    UNUSED(comm);
    UNUSED(rank);
    *rank = 0;
#endif
    return GHOST_SUCCESS;
}

ghost_error_t ghost_getNumberOfNodes(ghost_mpi_comm_t comm, int *nNodes)
{
#ifndef GHOST_HAVE_MPI
    UNUSED(stringcmp);
    UNUSED(comm);
    UNUSED(nNodes);
    return 1;
#else

    int nameLen,me,size,i,distinctNames = 1;
    char name[MPI_MAX_PROCESSOR_NAME] = "";
    char *names = NULL;

    GHOST_CALL_RETURN(ghost_getRank(comm,&me));
    GHOST_CALL_RETURN(ghost_getNumberOfRanks(comm,&size));
    MPI_Get_processor_name(name,&nameLen);


    if (me==0) {
        names = ghost_malloc(size*MPI_MAX_PROCESSOR_NAME*sizeof(char));
    }


    MPI_safecall(MPI_Gather(name,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,names,
                MPI_MAX_PROCESSOR_NAME,MPI_CHAR,0,comm));

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

    MPI_safecall(MPI_Bcast(&distinctNames,1,MPI_INT,0,comm));

    *nNodes = distinctNames;
#endif
    return GHOST_SUCCESS;
}

ghost_error_t ghost_getNumberOfRanks(ghost_mpi_comm_t comm, int *nRanks)
{
#ifdef GHOST_HAVE_MPI
    MPI_CALL_RETURN(MPI_Comm_size(comm,nRanks));
#else
    UNUSED(comm);
    UNUSED(nRanks);
    return 1;
#endif
    return GHOST_SUCCESS;
}

ghost_error_t ghost_setHwConfig(ghost_hw_config_t a)
{
   ghost_hw_config = a;
   return GHOST_SUCCESS; 
}

ghost_error_t ghost_getHwConfig(ghost_hw_config_t * hwconfig)
{
    if (!hwconfig) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }
    hwconfig->maxCores = ghost_hw_config.maxCores;
    hwconfig->smtLevel = ghost_hw_config.smtLevel;
    
    return GHOST_SUCCESS;
}

ghost_error_t ghost_setHybridMode(ghost_hybridmode_t hm)
{
    ghost_hybridmode = hm;

    return GHOST_SUCCESS;
}

ghost_error_t ghost_getHybridMode(ghost_hybridmode_t *hm)
{
    if (!hm) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }
    *hm = ghost_hybridmode;

    return GHOST_SUCCESS;

}
