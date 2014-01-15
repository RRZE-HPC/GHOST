#define _GNU_SOURCE
#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/constants.h"
#include "ghost/util.h"
#include "ghost/affinity.h"

#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>


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

void ghost_setCore(int coreNumber)
{
    DEBUG_LOG(2,"Pinning thread %d to core %d",ghost_ompGetThreadNum(),coreNumber);
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    hwloc_bitmap_set(cpuset,coreNumber);
    if (hwloc_set_cpubind(topology,cpuset,HWLOC_CPUBIND_THREAD)) {
        WARNING_LOG("CPU binding failed: %s",strerror(errno));
    }
    hwloc_bitmap_free(cpuset);

}

void ghost_unsetCore()
{
    DEBUG_LOG(2,"Unpinning thread %d from core %d",ghost_ompGetThreadNum(),ghost_getCore());
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    hwloc_bitmap_set_range(cpuset,0,hwloc_get_nbobjs_by_type(topology,HWLOC_OBJ_PU));
    hwloc_set_cpubind(topology,cpuset,HWLOC_CPUBIND_THREAD);
    hwloc_bitmap_free(cpuset);
}

int ghost_getCore()
{
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    hwloc_get_cpubind(topology,cpuset,HWLOC_CPUBIND_THREAD);
    int pu = hwloc_bitmap_first(cpuset);
    hwloc_bitmap_free(cpuset);
    return pu;
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

int ghost_getNumberOfPhysicalCores()
{
    return hwloc_get_nbobjs_by_type(topology,HWLOC_OBJ_CORE);    
}

int ghost_getNumberOfHwThreads()
{
    return hwloc_get_nbobjs_by_type(topology,HWLOC_OBJ_PU);    
}

int ghost_getSMTlevel()
{
    return ghost_getNumberOfHwThreads()/ghost_getNumberOfPhysicalCores();
}

int ghost_getNumberOfThreads() 
{
    int nthreads;
#pragma omp parallel
    nthreads = ghost_ompGetNumThreads();

    return nthreads;
}

int ghost_getNumberOfNumaNodes()
{
    int depth = hwloc_get_type_depth(topology,HWLOC_OBJ_NODE);
    return hwloc_get_nbobjs_by_depth(topology,depth);
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
