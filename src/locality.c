#define _GNU_SOURCE
#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include "ghost/log.h"
#include "ghost/machine.h"
#include "ghost/error.h"
#include "ghost/omp.h"

#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define LOCAL_HOSTNAME_MAX 256
#define ROTL32(num, amount) (((num) << (amount)) | ((num) >> (32 - (amount))))

static ghost_hwconfig my_hwconfig = GHOST_HWCONFIG_INITIALIZER;

static ghost_mpi_comm ghost_node_comm = MPI_COMM_NULL;

static int stringcmp(const void *x, const void *y) { return (strcmp((char *)x, (char *)y)); }

ghost_error ghost_thread_pin(int coreNumber)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL | GHOST_FUNCTYPE_TASKING);
    hwloc_topology_t topology;
    ghost_topology_get(&topology);

    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    hwloc_cpuset_t old_cpuset = hwloc_bitmap_alloc();
    if (!cpuset || !old_cpuset) {
        GHOST_ERROR_LOG("Could not allocate bitmap");
        return GHOST_ERR_HWLOC;
    }

    hwloc_bitmap_set(cpuset, coreNumber);
    int already_pinned = 0;

    if (hwloc_get_cpubind(topology, old_cpuset, HWLOC_CPUBIND_THREAD) != -1) {
        already_pinned = hwloc_bitmap_isequal(old_cpuset, cpuset);
    }


    if (!already_pinned) {
        if (hwloc_set_cpubind(topology, cpuset, HWLOC_CPUBIND_THREAD) == -1) {
            GHOST_ERROR_LOG("Pinning failed: %s", strerror(errno));
            hwloc_bitmap_free(cpuset);
            return GHOST_ERR_HWLOC;
        }
    }
    hwloc_bitmap_free(old_cpuset);
    hwloc_bitmap_free(cpuset);

    GHOST_IF_DEBUG(2)
    {
        int core;
        GHOST_CALL_RETURN(ghost_cpu(&core));
        if (already_pinned) {
            GHOST_DEBUG_LOG(2, "Successfully checked pinning of OpenMP thread %d to core %d",
                ghost_omp_threadnum(), core);
        } else {
            GHOST_DEBUG_LOG(
                2, "Successfully pinned OpenMP thread %d to core %d", ghost_omp_threadnum(), core);
        }
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL | GHOST_FUNCTYPE_TASKING);
    return GHOST_SUCCESS;
}

ghost_error ghost_thread_unpin()
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL | GHOST_FUNCTYPE_TASKING);
    GHOST_IF_DEBUG(2)
    {
        int core;
        GHOST_CALL_RETURN(ghost_cpu(&core));
        GHOST_DEBUG_LOG(2, "Unpinning OpenMP thread %d from core %d", ghost_omp_threadnum(), core);
    }
    hwloc_topology_t topology;
    ghost_topology_get(&topology);

    hwloc_const_cpuset_t cpuset = hwloc_topology_get_allowed_cpuset(topology);
    if (!cpuset) {
        GHOST_ERROR_LOG("Can not get allowed CPU set of entire topology");
        return GHOST_ERR_HWLOC;
    }

    hwloc_set_cpubind(topology, cpuset, HWLOC_CPUBIND_THREAD);

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL | GHOST_FUNCTYPE_TASKING);
    return GHOST_SUCCESS;
}

ghost_error ghost_cpu(int *core)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    hwloc_topology_t topology;
    ghost_topology_get(&topology);

    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    hwloc_get_cpubind(topology, cpuset, HWLOC_CPUBIND_THREAD);

    if (hwloc_bitmap_weight(cpuset) == 0) {
        GHOST_ERROR_LOG("No CPU is set");
        hwloc_bitmap_free(cpuset);
        return GHOST_ERR_HWLOC;
    }

    *core = hwloc_bitmap_first(cpuset);
    hwloc_bitmap_free(cpuset);

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}

ghost_error ghost_rank(int *rank, ghost_mpi_comm comm)
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    MPI_CALL_RETURN(MPI_Comm_rank(comm, rank));
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
#else
    UNUSED(comm);
    UNUSED(rank);
    *rank = 0;
#endif
    return GHOST_SUCCESS;
}

ghost_error ghost_nnode(int *nNodes, ghost_mpi_comm comm)
{
#ifndef GHOST_HAVE_MPI
    UNUSED(stringcmp);
    UNUSED(comm);
    *nNodes = 1;
    return GHOST_SUCCESS;
#else

    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL | GHOST_FUNCTYPE_COMMUNICATION);
    ghost_error ret = GHOST_SUCCESS;
    int nameLen, me, size, i, distinctNames = 1;
    char name[MPI_MAX_PROCESSOR_NAME] = "";
    char *names = NULL;

    GHOST_CALL_RETURN(ghost_rank(&me, comm));
    GHOST_CALL_RETURN(ghost_nrank(&size, comm));
    MPI_Get_processor_name(name, &nameLen);


    if (me == 0) {
        GHOST_CALL_GOTO(
            ghost_malloc((void **)&names, size * MPI_MAX_PROCESSOR_NAME * sizeof(char)), err, ret);
    }


    MPI_CALL_GOTO(MPI_Gather(name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, names, MPI_MAX_PROCESSOR_NAME,
                      MPI_CHAR, 0, comm),
        err, ret);


    if (me == 0) {
        qsort(names, size, MPI_MAX_PROCESSOR_NAME * sizeof(char), stringcmp);
        for (i = 1; i < size; i++) {
            if (strcmp(names + (i - 1) * MPI_MAX_PROCESSOR_NAME, names + i * MPI_MAX_PROCESSOR_NAME)) {
                distinctNames++;
            }
        }
        free(names);
        names = NULL;
    }

    MPI_CALL_GOTO(MPI_Bcast(&distinctNames, 1, MPI_INT, 0, comm), err, ret);

    *nNodes = distinctNames;

    goto out;

err:
    free(names);
    names = NULL;

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL | GHOST_FUNCTYPE_COMMUNICATION);
    return ret;
    ;
#endif
}

ghost_error ghost_nrank(int *nRanks, ghost_mpi_comm comm)
{
#ifdef GHOST_HAVE_MPI
    if (comm == MPI_COMM_NULL) {
        *nRanks = 1;
        return GHOST_SUCCESS;
    }
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    MPI_CALL_RETURN(MPI_Comm_size(comm, nRanks));
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
#else
    UNUSED(comm);
    *nRanks = 1;
#endif
    return GHOST_SUCCESS;
}

ghost_error ghost_hwconfig_set(ghost_hwconfig a)
{
    // function macros disabled because the instrumentation keys get created in ghost_init()
    // and hwconfig_set() is called before that

    // GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    my_hwconfig = a;
    // GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}

ghost_error ghost_hwconfig_get(ghost_hwconfig *hwconfig)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    if (!hwconfig) {
        GHOST_ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }
    *hwconfig = my_hwconfig;

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}

ghost_error ghost_nodecomm_get(ghost_mpi_comm *comm)
{
    if (!comm) {
        GHOST_ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }
#ifdef GHOST_HAVE_MPI
    *comm = ghost_node_comm;
#else
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    UNUSED(ghost_node_comm);
    *comm = 0;
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
#endif

    return GHOST_SUCCESS;
}

ghost_error ghost_nodecomm_setup(ghost_mpi_comm comm)
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL | GHOST_FUNCTYPE_COMMUNICATION);
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &ghost_node_comm);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL | GHOST_FUNCTYPE_COMMUNICATION);
#else
    UNUSED(comm);
#endif

    return GHOST_SUCCESS;
}
