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

#define LOCAL_HOSTNAME_MAX 	256
#define ROTL32(num,amount) (((num) << (amount)) | ((num) >> (32 - (amount))))

static ghost_hwconfig_t ghost_hwconfig = GHOST_HWCONFIG_INITIALIZER;

static ghost_mpi_comm_t ghost_node_comm = MPI_COMM_NULL;

static int stringcmp(const void *x, const void *y)
{
    return (strcmp((char *)x, (char *)y));
}

ghost_error_t ghost_thread_pin(int coreNumber)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_TASKING);
    hwloc_topology_t topology;
    ghost_topology_get(&topology);
    
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    hwloc_cpuset_t old_cpuset = hwloc_bitmap_alloc();
    if (!cpuset || !old_cpuset) {
        ERROR_LOG("Could not allocate bitmap");
        return GHOST_ERR_HWLOC;
    }

    hwloc_bitmap_set(cpuset,coreNumber);
    int already_pinned = 0;

    if (hwloc_get_cpubind(topology,old_cpuset,HWLOC_CPUBIND_THREAD) != -1) {
        already_pinned = hwloc_bitmap_isequal(old_cpuset,cpuset);
    }

    
    if (!already_pinned) {
        if (hwloc_set_cpubind(topology,cpuset,HWLOC_CPUBIND_THREAD) == -1) {
            ERROR_LOG("Pinning failed: %s",strerror(errno));
            hwloc_bitmap_free(cpuset);
            return GHOST_ERR_HWLOC;
        }
    }
    hwloc_bitmap_free(old_cpuset);
    hwloc_bitmap_free(cpuset);
    
    IF_DEBUG(2) {
        int core;
        GHOST_CALL_RETURN(ghost_cpu(&core));
        if (already_pinned) {
            DEBUG_LOG(2,"Successfully checked pinning of OpenMP thread %d to core %d",ghost_omp_threadnum(),core);
        } else {
            DEBUG_LOG(2,"Successfully pinned OpenMP thread %d to core %d",ghost_omp_threadnum(),core);
        }
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_TASKING);
    return GHOST_SUCCESS;
}

ghost_error_t ghost_thread_unpin()
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_TASKING);
    IF_DEBUG(2) {
        int core;
        GHOST_CALL_RETURN(ghost_cpu(&core));
        DEBUG_LOG(2,"Unpinning OpenMP thread %d from core %d",ghost_omp_threadnum(),core);
    }
    hwloc_topology_t topology;
    ghost_topology_get(&topology);
   
    hwloc_const_cpuset_t cpuset = hwloc_topology_get_allowed_cpuset(topology);
    if (!cpuset) {
        ERROR_LOG("Can not get allowed CPU set of entire topology");
        return GHOST_ERR_HWLOC;
    }

    hwloc_set_cpubind(topology,cpuset,HWLOC_CPUBIND_THREAD);

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_TASKING);
    return GHOST_SUCCESS;
}

ghost_error_t ghost_cpu(int *core)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    hwloc_topology_t topology;
    ghost_topology_get(&topology);
    
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    hwloc_get_cpubind(topology,cpuset,HWLOC_CPUBIND_THREAD);

    if (hwloc_bitmap_weight(cpuset) == 0) {
        ERROR_LOG("No CPU is set");
        hwloc_bitmap_free(cpuset);
        return GHOST_ERR_HWLOC;
    }

    *core = hwloc_bitmap_first(cpuset);
    hwloc_bitmap_free(cpuset);
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}

ghost_error_t ghost_rank(int *rank, ghost_mpi_comm_t comm) 
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    MPI_CALL_RETURN(MPI_Comm_rank(comm,rank));
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
#else
    UNUSED(comm);
    UNUSED(rank);
    *rank = 0;
#endif
    return GHOST_SUCCESS;
}

ghost_error_t ghost_nnode(int *nNodes, ghost_mpi_comm_t comm)
{
#ifndef GHOST_HAVE_MPI
    UNUSED(stringcmp);
    UNUSED(comm);
    *nNodes = 1;
    return GHOST_SUCCESS;
#else

    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_COMMUNICATION);
    ghost_error_t ret = GHOST_SUCCESS;
    int nameLen,me,size,i,distinctNames = 1;
    char name[MPI_MAX_PROCESSOR_NAME] = "";
    char *names = NULL;

    GHOST_CALL_RETURN(ghost_rank( &me,  comm));
    GHOST_CALL_RETURN(ghost_nrank( &size,  comm));
    MPI_Get_processor_name(name,&nameLen);


    if (me==0) {
        GHOST_CALL_GOTO(ghost_malloc((void **)&names,size*MPI_MAX_PROCESSOR_NAME*sizeof(char)),err,ret);
    }


    MPI_CALL_GOTO(MPI_Gather(name,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,names,
                MPI_MAX_PROCESSOR_NAME,MPI_CHAR,0,comm),err,ret);


    if (me==0) {
        qsort(names,size,MPI_MAX_PROCESSOR_NAME*sizeof(char),stringcmp);
        for (i=1; i<size; i++) {
            if (strcmp(names+(i-1)*MPI_MAX_PROCESSOR_NAME,names+
                        i*MPI_MAX_PROCESSOR_NAME)) {
                distinctNames++;
            }
        }
        free(names); names = NULL;
    }

    MPI_CALL_GOTO(MPI_Bcast(&distinctNames,1,MPI_INT,0,comm),err,ret);

    *nNodes = distinctNames;

    goto out;

err:
    free(names); names = NULL;

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_COMMUNICATION);
    return ret;;
#endif
}

ghost_error_t ghost_nrank(int *nRanks, ghost_mpi_comm_t comm)
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    MPI_CALL_RETURN(MPI_Comm_size(comm,nRanks));
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
#else
    UNUSED(comm);
    *nRanks = 1;
#endif
    return GHOST_SUCCESS;
}

ghost_error_t ghost_hwconfig_set(ghost_hwconfig_t a)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    ghost_hwconfig = a;
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS; 
}

ghost_error_t ghost_hwconfig_get(ghost_hwconfig_t * hwconfig)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    if (!hwconfig) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }
    *hwconfig = ghost_hwconfig;
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}

//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.
static void MurmurHash3_x86_32 ( const void * key, int len,
        uint32_t seed, void * out )
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    const uint8_t * data = (const uint8_t*)key;
    const int nblocks = len / 4;

    uint32_t h1 = seed;

    uint32_t c1 = 0xcc9e2d51;
    uint32_t c2 = 0x1b873593;

    //----------
    // body

    const uint32_t * blocks = (const uint32_t *)(data + nblocks*4);

    for(int i = -nblocks; i; i++)
    {
        uint32_t k1 = blocks[i];

        k1 *= c1;
        k1 = ROTL32(k1,15);
        k1 *= c2;

        h1 ^= k1;
        h1 = ROTL32(h1,13);
        h1 = h1*5+0xe6546b64;
    }

    //----------
    // tail

    const uint8_t * tail = (const uint8_t*)(data + nblocks*4);

    uint32_t k1 = 0;

    switch(len & 3)
    {
        case 3: k1 ^= tail[2] << 16;
        case 2: k1 ^= tail[1] << 8;
        case 1: k1 ^= tail[0];
                k1 *= c1; k1 = ROTL32(k1,15); k1 *= c2; h1 ^= k1;
    };

    //----------
    // finalization

    h1 ^= len;

    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >> 16;


    *(uint32_t*)out = h1;
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
} 

static ghost_error_t ghost_hostname(char ** hostnamePtr, size_t * hostnameLength)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    // Trace();

    char * hostname = NULL;
    size_t nHostname = 0;

    int allocateMore = 0;

    *hostnamePtr = NULL;
    *hostnameLength = 0;

    do {
        nHostname += MAX(HOST_NAME_MAX, LOCAL_HOSTNAME_MAX);

        GHOST_CALL_RETURN(ghost_malloc((void **)&hostname,sizeof(char) * nHostname));
        memset(hostname,0,nHostname);

        int error;

        error = gethostname(hostname, nHostname);

        if (error == -1) {
            if (errno == ENAMETOOLONG) {
                allocateMore = 1;
                free(hostname); hostname = NULL;
            }
            else {
                free(hostname);
                hostname = NULL;

                ERROR_LOG("gethostname failed with error %d: %s", errno, strerror(errno));
                return GHOST_ERR_UNKNOWN;
            }

        }
        else {
            allocateMore = 0;
        }

    } while (allocateMore);

    // Make sure hostname is \x00 terminated.
    hostname[nHostname - 1] = 0x00;

    *hostnameLength = strnlen(hostname, nHostname) + 1;
    *hostnamePtr = hostname;

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}

ghost_error_t ghost_nodecomm_get(ghost_mpi_comm_t *comm)
{
    if (!comm) {
        ERROR_LOG("NULL pointer");
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

ghost_error_t ghost_nodecomm_setup(ghost_mpi_comm_t comm)
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_COMMUNICATION);
    int mpiRank;
    GHOST_CALL_RETURN(ghost_rank( &mpiRank,  comm));
    int error;
    char * hostname = NULL;
    size_t hostnameLength = 0;

    error = ghost_hostname(&hostname, &hostnameLength);
    if (error != 0) {
        return GHOST_ERR_MPI;
    }

    uint32_t checkSum;
    MurmurHash3_x86_32(hostname, hostnameLength, 1234172, &checkSum);
    int checkSumSigned = (int)(checkSum>>1);

    int commRank = -1;
    MPI_CALL_RETURN(MPI_Comm_rank(comm, &commRank));

    MPI_Comm nodeComm = MPI_COMM_NULL;

    DEBUG_LOG(2," comm_split:  color:  %u  rank:  %d   hostnameLength: %zu", checkSum, mpiRank, hostnameLength);

    MPI_CALL_RETURN(MPI_Comm_split(comm, checkSumSigned, mpiRank, &nodeComm));

    int nodeRank;
    MPI_CALL_RETURN(MPI_Comm_rank(nodeComm, &nodeRank));

    int nodeSize;
    MPI_CALL_RETURN(MPI_Comm_size(nodeComm, &nodeSize));

    // Determine if collisions of the hashed hostname occured.

    int nSend = MAX(HOST_NAME_MAX, LOCAL_HOSTNAME_MAX);
    char * send = NULL;
    GHOST_CALL_RETURN(ghost_malloc((void **)&send,sizeof(char) * nSend));
    memset(send,0,nSend);
    
    strncpy(send, hostname, strlen(hostname));

    // Ensure terminating \x00 at the end, this may not be
    // garanteed if if len(send) = nSend.
    send[nSend - 1] = 0x00;

    char * recv = (char *)malloc(sizeof(char) * nSend * nodeSize);
    MPI_CALL(MPI_Allgather(send, nSend, MPI_CHAR, recv, nSend, MPI_CHAR, nodeComm),error);
    if (error != MPI_SUCCESS) {
        free(send); send = NULL;
        free(recv); recv = NULL;
        free(hostname); hostname = NULL;
        return GHOST_ERR_MPI;
    }

    char * neighbor = recv;
    int localNodeRank = 0;

#define STREQ(a, b)  (strcmp((a), (b)) == 0)

    // recv contains now an array of hostnames from all MPI ranks of
    // this communicator. They are sorted ascending by the MPI rank.
    // Also if collisions occur these are handled here.

    for (int i = 0; i < nodeSize; ++i) {

        if (STREQ(send, neighbor)) {
            if (i < nodeRank) {
                // Compared neighbors still have lower rank than we have.
                ++localNodeRank;
            }
            else {
                break;
            }
        }
        else {
            // Collision of the hash.
        }

        neighbor += nSend;
    }

#undef STREQ


    if (nodeRank != localNodeRank) {
        INFO_LOG("Collisions occured during node rank determinaton: "
                "node rank:  %5d, local node rank:  %5d, host: %s",
                nodeRank, localNodeRank, send);
        WARNING_LOG("The nodal rank is fixed now but the nodal communicator is not. This will lead to problems...");
        nodeRank = localNodeRank;
    }
    MPI_CALL(MPI_Comm_split(comm, checkSumSigned, mpiRank, &nodeComm),error);
    if (error != MPI_SUCCESS) {
        free(send); send = NULL;
        free(recv); recv = NULL;
        free(hostname); hostname = NULL;

        return GHOST_ERR_MPI;
    }


    // Clean up.

    free(send); send = NULL;
    free(recv); recv = NULL;

    ghost_node_comm = nodeComm;

    free(hostname); hostname = NULL;
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_COMMUNICATION);
#else
    UNUSED(comm);
    UNUSED(&MurmurHash3_x86_32);
    UNUSED(&ghost_hostname);
#endif

    return GHOST_SUCCESS;
}
