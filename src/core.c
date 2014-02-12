#include "ghost/config.h"
#include "ghost/core.h"
#include "ghost/log.h"
#include "ghost/util.h"
#include "ghost/math.h"
#include "ghost/machine.h"
#include "ghost/locality.h"
#include "ghost/taskq.h"
#include "ghost/thpool.h"
#include "ghost/timing.h"
#include "ghost/pumap.h"
#include "ghost/omp.h"

#include <hwloc.h>
#ifdef GHOST_HAVE_INSTR_LIKWID
#include <likwid.h>
#endif

static ghost_type_t ghost_type = GHOST_TYPE_INVALID;
static int MPIwasInitialized = 0;
static unsigned int* ghost_rand_states=NULL;


static ghost_error_t ghost_rand_init()
{
    int N_Th = 1;
#pragma omp parallel
    {
#pragma omp single
        N_Th = ghost_ompGetNumThreads();
    }

    int rank;
    GHOST_CALL_RETURN(ghost_getRank(MPI_COMM_WORLD,&rank));

    if(ghost_rand_states == NULL) {
        ghost_rand_states=(unsigned int*)malloc(N_Th*sizeof(unsigned int));
    }

    ghost_error_t ret = GHOST_SUCCESS;
#pragma omp parallel
    {
        double time;
        GHOST_CALL(ghost_wctimeMilli(&time),ret);

        unsigned int seed=(unsigned int)ghost_hash(
                (int)time,
                rank,
                (int)ghost_ompGetThreadNum());
        ghost_rand_states[ghost_ompGetThreadNum()] = seed;
    }

    if (ret != GHOST_SUCCESS) {
        goto err;
    }

    goto out;
err:
    free(ghost_rand_states);

out:

    return ret;
}

ghost_error_t ghost_setType(ghost_type_t t)
{
    ghost_type = t;

    return GHOST_SUCCESS;
}

ghost_error_t ghost_getType(ghost_type_t *t)
{
    if (!t) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }

    *t = ghost_type;

    return GHOST_SUCCESS;
}

ghost_error_t ghost_init(int argc, char **argv)
{
#ifdef GHOST_HAVE_MPI
    int req, prov;

#ifdef GHOST_HAVE_OPENMP
    req = MPI_THREAD_MULTIPLE; 
#else
    req = MPI_THREAD_SINGLE;
#endif

    MPI_CALL_RETURN(MPI_Initialized(&MPIwasInitialized));
    if (!MPIwasInitialized) {
        MPI_CALL_RETURN(MPI_Init_thread(&argc, &argv, req, &prov));

        if (req != prov) {
            WARNING_LOG("Required MPI threading level (%d) is not "
                    "provided (%d)!",req,prov);
        }
    } else {
        INFO_LOG("MPI was already initialized, not doing it!");
    }


    ghost_setupNodeMPI(MPI_COMM_WORLD);
    ghost_mpi_createDatatypes();
    ghost_mpi_createOperations();

#else // ifdef GHOST_HAVE_MPI
    UNUSED(MPIwasInitialized);
    UNUSED(argc);
    UNUSED(argv);

#endif // ifdef GHOST_HAVE_MPI

#ifdef GHOST_HAVE_INSTR_LIKWID
    LIKWID_MARKER_INIT;

#pragma omp parallel
    LIKWID_MARKER_THREADINIT;
#endif

    hwloc_topology_t topology;

    ghost_createTopology();
    ghost_getTopology(&topology);


    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    hwloc_get_cpubind(topology,cpuset,HWLOC_CPUBIND_PROCESS);
    if (hwloc_bitmap_weight(cpuset) < hwloc_get_nbobjs_by_type(topology,HWLOC_OBJ_PU)) {
        WARNING_LOG("GHOST is running in a restricted CPU set. This is probably not what you want because GHOST cares for pinning itself...");
    }
    hwloc_bitmap_free(cpuset); cpuset = NULL;


    // auto-set rank types 
    ghost_mpi_comm_t nodeComm;
    int nnoderanks;
    int noderank;
    GHOST_CALL_RETURN(ghost_getNodeComm(&nodeComm));
    GHOST_CALL_RETURN(ghost_getNumberOfRanks(nodeComm,&nnoderanks));
    GHOST_CALL_RETURN(ghost_getRank(nodeComm,&noderank));

    int ncudadevs = 0;
    int nnumanodes;
    ghost_getNumberOfNumaNodes(&nnumanodes);

#ifdef GHOST_HAVE_CUDA
    ghost_cu_getDeviceCount(&ncudadevs);
#endif

    ghost_type_t ghost_type;
    GHOST_CALL_RETURN(ghost_getType(&ghost_type));

    if (ghost_type == GHOST_TYPE_INVALID) {
        if (noderank == 0) {
            ghost_setType(GHOST_TYPE_WORK);
        } else if (noderank <= ncudadevs) {
            ghost_setType(GHOST_TYPE_CUDA);
        } else {
            ghost_setType(GHOST_TYPE_WORK);
        }
    } 
    GHOST_CALL_RETURN(ghost_getType(&ghost_type));

#ifndef GHOST_HAVE_CUDA
    if (ghost_type == GHOST_TYPE_CUDA) {
        WARNING_LOG("This rank is supposed to be a CUDA management rank but CUDA is not available. Re-setting GHOST type");
        ghost_setType(GHOST_TYPE_WORK);
    }
#endif


    int nLocalCuda = ghost_type==GHOST_TYPE_CUDA;

    int i;
    int localTypes[nnoderanks];

    for (i=0; i<nnoderanks; i++) {
        localTypes[i] = GHOST_TYPE_INVALID;
    }
    localTypes[noderank] = ghost_type;
#ifdef GHOST_HAVE_MPI
    ghost_mpi_comm_t ghost_node_comm;
    GHOST_CALL_RETURN(ghost_getNodeComm(&ghost_node_comm));
    MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE,&nLocalCuda,1,MPI_INT,MPI_SUM,ghost_node_comm));

#ifdef GHOST_HAVE_CUDA
    if (ncudadevs < nLocalCuda) {
        WARNING_LOG("There are %d CUDA management ranks on this node but only %d CUDA devices.",nLocalCuda,ncudadevs);
    }
#endif


    MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE,&localTypes,nnoderanks,MPI_INT,MPI_MAX,ghost_node_comm));
#endif   

    ghost_hybridmode_t ghost_hybridmode;
    GHOST_CALL_RETURN(ghost_getHybridMode(&ghost_hybridmode));

    int oversubscribed = 0;
    if (ghost_hybridmode == GHOST_HYBRIDMODE_INVALID) {
        if (nnoderanks <=  nLocalCuda+1) {
            GHOST_CALL_RETURN(ghost_setHybridMode(GHOST_HYBRIDMODE_ONEPERNODE));
        } else if (nnoderanks == nLocalCuda+nnumanodes) {
            GHOST_CALL_RETURN(ghost_setHybridMode(GHOST_HYBRIDMODE_ONEPERNUMA));
        } else if (nnoderanks == hwloc_get_nbobjs_by_type(topology,HWLOC_OBJ_CORE)) {
            GHOST_CALL_RETURN(ghost_setHybridMode(GHOST_HYBRIDMODE_ONEPERCORE));
            WARNING_LOG("One MPI process per core not supported");
        } else {
            WARNING_LOG("Invalid number of ranks on node");
            // TODO handle this correctly
            oversubscribed = 1;
        }
    }
    GHOST_CALL_RETURN(ghost_getHybridMode(&ghost_hybridmode));

    hwloc_cpuset_t mycpuset = hwloc_bitmap_alloc();
    hwloc_cpuset_t globcpuset  = hwloc_bitmap_dup(hwloc_topology_get_allowed_cpuset(topology));

    hwloc_obj_t obj;
    ghost_hw_config_t hwconfig;
    ghost_getHwConfig(&hwconfig);

    if (hwconfig.maxCores == GHOST_HW_CONFIG_INVALID) {
        ghost_getNumberOfCores(&hwconfig.maxCores, GHOST_NUMANODE_ANY);
    }
    if (hwconfig.smtLevel == GHOST_HW_CONFIG_INVALID) {
        ghost_getSMTlevel(&hwconfig.smtLevel);
    }
    ghost_setHwConfig(hwconfig);

    int cpu;
    hwloc_bitmap_foreach_begin(cpu,globcpuset);
    obj = hwloc_get_pu_obj_by_os_index(topology,cpu);
    if ((int)obj->sibling_rank >= hwconfig.smtLevel) {
        hwloc_bitmap_clr(globcpuset,cpu);
    }
    if ((int)obj->parent->logical_index >= hwconfig.maxCores) { 
        hwloc_bitmap_clr(globcpuset,cpu);
    }
    hwloc_bitmap_foreach_end();


#ifdef GHOST_HAVE_CUDA
    int cudaDevice = 0;

    for (i=0; i<nnoderanks; i++) {
        if (localTypes[i] == GHOST_TYPE_CUDA) {
            if (i == noderank) {
                ghost_cu_init(cudaDevice);
            }
            cudaDevice++;
        }
    }


    // CUDA ranks have a physical core
    cudaDevice = 0;
    for (i=0; i<nnoderanks; i++) {
        if (localTypes[i] == GHOST_TYPE_CUDA) {
            hwloc_obj_t mynode = hwloc_get_obj_by_type(topology,HWLOC_OBJ_NODE,cudaDevice%nnumanodes);
            hwloc_obj_t runner = mynode;
            while (hwloc_compare_types(runner->type, HWLOC_OBJ_CORE) < 0) {
                runner = runner->first_child;
                char *foo;
                hwloc_bitmap_list_asprintf(&foo,runner->cpuset);
            }
            if (i == noderank) {
                hwloc_bitmap_copy(mycpuset,runner->cpuset);
                //    corestaken[runner->logical_index] = 1;
            }
            cudaDevice++;

            // delete CUDA cores from global cpuset
            hwloc_bitmap_andnot(globcpuset,globcpuset,runner->cpuset);
        }
    }
#endif

    if (ghost_hybridmode == GHOST_HYBRIDMODE_ONEPERNODE) {
        if (ghost_type == GHOST_TYPE_WORK) {
            hwloc_bitmap_copy(mycpuset,globcpuset);
        }
        hwloc_bitmap_andnot(globcpuset,globcpuset,globcpuset);
    } else if (ghost_hybridmode == GHOST_HYBRIDMODE_ONEPERNUMA) {
        int numaNode = 0;
        for (i=0; i<nnoderanks; i++) {
            if (localTypes[i] == GHOST_TYPE_WORK) {
                if (nnumanodes > numaNode) {
                    hwloc_cpuset_t nodeCpuset;
                    if (hwloc_get_nbobjs_by_type(topology,HWLOC_OBJ_NODE) > 0) {
                        nodeCpuset = hwloc_get_obj_by_type(topology,HWLOC_OBJ_NODE,numaNode)->cpuset;
                    } else {
                        nodeCpuset = hwloc_get_obj_by_type(topology,HWLOC_OBJ_SOCKET,numaNode)->cpuset;
                    }
                    if (i == noderank) {
                        hwloc_bitmap_and(mycpuset,globcpuset,nodeCpuset);
                    }
                    hwloc_bitmap_andnot(globcpuset,globcpuset,nodeCpuset);
                    numaNode++;
                } else {
                    oversubscribed = 1;
                    WARNING_LOG("More processes (%d) than NUMA nodes (%d)",numaNode,nnumanodes);
                    break;
                }
            }
        }
    }


    if (oversubscribed) {
        mycpuset = hwloc_bitmap_dup(hwloc_get_obj_by_depth(topology,HWLOC_OBJ_SYSTEM,0)->cpuset);
    }

    void *(*threadFunc)(void *);

    ghost_taskq_create();
    ghost_taskq_getStartRoutine(&threadFunc);
    ghost_thpool_create(hwloc_bitmap_weight(mycpuset),threadFunc);
    ghost_pumap_create(mycpuset);

    ghost_rand_init();

    hwloc_bitmap_free(mycpuset); mycpuset = NULL; 
    hwloc_bitmap_free(globcpuset); globcpuset = NULL;
    return GHOST_SUCCESS;
}

ghost_error_t ghost_getRandState(unsigned int *s)
{
    if (!s) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }
    *s = ghost_rand_states[ghost_ompGetThreadNum()];

    return GHOST_SUCCESS;
}

ghost_error_t ghost_finalize()
{


    free(ghost_rand_states);
    ghost_rand_states=NULL;

#ifdef GHOST_HAVE_INSTR_LIKWID
    LIKWID_MARKER_CLOSE;
#endif

    ghost_mpi_destroyDatatypes();
    ghost_mpi_destroyOperations();

    ghost_taskq_waitall();
    ghost_taskq_destroy();
    ghost_thpool_destroy();
    ghost_pumap_destroy();
    ghost_destroyTopology();

#ifdef GHOST_HAVE_MPI
    if (!MPIwasInitialized) {
        MPI_Finalize();
    }
#endif

    return GHOST_SUCCESS;
}

