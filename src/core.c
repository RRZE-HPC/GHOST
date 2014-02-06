//#include "ghost/config.h"
//#include "ghost/types.h"
//#include "ghost/constants.h"

//#include <stddef.h>

#include <hwloc.h>

#include "ghost/core.h"
#include "ghost/log.h"
#include "ghost/util.h"
#include "ghost/math.h"
#include "ghost/machine.h"
#include "ghost/affinity.h"
#include "ghost/task.h"

static ghost_type_t ghost_type = GHOST_TYPE_INVALID;
static int MPIwasInitialized;
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

#pragma omp parallel
    {
        unsigned int seed=(unsigned int)ghost_hash(
                (int)ghost_wctimemilli(),
                rank,
                (int)ghost_ompGetThreadNum());
        ghost_rand_states[ghost_ompGetThreadNum()] = seed;
    }

    return GHOST_SUCCESS;
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

    MPI_safecall(MPI_Initialized(&MPIwasInitialized));
    if (!MPIwasInitialized) {
        MPI_safecall(MPI_Init_thread(&argc, &argv, req, &prov ));

        if (req != prov) {
            WARNING_LOG("Required MPI threading level (%d) is not "
                    "provided (%d)!",req,prov);
        }
    } else {
        WARNING_LOG("MPI was already initialized, not doing it!");
    }

    MPI_safecall(MPI_Type_contiguous(2,MPI_FLOAT,&GHOST_MPI_DT_C));
    MPI_safecall(MPI_Type_commit(&GHOST_MPI_DT_C));
    MPI_safecall(MPI_Op_create((MPI_User_function *)&ghost_mpi_add_c,1,&GHOST_MPI_OP_SUM_C));

    MPI_safecall(MPI_Type_contiguous(2,MPI_DOUBLE,&GHOST_MPI_DT_Z));
    MPI_safecall(MPI_Type_commit(&GHOST_MPI_DT_Z));
    MPI_safecall(MPI_Op_create((MPI_User_function *)&ghost_mpi_add_z,1,&GHOST_MPI_OP_SUM_Z));

    ghost_setupNodeMPI(MPI_COMM_WORLD);

#else // ifdef GHOST_HAVE_MPI
    UNUSED(argc);
    UNUSED(argv);

#endif // ifdef GHOST_HAVE_MPI

#if GHOST_HAVE_INSTR_LIKWID
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
    hwloc_bitmap_free(cpuset);


    // auto-set rank types 
    int nnoderanks;
    int noderank;
    GHOST_CALL_RETURN(ghost_getNumberOfRanks(ghost_node_comm,&nnoderanks));
    GHOST_CALL_RETURN(ghost_getRank(ghost_node_comm,&noderank));

    int ncudadevs = 0;
    int ndomains = 0;
    int nnumanodes;
    ghost_getNumberOfNumaNodes(&nnumanodes);

    ndomains += nnumanodes;
#if GHOST_HAVE_CUDA
    CU_getDeviceCount(&ncudadevs);
#endif
    ndomains += ncudadevs;

    ghost_type_t ghost_type;
    GHOST_CALL_RETURN(ghost_getType(&ghost_type));

    if (ghost_type == GHOST_TYPE_INVALID) {
        if (noderank == 0) {
            ghost_setType(GHOST_TYPE_COMPUTE);
        } else if (noderank <= ncudadevs) {
            ghost_setType(GHOST_TYPE_CUDAMGMT);
        } else {
            ghost_setType(GHOST_TYPE_COMPUTE);
        }
    } 

#ifndef GHOST_HAVE_CUDA
    if (ghost_type == GHOST_TYPE_CUDAMGMT) {
        WARNING_LOG("This rank is supposed to be a CUDA management rank but CUDA is not available. Re-setting GHOST type");
        ghost_setType(GHOST_TYPE_COMPUTE);
    }
#endif


    int nLocalCompute = ghost_type==GHOST_TYPE_COMPUTE;
    int nLocalCuda = ghost_type==GHOST_TYPE_CUDAMGMT;

    int i;
    int localTypes[nnoderanks];

    for (i=0; i<nnoderanks; i++) {
        localTypes[i] = GHOST_TYPE_INVALID;
    }
    localTypes[noderank] = ghost_type;
#if GHOST_HAVE_MPI
    MPI_safecall(MPI_Allreduce(MPI_IN_PLACE,&nLocalCompute,1,MPI_INT,MPI_SUM,ghost_node_comm));
    MPI_safecall(MPI_Allreduce(MPI_IN_PLACE,&nLocalCuda,1,MPI_INT,MPI_SUM,ghost_node_comm));

#ifdef GHOST_HAVE_CUDA
    if (ncudadevs < nLocalCuda) {
        WARNING_LOG("There are %d CUDA management ranks on this node but only %d CUDA devices.",nLocalCuda,ncudadevs);
    }
#endif


    MPI_safecall(MPI_Allreduce(MPI_IN_PLACE,&localTypes,nnoderanks,MPI_INT,MPI_MAX,ghost_node_comm));
#endif   

    ghost_hybridmode_t ghost_hybridmode;
    GHOST_CALL_RETURN(ghost_getHybridMode(&ghost_hybridmode));

    int oversubscribed = 0;
    if (ghost_hybridmode == GHOST_HYBRIDMODE_INVALID) {
        if (nnoderanks <=  nLocalCuda+1) {
            GHOST_CALL_RETURN(ghost_setHybridMode(GHOST_HYBRIDMODE_ONEPERNODE));
            INFO_LOG("One CPU rank per node");
        } else if (nnoderanks == nLocalCuda+nnumanodes) {
            GHOST_CALL_RETURN(ghost_setHybridMode(GHOST_HYBRIDMODE_ONEPERNUMA));
            INFO_LOG("One CPU rank per NUMA domain");
        } else if (nnoderanks == hwloc_get_nbobjs_by_type(topology,HWLOC_OBJ_CORE)) {
            GHOST_CALL_RETURN(ghost_setHybridMode(GHOST_HYBRIDMODE_ONEPERCORE));
            WARNING_LOG("One MPI process per core not supported");
        } else {
            WARNING_LOG("Invalid number of ranks on node");
            // TODO handle this correctly
            oversubscribed = 1;
        }
    }

    hwloc_cpuset_t mycpuset = hwloc_bitmap_alloc();
    hwloc_cpuset_t globcpuset = hwloc_bitmap_alloc();

    globcpuset = hwloc_bitmap_dup(hwloc_topology_get_allowed_cpuset(topology));

    hwloc_obj_t obj;
    ghost_hw_config_t hwconfig;
    ghost_getHwConfig(&hwconfig);

    if (hwconfig.maxCores == GHOST_HW_CONFIG_INVALID) {
        ghost_getNumberOfPhysicalCores(&hwconfig.maxCores);
    }
    if (hwconfig.smtLevel == GHOST_HW_CONFIG_INVALID) {
        ghost_getSMTlevel(&hwconfig.smtLevel);
    }
    ghost_setHwConfig(hwconfig);

    int cpu;
    hwloc_bitmap_foreach_begin(cpu,globcpuset);
    obj = hwloc_get_pu_obj_by_os_index(topology,cpu);
    if (obj->sibling_rank >= hwconfig.smtLevel) {
        hwloc_bitmap_clr(globcpuset,cpu);
    }
    if (obj->parent->logical_index >= hwconfig.maxCores) { 
        hwloc_bitmap_clr(globcpuset,cpu);
    }
    hwloc_bitmap_foreach_end();


#if GHOST_HAVE_CUDA
    int cudaDevice = 0;

    for (i=0; i<ghost_getNumberOfRanks(ghost_node_comm); i++) {
        if (localTypes[i] == GHOST_TYPE_CUDAMGMT) {
            if (i == ghost_getRank(ghost_node_comm)) {
                ghost_CUDA_init(cudaDevice);
            }
            cudaDevice++;
        }
    }


    // CUDA ranks have a physical core
    cudaDevice = 0;
    for (i=0; i<nnoderanks; i++) {
        if (localTypes[i] == GHOST_TYPE_CUDAMGMT) {
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
        if (ghost_type == GHOST_TYPE_COMPUTE) {
            hwloc_bitmap_copy(mycpuset,globcpuset);
        }
        hwloc_bitmap_andnot(globcpuset,globcpuset,globcpuset);
    } else if (ghost_hybridmode == GHOST_HYBRIDMODE_ONEPERNUMA) {
        int numaNode = 0;
        for (i=0; i<nnoderanks; i++) {
            if (localTypes[i] == GHOST_TYPE_COMPUTE) {
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


    char *cpusetstr;
    hwloc_bitmap_list_asprintf(&cpusetstr,mycpuset);
    INFO_LOG("Process cpuset (OS indexing): %s",cpusetstr);
    ghost_thpool_init(mycpuset);

    ghost_rand_init();

    hwloc_bitmap_free(mycpuset);   
    hwloc_bitmap_free(globcpuset);   
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

    ghost_taskq_finish();
    ghost_thpool_finish();
    ghost_destroyTopology();

    free(ghost_rand_states);
    ghost_rand_states=NULL;

#if GHOST_HAVE_INSTR_LIKWID
    LIKWID_MARKER_CLOSE;
#endif


#if GHOST_HAVE_MPI
    MPI_CALL_RETURN(MPI_Type_free(&GHOST_MPI_DT_C));
    MPI_CALL_RETURN(MPI_Type_free(&GHOST_MPI_DT_Z));
    if (!MPIwasInitialized) {
        MPI_Finalize();
    }
#endif

    return GHOST_SUCCESS;
}
