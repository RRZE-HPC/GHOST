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
#include "ghost/rand.h"

#include <hwloc.h>
#ifdef GHOST_HAVE_INSTR_LIKWID
#include <likwid.h>
#endif

#include <strings.h>

static ghost_type_t ghost_type = GHOST_TYPE_INVALID;
static int MPIwasInitialized = 0;

char * ghost_type_string(ghost_type_t t)
{

    switch (t) {
        case GHOST_TYPE_CUDA: 
            return "CUDA";
            break;
        case GHOST_TYPE_WORK:
            return "WORK";
            break;
        case GHOST_TYPE_INVALID:
            return "INVALID";
            break;
        default:
            return "Unknown";
    }
}

ghost_error_t ghost_type_set(ghost_type_t t)
{
    ghost_type = t;

    return GHOST_SUCCESS;
}

ghost_error_t ghost_type_get(ghost_type_t *t)
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

    ghost_nodecomm_setup(MPI_COMM_WORLD);
    ghost_mpi_datatypes_create();
    ghost_mpi_operations_create();

#else // ifdef GHOST_HAVE_MPI
    UNUSED(MPIwasInitialized);
    UNUSED(argc);
    UNUSED(argv);

#endif // ifdef GHOST_HAVE_MPI
    
    hwloc_topology_t topology;
    ghost_topology_create();
    ghost_topology_get(&topology);

#ifdef GHOST_HAVE_INSTR_LIKWID
    LIKWID_MARKER_INIT;

#pragma omp parallel
    LIKWID_MARKER_THREADINIT;
#endif


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
    GHOST_CALL_RETURN(ghost_nodecomm_get(&nodeComm));
    GHOST_CALL_RETURN(ghost_nrank(&nnoderanks, nodeComm));
    GHOST_CALL_RETURN(ghost_rank( &noderank,  nodeComm));

    int ncudadevs = 0;
    int nnumanodes;
    ghost_machine_nnuma(&nnumanodes);

#ifdef GHOST_HAVE_CUDA
    ghost_cu_ndevice(&ncudadevs);
#endif

    ghost_type_t ghost_type;
    GHOST_CALL_RETURN(ghost_type_get(&ghost_type));
    
    if (ghost_type == GHOST_TYPE_INVALID) {
        char *envtype = getenv("GHOST_TYPE");
        if (envtype) {
            if (!strncasecmp(envtype,"CUDA",4)) {
                ghost_type = GHOST_TYPE_CUDA;
            } else if (!strncasecmp(envtype,"WORK",4)) {
                INFO_LOG("Setting GHOST type to WORK due to environment variable.");
                ghost_type = GHOST_TYPE_WORK;
            }
            INFO_LOG("Setting GHOST type to %s due to environment variable.",ghost_type_string(ghost_type));
        }
    }

    if (ghost_type == GHOST_TYPE_INVALID) {
        if (noderank == 0) {
            ghost_type = GHOST_TYPE_WORK;
        } else if (noderank <= ncudadevs) {
            ghost_type = GHOST_TYPE_CUDA;
        } else {
            ghost_type = GHOST_TYPE_WORK;
        }
        INFO_LOG("Setting GHOST type to %s due to heuristics.",ghost_type_string(ghost_type));
    } 

#ifndef GHOST_HAVE_CUDA
    if (ghost_type == GHOST_TYPE_CUDA) {
        WARNING_LOG("This rank is supposed to be a CUDA management rank but CUDA is not available. Re-setting GHOST type");
        ghost_type = GHOST_TYPE_WORK;
    }
#endif
    
    GHOST_CALL_RETURN(ghost_type_set(ghost_type));


    int nLocalCuda = ghost_type==GHOST_TYPE_CUDA;

    int i;
    int localTypes[nnoderanks];

    for (i=0; i<nnoderanks; i++) {
        localTypes[i] = GHOST_TYPE_INVALID;
    }
    localTypes[noderank] = ghost_type;
#ifdef GHOST_HAVE_MPI
    ghost_mpi_comm_t ghost_node_comm;
    GHOST_CALL_RETURN(ghost_nodecomm_get(&ghost_node_comm));
    MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE,&nLocalCuda,1,MPI_INT,MPI_SUM,ghost_node_comm));

#ifdef GHOST_HAVE_CUDA
    if (ncudadevs < nLocalCuda) {
        WARNING_LOG("There are %d CUDA management ranks on this node but only %d CUDA devices.",nLocalCuda,ncudadevs);
    }
#endif


    MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE,&localTypes,nnoderanks,MPI_INT,MPI_MAX,ghost_node_comm));
#endif   

    ghost_hybridmode_t ghost_hybridmode;
    GHOST_CALL_RETURN(ghost_hybridmode_get(&ghost_hybridmode));

    int oversubscribed = 0;
    if (ghost_hybridmode == GHOST_HYBRIDMODE_INVALID) {
        if (nnoderanks <=  nLocalCuda+1) {
            ghost_hybridmode = GHOST_HYBRIDMODE_ONEPERNODE;
        } else if (nnoderanks == nLocalCuda+nnumanodes) {
            ghost_hybridmode = GHOST_HYBRIDMODE_ONEPERNUMA;
        } else if (nnoderanks == hwloc_get_nbobjs_by_type(topology,HWLOC_OBJ_CORE)) {
            ghost_hybridmode = GHOST_HYBRIDMODE_ONEPERCORE;
        } else {
            ghost_hybridmode = GHOST_HYBRIDMODE_CUSTOM;
        }
    }
    GHOST_CALL_RETURN(ghost_hybridmode_set(ghost_hybridmode));

    int maxcore;
    ghost_machine_ncore(&maxcore, GHOST_NUMANODE_ANY);
    
    int maxpu;
    ghost_machine_npu(&maxpu, GHOST_NUMANODE_ANY);

    hwloc_cpuset_t mycpuset = hwloc_bitmap_alloc();
    hwloc_cpuset_t globcpuset = hwloc_bitmap_alloc();

    hwloc_bitmap_copy(globcpuset,hwloc_topology_get_allowed_cpuset(topology));
    ghost_hwconfig_t hwconfig;
    ghost_hwconfig_get(&hwconfig);

    if (hwconfig.ncore == GHOST_HWCONFIG_INVALID) {
        ghost_machine_ncore(&hwconfig.ncore, GHOST_NUMANODE_ANY);
    }
    if (hwconfig.nsmt == GHOST_HWCONFIG_INVALID) {
        ghost_machine_nsmt(&hwconfig.nsmt);
    }
    ghost_hwconfig_set(hwconfig);

    IF_DEBUG(2) {
        char *cpusetStr;
        hwloc_bitmap_list_asprintf(&cpusetStr,globcpuset);
        DEBUG_LOG(2,"Available CPU set: %s",cpusetStr);
        free(cpusetStr);
    }

#ifdef GHOST_HAVE_CUDA
    int cudaDevice = 0;

    for (i=0; i<nnoderanks; i++) {
        if (localTypes[i] == GHOST_TYPE_CUDA) {
            if (i == noderank) {
                ghost_cu_init(cudaDevice%ncudadevs);
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
    } else if (ghost_hybridmode == GHOST_HYBRIDMODE_ONEPERCORE) {
        if (nnoderanks > maxcore) {
            oversubscribed = 1;
            WARNING_LOG("More processes (%d) than cores available (%d)",nnoderanks,maxcore);
        } else {
            for (i=0; i<nnoderanks; i++) {
                if (localTypes[i] == GHOST_TYPE_WORK) {
                    hwloc_cpuset_t coreCpuset;
                    coreCpuset = hwloc_get_obj_by_type(topology,HWLOC_OBJ_CORE,i)->cpuset;
                    if (i == noderank) {
                        hwloc_bitmap_and(mycpuset,globcpuset,coreCpuset);
                    }
                    hwloc_bitmap_andnot(globcpuset,globcpuset,coreCpuset);
                }
            }
        }
    } else if (ghost_hybridmode == GHOST_HYBRIDMODE_CUSTOM) {
        if (nnoderanks > maxpu) {
            oversubscribed = 1;
            WARNING_LOG("More processes (%d) than PUs available (%d)",nnoderanks,maxpu);
        } else {
            int pusperrank = maxpu/nnoderanks;

            for (i=0; i<nnoderanks-1; i++) { // the last rank will get the remaining PUs
                if (localTypes[i] == GHOST_TYPE_WORK) {
                    hwloc_cpuset_t puCpuset;
                    int pu;
                    for (pu=0; pu<pusperrank; pu++) {
                        puCpuset = hwloc_get_obj_by_type(topology,HWLOC_OBJ_PU,i*pusperrank+pu)->cpuset;
                        if (i == noderank) {
                            hwloc_bitmap_t bak = hwloc_bitmap_dup(mycpuset);
                            hwloc_bitmap_and(mycpuset,globcpuset,puCpuset);
                            hwloc_bitmap_or(mycpuset,mycpuset,bak);
                            hwloc_bitmap_free(bak);
                        }
                        hwloc_bitmap_andnot(globcpuset,globcpuset,puCpuset);
                    }
                }
            }
            if (localTypes[i] == GHOST_TYPE_WORK && i == noderank) {
                hwloc_bitmap_copy(mycpuset,globcpuset);
            }
            hwloc_bitmap_andnot(globcpuset,globcpuset,globcpuset);
        }

    }

    if (oversubscribed) {
        mycpuset = hwloc_bitmap_dup(hwloc_get_obj_by_depth(topology,HWLOC_OBJ_SYSTEM,0)->cpuset);
    }

    // delete PUs from cpuset according to hwconfig
    hwloc_obj_t obj;
    unsigned int cpu;
    hwloc_bitmap_t backup = hwloc_bitmap_dup(mycpuset);

    // delete excess SMT threads
    hwloc_bitmap_foreach_begin(cpu,backup);
    obj = hwloc_get_pu_obj_by_os_index(topology,cpu);

    if ((int)(obj->sibling_rank) >= hwconfig.nsmt) {
        hwloc_bitmap_clr(mycpuset,obj->os_index);
    } 
    hwloc_bitmap_foreach_end();

    // delete excess cores
    hwloc_obj_t core_to_delete = hwloc_get_obj_inside_cpuset_by_type(topology,mycpuset,HWLOC_OBJ_CORE,hwconfig.ncore);
    while (core_to_delete) {
        hwloc_bitmap_andnot(mycpuset,mycpuset,core_to_delete->cpuset);
        core_to_delete = hwloc_get_next_obj_inside_cpuset_by_type(topology,mycpuset,HWLOC_OBJ_CORE,core_to_delete);
    }

    void *(*threadFunc)(void *);

    ghost_taskq_create();
    ghost_taskq_startroutine(&threadFunc);
    ghost_thpool_create(hwloc_bitmap_weight(mycpuset)+1,threadFunc);
    ghost_pumap_create(mycpuset);

    ghost_rand_create();
    hwloc_bitmap_free(mycpuset); mycpuset = NULL; 
    hwloc_bitmap_free(globcpuset); globcpuset = NULL;

    return GHOST_SUCCESS;
}

ghost_error_t ghost_finalize()
{


    ghost_rand_destroy();

#ifdef GHOST_HAVE_INSTR_LIKWID
    LIKWID_MARKER_CLOSE;
#endif

    ghost_mpi_datatypes_destroy();
    ghost_mpi_operations_destroy();

    ghost_taskq_waitall();
    ghost_taskq_destroy();
    ghost_thpool_destroy();
    ghost_pumap_destroy();
    ghost_topology_destroy();

#ifdef GHOST_HAVE_MPI
    if (!MPIwasInitialized) {
        MPI_Finalize();
    }
#endif

    return GHOST_SUCCESS;
}

ghost_error_t ghost_string(char **str) 
{
    GHOST_CALL_RETURN(ghost_malloc((void **)str,1));
    memset(*str,'\0',1);

    ghost_header_string(str,"%s", GHOST_NAME); 
    ghost_line_string(str,"Version",NULL,"%s",GHOST_VERSION);
    ghost_line_string(str,"Build date",NULL,"%s",__DATE__);
    ghost_line_string(str,"Build time",NULL,"%s",__TIME__);
#ifdef GHOST_HAVE_MIC
    ghost_line_string(str,"MIC kernels",NULL,"Enabled");
#else
    ghost_line_string(str,"MIC kernels",NULL,"Disabled");
#endif
#ifdef GHOST_HAVE_AVX
    ghost_line_string(str,"AVX kernels",NULL,"Enabled");
#else
    ghost_line_string(str,"AVX kernels",NULL,"Disabled");
#endif
#ifdef GHOST_HAVE_SSE
    ghost_line_string(str,"SSE kernels",NULL,"Enabled");
#else
    ghost_line_string(str,"SSE kernels",NULL,"Disabled");
#endif
#ifdef GHOST_HAVE_OPENMP
    ghost_line_string(str,"OpenMP support",NULL,"Enabled");
#else
    ghost_line_string(str,"OpenMP support",NULL,"Disabled");
#endif
#ifdef GHOST_HAVE_MPI
    ghost_line_string(str,"MPI support",NULL,"Enabled");
#else
    ghost_line_string(str,"MPI support",NULL,"Disabled");
#endif
#ifdef GHOST_HAVE_CUDA
    ghost_line_string(str,"CUDA support",NULL,"Enabled");
#else
    ghost_line_string(str,"CUDA support",NULL,"Disabled");
#endif
#ifdef GHOST_HAVE_INSTR_LIKWID
    ghost_line_string(str,"Instrumentation",NULL,"Likwid");
#elif defined(GHOST_HAVE_INSTR_TIMING)
    ghost_line_string(str,"Instrumentation",NULL,"Timing");
#else
    ghost_line_string(str,"Instrumentation",NULL,"Disabled");
#endif
    ghost_footer_string(str);

    return GHOST_SUCCESS;

}

