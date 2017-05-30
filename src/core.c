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
//#include "ghost/tsmm.h"
//#include "ghost/tsmm_inplace.h"
//#include "ghost/tsmttsm.h"
#include "ghost/instr.h"
#include "ghost/autogen.h"

#include <hwloc.h>
#if HWLOC_API_VERSION >= 0x00010700
#include <hwloc/intel-mic.h>
#else
#warning "The HWLOC version is too old. Cannot detect Intel Xeon Phis!"
#endif

#ifdef GHOST_INSTR_LIKWID
#include <likwid.h>
#endif

#ifdef GHOST_HAVE_CUDA
#include <hwloc/cudart.h>
#include <cuda_runtime.h>
#endif

#ifdef GHOST_HAVE_ZOLTAN
#include <zoltan.h>
#endif

#include <strings.h>

static ghost_type mytype = GHOST_TYPE_INVALID;
static int MPIwasInitialized = 0;
static int initialized = 0;

/**
 * @brief A communicator containing only the processes with GHOST_HAVE_CUDA enabled.
 *
 * This is necessary, e.g., for gathering CUDA information in heterogeneous runs containing Xeon Phis.
 */
static ghost_mpi_comm ghost_cuda_comm = MPI_COMM_NULL;

char * ghost_type_string(ghost_type t)
{
    char *ret;
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    switch (t) {
        case GHOST_TYPE_CUDA: 
            ret = "CUDA";
            break;
        case GHOST_TYPE_WORK:
            ret = "WORK";
            break;
        case GHOST_TYPE_INVALID:
            ret = "INVALID";
            break;
        default:
            ret = "Unknown";
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return ret;
}

ghost_error ghost_type_set(ghost_type t)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    
    mytype = t;
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}

ghost_error ghost_type_get(ghost_type *t)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
 
    if (!t) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }

    *t = mytype;
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}

int ghost_initialized()
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return initialized; 
}

#ifdef GHOST_INSTR_LIKWID
static void *likwidThreadInitTask(void *arg)
{
    UNUSED(arg);
#pragma omp parallel
    {
        likwid_markerThreadInit();
        ghost_instr_prefix_set("");
        ghost_instr_suffix_set("");
    }

    return NULL;
}
#endif

ghost_error ghost_init(int argc, char **argv)
{
    if (initialized) {
        return GHOST_SUCCESS;
    } else {
        initialized=1;
    }

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
  
#ifdef GHOST_HAVE_ZOLTAN
    float ver;
    ZOLTAN_CALL_RETURN(Zoltan_Initialize(argc, argv, &ver));
#endif

    ghost_instr_create();
    ghost_instr_prefix_set("");
    ghost_instr_suffix_set("");
    ghost_nodecomm_setup(MPI_COMM_WORLD);
    ghost_mpi_datatypes_create();
    ghost_mpi_operations_create();

#else // ifdef GHOST_HAVE_MPI
    UNUSED(MPIwasInitialized);
    UNUSED(argc);
    UNUSED(argv);

#endif // ifdef GHOST_HAVE_MPI
    
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_SETUP);

    GHOST_CALL_RETURN(ghost_timing_start());
    
    hwloc_topology_t topology;
    ghost_topology_create();
    ghost_topology_get(&topology);


    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    hwloc_get_cpubind(topology,cpuset,HWLOC_CPUBIND_PROCESS);
    if (hwloc_bitmap_weight(cpuset) < hwloc_get_nbobjs_by_type(topology,HWLOC_OBJ_PU)) {
        char *cpusetstr;
        ghost_bitmap_list_asprintf(&cpusetstr,cpuset);
        WARNING_LOG("GHOST is running in a restricted CPU set: %s. This is probably not what you want because GHOST cares for pinning itself. If you want to restrict the resources exposed to GHOST use the GHOST_CPUSET environment variable.",cpusetstr);
        free(cpusetstr);
    }
    hwloc_bitmap_free(cpuset); cpuset = NULL;

    hwloc_cpuset_t mycpuset = hwloc_bitmap_alloc();

    // auto-set rank types 
    ghost_mpi_comm nodeComm;
    int nnoderanks;
    int noderank;
    GHOST_CALL_RETURN(ghost_nodecomm_get(&nodeComm));
    GHOST_CALL_RETURN(ghost_nrank(&nnoderanks, nodeComm));
    GHOST_CALL_RETURN(ghost_rank( &noderank,  nodeComm));

    hwloc_cpuset_t availcpuset = hwloc_bitmap_alloc();

    char *envset = getenv("GHOST_CPUSET");
    if (envset) {
        hwloc_bitmap_list_sscanf(availcpuset,envset);
    } else {
        hwloc_bitmap_copy(availcpuset,hwloc_topology_get_allowed_cpuset(topology));
    }
    IF_DEBUG(2) {
        char *cpusetStr;
        hwloc_bitmap_list_asprintf(&cpusetStr,availcpuset);
        DEBUG_LOG(2,"Available CPU set: %s",cpusetStr);
        free(cpusetStr);
    }

    int nxeonphis_total;
    int ncudadevs = 0;
    int nxeonphis = -1;
    int nnumanodes;
    int npus;
    int ncores;
    int nsockets;

    nsockets = hwloc_get_nbobjs_inside_cpuset_by_type(topology,availcpuset,HWLOC_OBJ_SOCKET);
    nnumanodes = hwloc_get_nbobjs_inside_cpuset_by_type(topology,availcpuset,HWLOC_OBJ_NODE);
    ncores = hwloc_get_nbobjs_inside_cpuset_by_type(topology,availcpuset,HWLOC_OBJ_CORE);
    npus = hwloc_get_nbobjs_inside_cpuset_by_type(topology,availcpuset,HWLOC_OBJ_PU);

    INFO_LOG("# sockets: %d, # NUMA nodes: %d, # cores: %d, # PUs: %d",nsockets,nnumanodes,ncores,npus);

#ifdef GHOST_HAVE_CUDA
    GHOST_CALL_RETURN(ghost_cu_ndevice(&ncudadevs));
#endif


#if HWLOC_API_VERSION >= 0x00010700
    hwloc_obj_t phi = NULL;

    do {
        nxeonphis++;
        phi = hwloc_intel_mic_get_device_osdev_by_index(topology,nxeonphis);
    } while (phi);

    if (noderank == 0) {
        nxeonphis_total = nxeonphis;
    } else {
        nxeonphis_total = 0;
    }

#ifdef GHOST_HAVE_MPI    
    MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE,&nxeonphis_total,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD));
#endif

#else
    WARNING_LOG("Possibly wrong information about the number of Xeon Phis due to outdated HWLOC!");
    nxeonphis_total = 0;
#endif

    int nactivephis = 0;
#ifdef GHOST_BUILD_MIC
    nactivephis = 1;
#endif

#ifdef GHOST_HAVE_MPI
    MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE,&nactivephis,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD));
#endif

    if (nactivephis < nxeonphis_total) {
        PERFWARNING_LOG("There %s %d Xeon Phi%s in the set of active nodes but only %d %s used!",
                nxeonphis_total>1?"are":"is",nxeonphis_total,nxeonphis_total>1?"s":"",nactivephis,nactivephis==1?"is":"are");
    }

    if (nnoderanks != nnumanodes+ncudadevs) {
        PERFWARNING_LOG("The number of MPI processes (%d) on this node is not "
                "optimal! Suggested number: %d (%d NUMA domain%s + %d CUDA device%s)",
                nnoderanks,nnumanodes+ncudadevs,nnumanodes,nnumanodes==1?"":"s",ncudadevs,ncudadevs==1?"":"s");
    }

    // get GHOST type set by the user
    ghost_type settype;
    GHOST_CALL_RETURN(ghost_type_get(&settype));
    
    if (settype == GHOST_TYPE_INVALID) {
        char *envtype = getenv("GHOST_TYPE");
        if (envtype) {
            if (!strncasecmp(envtype,"CUDA",4) || !strncasecmp(envtype,"GPU",3)) {
                mytype = GHOST_TYPE_CUDA;
            } else if (!strncasecmp(envtype,"WORK",4) || !strncasecmp(envtype,"CPU",3)) {
                mytype = GHOST_TYPE_WORK;
            }
        }
    }

    // type has been set by neither env nor API
    if (settype == GHOST_TYPE_INVALID && mytype == GHOST_TYPE_INVALID) {
        if (noderank == 0) {
            mytype = GHOST_TYPE_WORK;
        } else if (noderank <= ncudadevs) {
            mytype = GHOST_TYPE_CUDA;
        } else {
            mytype = GHOST_TYPE_WORK;
        }
        if (ncudadevs && nnoderanks > 1) {
            INFO_LOG("Setting GHOST type to %s due to heuristics.",ghost_type_string(mytype));
        }
    } 

#ifndef GHOST_HAVE_CUDA
    if (mytype == GHOST_TYPE_CUDA) {
        WARNING_LOG("This rank is supposed to be a CUDA management rank but CUDA is not available. Re-setting GHOST type");
        mytype = GHOST_TYPE_WORK;
    }
#endif
    
    GHOST_CALL_RETURN(ghost_type_set(mytype));

    int i;
    int localTypes[nnoderanks];

    for (i=0; i<nnoderanks; i++) {
        localTypes[i] = GHOST_TYPE_INVALID;
    }
    localTypes[noderank] = mytype;
    
    int ncudaranks_on_node = mytype==GHOST_TYPE_CUDA;
#ifdef GHOST_HAVE_MPI
    ghost_mpi_comm ghost_node_comm;
    GHOST_CALL_RETURN(ghost_nodecomm_get(&ghost_node_comm));
    MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE,&ncudaranks_on_node,1,MPI_INT,MPI_SUM,ghost_node_comm));

#ifdef GHOST_HAVE_CUDA
    if (ncudadevs < ncudaranks_on_node) {
        WARNING_LOG("There are %d CUDA management ranks on this node but only %d CUDA devices.",ncudaranks_on_node,ncudadevs);
    }
#endif


    MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE,&localTypes,nnoderanks,MPI_INT,MPI_MAX,ghost_node_comm));
#endif   
    
    ghost_hwconfig hwconfig;
    ghost_hwconfig_get(&hwconfig);

    int hasCuda = 0;
    hwloc_cpuset_t cudaOccupiedCpuset = hwloc_bitmap_alloc();
#ifdef GHOST_HAVE_CUDA
    hasCuda = 1;
    int cudaDevice = 0;

    if (hwconfig.cudevice != GHOST_HWCONFIG_INVALID) {
        GHOST_CALL_RETURN(ghost_cu_init(hwconfig.cudevice));
    } else { // automatically assign a CUDA device
        for (i=0; i<nnoderanks; i++) {
            if (localTypes[i] == GHOST_TYPE_CUDA) {
                if (i == noderank) {
                    hwconfig.cudevice = cudaDevice%ncudadevs;
                    GHOST_CALL_RETURN(ghost_cu_init(hwconfig.cudevice));
                }
                cudaDevice++;
            }
        }
    }
    GHOST_CALL_RETURN(ghost_hwconfig_set(hwconfig));

    // CUDA ranks have a physical core
    cudaDevice = 0;
    hwloc_obj_t cudaCore = NULL;
    for (i=0; i<nnoderanks; i++) {
        if (localTypes[i] == GHOST_TYPE_CUDA) {
            hwloc_cpuset_t fullCuCpuset = hwloc_bitmap_alloc();
            hwloc_cpuset_t reducedCuCpuset;
            
            HWLOC_CALL_RETURN(hwloc_cudart_get_device_cpuset(topology,cudaDevice,fullCuCpuset));
            
            // restrict CUDA cpuset to CPUs which are still in global cpuset
            hwloc_bitmap_and(fullCuCpuset,fullCuCpuset,availcpuset);
            
            if (hwloc_bitmap_iszero(fullCuCpuset)) {
                PERFWARNING_LOG("Placing CUDA process on far socket!");
                hwloc_bitmap_copy(fullCuCpuset,availcpuset);
            }

            if (nnoderanks > 1) {
                // select a single core for this CUDA rank 
                cudaCore = hwloc_get_next_obj_inside_cpuset_by_type(topology,fullCuCpuset,HWLOC_OBJ_CORE,cudaCore);
                reducedCuCpuset = cudaCore->cpuset;
            } else {
                reducedCuCpuset = fullCuCpuset;
            }
        
            if (noderank == i) {
                hwloc_bitmap_copy(mycpuset,reducedCuCpuset);
            }
            hwloc_bitmap_or(cudaOccupiedCpuset,cudaOccupiedCpuset,reducedCuCpuset);

            hwloc_bitmap_free(fullCuCpuset);
            
            cudaDevice++;
        }
    }
#endif
        
    int ncpuranks_on_node = nnoderanks-ncudaranks_on_node;
    
    if (ncpuranks_on_node > 1) {   
        // indicate whether the CPU ranks cover a full hwloc obj
        bool ranks_cover_obj = true; 
        hwloc_obj_type_t distr_type;
        if (nsockets == ncpuranks_on_node) {
            INFO_LOG("One process per socket");
            distr_type = HWLOC_OBJ_SOCKET;
        } else if (nnumanodes == ncpuranks_on_node) {
            INFO_LOG("One process per NUMA node");
            distr_type = HWLOC_OBJ_NODE;
        } else if (ncores == ncpuranks_on_node) {
            INFO_LOG("One process per core");
            distr_type = HWLOC_OBJ_CORE;
        } else if (npus == ncpuranks_on_node) {
            INFO_LOG("One process per PU");
            distr_type = HWLOC_OBJ_PU;
        } else if (npus < ncpuranks_on_node) {
            distr_type = HWLOC_OBJ_PU;
            PERFWARNING_LOG("Oversubscription! Some processes will share PUs!");
            ranks_cover_obj = false;
        } else {
            PERFWARNING_LOG("Naively sharing %d PUs among %d ranks",npus,ncpuranks_on_node);
            ranks_cover_obj = false;
        }

        hwloc_obj_t coverobj = NULL;
        int cpurank = 0;
        hwloc_bitmap_t rank_cpuset = hwloc_bitmap_alloc();

        // we need a copy because we delete PUs from availcpuset as we go through the processes
        hwloc_cpuset_t fullavailcpuset = hwloc_bitmap_dup(availcpuset);
        
        for (i=0; i<nnoderanks; i++) {
            hwloc_bitmap_zero(rank_cpuset);
            if (localTypes[i] == GHOST_TYPE_WORK) {

                if (ranks_cover_obj) {
                    // the obj covered by this rank
                    coverobj = hwloc_get_obj_inside_cpuset_by_type(topology, fullavailcpuset, distr_type,cpurank);
                    hwloc_bitmap_copy(rank_cpuset,coverobj->cpuset);
                } else {
                    hwloc_obj_type_t dist_obj;
                    int obj_per_rank;
                    int nobj;
                    if (ncpuranks_on_node <= ncores) {
                        PERFWARNING_LOG("Distributing cores among processes");
                        dist_obj = HWLOC_OBJ_CORE;
                        obj_per_rank = ncores/ncpuranks_on_node;
                        nobj = ncores;
                    } else {
                        dist_obj = HWLOC_OBJ_PU;
                        nobj = npus;
                        if (ncpuranks_on_node <= npus) {
                            PERFWARNING_LOG("Distributing PUs among processes");
                            obj_per_rank = npus/ncpuranks_on_node;
                        } else {
                            PERFWARNING_LOG("More processes than PUs!");
                            obj_per_rank = 1;
                        }
                    }
                        
                    if (i == noderank) {
                        int r,oi;

                        
                        // assign cores
                        r = MIN(cpurank*obj_per_rank,(nobj-1));
                        for (oi=0; oi < obj_per_rank; oi++) {
                            hwloc_bitmap_or(rank_cpuset,rank_cpuset,hwloc_get_obj_inside_cpuset_by_type(topology,fullavailcpuset,dist_obj,r)->cpuset);
                            if (r<(nobj-1)) {
                                r++;
                            }
                        }


                        // remainder
                        if (cpurank == ncpuranks_on_node-1) {
                            for (; r<nobj; r++) {
                                hwloc_bitmap_or(rank_cpuset,rank_cpuset,hwloc_get_obj_inside_cpuset_by_type(topology,fullavailcpuset,dist_obj,r)->cpuset);
                            }
                        }

                    }
                    cpurank++;
                }

                // set mycpuset
                if (i == noderank) {
                    hwloc_bitmap_copy(mycpuset,rank_cpuset);
                }

                // delete my PUs from available CPU set
                hwloc_bitmap_andnot(availcpuset,availcpuset,rank_cpuset);
              
                if (ranks_cover_obj) { 
                    // only go to next obj if no oversubscription 
                    if (cpurank < hwloc_get_nbobjs_inside_cpuset_by_type(topology, fullavailcpuset, distr_type)-1) {
                        cpurank++; 
                    }
                }
            }
        }

        hwloc_bitmap_free(rank_cpuset);
        hwloc_bitmap_free(fullavailcpuset);
       
    } else {
        INFO_LOG("One process per node");
        if (mytype == GHOST_TYPE_WORK) {
            hwloc_bitmap_copy(mycpuset,availcpuset);
        }
    }    

    if (mytype == GHOST_TYPE_WORK) {
    // exclude CUDA cores from CPU set
        hwloc_bitmap_andnot(mycpuset,mycpuset,cudaOccupiedCpuset);
    }

    if (hwloc_bitmap_iszero(mycpuset)) {
        WARNING_LOG("Something went wrong and I ended up with an empty CPU set! I will use all CPUs instead which will probably lead to resource conflicts!");
        hwloc_bitmap_copy(mycpuset,availcpuset);
    }
    


    if (hwconfig.ncore == GHOST_HWCONFIG_INVALID) {
        ghost_machine_ncore(&hwconfig.ncore,GHOST_NUMANODE_ANY);
    }
    if (hwconfig.nsmt == GHOST_HWCONFIG_INVALID) {
        ghost_machine_nsmt(&hwconfig.nsmt);
    }



#ifdef GHOST_HAVE_MPI
    int rank;
    ghost_mpi_comm tmpcomm;
    GHOST_CALL_RETURN(ghost_rank(&rank,MPI_COMM_WORLD));
    MPI_CALL_RETURN(MPI_Comm_dup(MPI_COMM_WORLD,&tmpcomm));
    MPI_CALL_RETURN(MPI_Comm_split(tmpcomm,hasCuda,rank,&ghost_cuda_comm));
    MPI_CALL_RETURN(MPI_Comm_split(tmpcomm,hasCuda,rank,&ghost_cuda_comm));
    MPI_CALL_RETURN(MPI_Comm_free(&tmpcomm));
#else
    UNUSED(hasCuda);
#endif

    // delete excess PUs
    unsigned int firstcpu = hwloc_get_pu_obj_by_os_index(topology,hwloc_bitmap_first(mycpuset))->parent->logical_index;
    unsigned int cpu;
    hwloc_bitmap_foreach_begin(cpu,mycpuset);
        hwloc_obj_t obj = hwloc_get_pu_obj_by_os_index(topology,cpu);

        if (obj->parent->logical_index-firstcpu >= (unsigned)hwconfig.ncore) {
            hwloc_bitmap_clr(mycpuset,obj->os_index);
            if (hwloc_bitmap_iszero(mycpuset)) {
                WARNING_LOG("Ignoring hwconfig setting as it would zero the CPU set!");
                hwloc_bitmap_set(mycpuset,obj->os_index);
            }
        }
        if ((int)(obj->sibling_rank) >= hwconfig.nsmt) {
            hwloc_bitmap_clr(mycpuset,obj->os_index);
            if (hwloc_bitmap_iszero(mycpuset)) {
                WARNING_LOG("Ignoring hwconfig setting as it would zero the CPU set!");
                hwloc_bitmap_set(mycpuset,obj->os_index);
            }
        } 
    hwloc_bitmap_foreach_end();

    ghost_pumap_create(mycpuset);

    ghost_rand_create();
    
#ifdef GHOST_INSTR_LIKWID
    likwid_markerInit();

    if (ghost_tasking_enabled()) {
        ghost_task *t;
        ghost_task_create(&t,GHOST_TASK_FILL_ALL,0,&likwidThreadInitTask,NULL,GHOST_TASK_DEFAULT, NULL, 0);
        ghost_task_enqueue(t);
        ghost_task_wait(t);
        ghost_task_destroy(t);
    } else {
#pragma omp parallel
        likwid_markerThreadInit();
    }
#endif
   

    hwloc_bitmap_free(cudaOccupiedCpuset);
    hwloc_bitmap_free(mycpuset); mycpuset = NULL; 
    hwloc_bitmap_free(availcpuset); availcpuset = NULL;

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_SETUP);
    return GHOST_SUCCESS;
}

ghost_error ghost_finalize()
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_TEARDOWN);
    static int finalized = 0;

    if (finalized) {
        return GHOST_SUCCESS;
    } else {
        finalized = 1;
    }

    if (ghost_autogen_missing()) {
        PERFWARNING_LOG("Found missing autogenerated kernels! Please re-configure GHOST using:\ncmake . %s",ghost_autogen_string());
    }

    ghost_rand_destroy();

#ifdef GHOST_INSTR_LIKWID
    likwid_markerClose();
#endif
    
#ifdef GHOST_INSTR_TIMING
#if GHOST_VERBOSITY
//    char *str;
//    ghost_timing_summarystring(&str);
//    INFO_LOG("\n%s",str);
//    free(str);
#endif
#endif

    ghost_mpi_datatypes_destroy();
    ghost_mpi_operations_destroy();

    ghost_taskq_waitall();
    ghost_taskq_destroy();
    ghost_thpool_destroy();
    ghost_pumap_destroy();
    ghost_topology_destroy();

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_TEARDOWN);

    ghost_timing_destroy();    
    ghost_instr_destroy();
#ifdef GHOST_HAVE_MPI
    if (!MPIwasInitialized) {
        MPI_Finalize();
    }
#endif
    
    // needs to be done _after_ MPI_Finalize() (for GPUdirect)
    ghost_cu_finalize();

    return GHOST_SUCCESS;
}

ghost_error ghost_string(char **str) 
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    GHOST_CALL_RETURN(ghost_malloc((void **)str,1));
    memset(*str,'\0',1);

    ghost_header_string(str,"%s", GHOST_NAME); 
    ghost_line_string(str,"Version",NULL,"%s",GHOST_VERSION);
    ghost_line_string(str,"Build date",NULL,"%s",__DATE__);
    ghost_line_string(str,"Build time",NULL,"%s",__TIME__);
#ifdef GHOST_BUILD_MIC
    ghost_line_string(str,"MIC kernels",NULL,"Enabled");
#else
    ghost_line_string(str,"MIC kernels",NULL,"Disabled");
#endif
#ifdef GHOST_BUILD_AVX
    ghost_line_string(str,"AVX kernels",NULL,"Enabled");
#else
    ghost_line_string(str,"AVX kernels",NULL,"Disabled");
#endif
#ifdef GHOST_BUILD_SSE
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
#ifdef GHOST_INSTR_LIKWID
#ifdef GHOST_INSTR_TIMING
    ghost_line_string(str,"Instrumentation",NULL,"Likwid+Timing");
#else
    ghost_line_string(str,"Instrumentation",NULL,"Likwid");
#endif
#else
#ifdef GHOST_INSTR_TIMING
    ghost_line_string(str,"Instrumentation",NULL,"Timing");
#else
    ghost_line_string(str,"Instrumentation",NULL,"Disabled");
#endif
#endif
#ifdef GHOST_IDX64_GLOBAL
    ghost_line_string(str,"Global index size","bits","64");
#else
    ghost_line_string(str,"Global index size","bits","32");
#endif
#ifdef GHOST_IDX64_LOCAL
    ghost_line_string(str,"Local index size","bits","64");
#else
    ghost_line_string(str,"Local index size","bits","32");
#endif
    ghost_footer_string(str);

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;

}

ghost_error ghost_barrier()
{
#ifdef GHOST_HAVE_MPI
    MPI_CALL_RETURN(MPI_Barrier(MPI_COMM_WORLD));
#endif
#ifdef GHOST_HAVE_CUDA
    GHOST_CALL_RETURN(ghost_cu_barrier());
#endif

    return GHOST_SUCCESS;
}
    
ghost_error ghost_cuda_comm_get(ghost_mpi_comm *comm)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    *comm = ghost_cuda_comm;
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}
