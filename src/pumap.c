#include "ghost/pumap.h"
#include "ghost/machine.h"
#include "ghost/util.h"
#include "ghost/locality.h"

static ghost_pumap_t *pumap = NULL;
pthread_mutex_t ghost_pumap_mutex;

ghost_error_t ghost_pumap_npu(int *nPUs, int numanode)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_TASKING);

    if (!nPUs) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }

    if (numanode == GHOST_NUMANODE_ANY) {
        *nPUs = hwloc_bitmap_weight(pumap->cpuset);
    } else {
        hwloc_obj_t node;
        ghost_machine_numanode(&node,numanode);
        hwloc_bitmap_t LDBM = node->cpuset;

        hwloc_bitmap_t contained = hwloc_bitmap_alloc();
        hwloc_bitmap_and(contained,LDBM,pumap->cpuset);

        *nPUs = hwloc_bitmap_weight(contained);

        hwloc_bitmap_free(contained);
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_TASKING);
    return GHOST_SUCCESS;
}

ghost_error_t ghost_pumap_nidle(int *nPUs, int numanode)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_TASKING);

    if (numanode == GHOST_NUMANODE_ANY) {
        *nPUs = hwloc_bitmap_weight(pumap->cpuset)-hwloc_bitmap_weight(pumap->busy);
    } else {
        hwloc_obj_t node;
        ghost_machine_numanode(&node,numanode);
        hwloc_bitmap_t LDBM = node->cpuset;

        hwloc_bitmap_t idle = hwloc_bitmap_alloc();
        hwloc_bitmap_andnot(idle,LDBM,pumap->busy);

        int w = hwloc_bitmap_weight(idle);

        hwloc_bitmap_free(idle);
        *nPUs = w;
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_TASKING);
    return GHOST_SUCCESS;
}

ghost_error_t ghost_pumap_setidle(hwloc_bitmap_t cpuset)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_TASKING);
    
    if (!hwloc_bitmap_isincluded(cpuset,pumap->cpuset)) {
        ERROR_LOG("The given CPU set is not included in the PU map's CPU set");
        return GHOST_ERR_INVALID_ARG;
    }
    
    hwloc_bitmap_andnot(pumap->busy,pumap->busy,cpuset);

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_TASKING);
    return GHOST_SUCCESS;
}

ghost_error_t ghost_pumap_setidle_idx(int idx)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_TASKING);
    
    if (!hwloc_bitmap_isset(pumap->cpuset,idx)) {
        ERROR_LOG("The given index %d is not included in the PU map's CPU set",idx);
        return GHOST_ERR_INVALID_ARG;
    }

    hwloc_bitmap_clr(pumap->busy,idx);

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_TASKING);
    return GHOST_SUCCESS;
}

ghost_error_t ghost_pumap_setbusy(hwloc_bitmap_t cpuset)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_TASKING);
    
    if (!hwloc_bitmap_isincluded(cpuset,pumap->cpuset)) {
        ERROR_LOG("The given CPU set is not included in the PU map's CPU set");
        return GHOST_ERR_INVALID_ARG;
    }

    hwloc_bitmap_or(pumap->busy,pumap->busy,cpuset);
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_TASKING);
    return GHOST_SUCCESS;
}

ghost_error_t ghost_pumap_get(ghost_pumap_t **map)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_TASKING);
    

    if (!map) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }

    *map = pumap;

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_TASKING);
    return GHOST_SUCCESS;
}

ghost_error_t ghost_pumap_create(hwloc_cpuset_t cpuset)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_TASKING|GHOST_FUNCTYPE_SETUP);
    
    if (!cpuset) {
        ERROR_LOG("CPU set is NULL");
        return GHOST_ERR_INVALID_ARG;
    }
    ghost_error_t ret = GHOST_SUCCESS;
    pthread_mutex_init(&ghost_pumap_mutex,NULL);
    
    pumap = NULL;
    hwloc_nodeset_t nodeset = NULL;
    hwloc_obj_t numanode = NULL;
    hwloc_cpuset_t remoteCPUset = NULL;
    hwloc_cpuset_t localCPUset = NULL;

    int *domains = NULL;
    int q,t;
    int localthreads;

    int totalThreads = hwloc_bitmap_weight(cpuset);
    hwloc_topology_t topology;
    GHOST_CALL_GOTO(ghost_topology_get(&topology),err,ret);

    GHOST_CALL_GOTO(ghost_malloc((void **)&pumap,sizeof(ghost_pumap_t)),err,ret);
    pumap->cpuset = hwloc_bitmap_dup(cpuset);
    pumap->busy = hwloc_bitmap_alloc();

    if (!pumap->cpuset || !pumap->busy) {
        ERROR_LOG("Could not allocate CPU set");
        ret = GHOST_ERR_HWLOC;
        goto err;
    }

    // get number of NUMA nodes
    nodeset = hwloc_bitmap_alloc();
    if (!nodeset) {
        ERROR_LOG("Could not allocate node set");
        ret = GHOST_ERR_HWLOC;
        goto err;
    }
    hwloc_cpuset_to_nodeset(topology,cpuset,nodeset);
    if (hwloc_bitmap_isfull(nodeset)) { // non-NUMA case: we assume a single NUMA node
        hwloc_bitmap_only(nodeset,0);
    }
    pumap->nDomains = hwloc_bitmap_weight(nodeset);
    GHOST_CALL_GOTO(ghost_malloc((void **)&pumap->PUs,pumap->nDomains*sizeof(hwloc_obj_t *)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&domains,pumap->nDomains*sizeof(int)),err,ret);
    unsigned int domainIdx;
    q = 0;
    hwloc_bitmap_foreach_begin(domainIdx,nodeset);
    domains[q] = domainIdx;
    q++;
    hwloc_bitmap_foreach_end();

    remoteCPUset = hwloc_bitmap_alloc();
    localCPUset = hwloc_bitmap_alloc();
    // sort the cores according to the locality domain
    for (q=0; q<pumap->nDomains; q++) {

        GHOST_CALL_GOTO(ghost_malloc((void **)&(pumap->PUs[q]),sizeof(hwloc_obj_t)*totalThreads),err,ret);
        GHOST_CALL_GOTO(ghost_machine_numanode(&numanode,domains[q]),err,ret); 
        
        hwloc_bitmap_and(localCPUset,pumap->cpuset,numanode->cpuset);
        localthreads = hwloc_bitmap_weight(localCPUset); 
        hwloc_bitmap_andnot(remoteCPUset,pumap->cpuset,numanode->cpuset);

        for (t=0; t<localthreads; t++) { // my own threads
            pumap->PUs[q][t] = hwloc_get_obj_inside_cpuset_by_type(topology,localCPUset,HWLOC_OBJ_PU,t);
        }
        for (t=0; t<totalThreads-localthreads; t++) { // other NUMA nodes
            pumap->PUs[q][localthreads+t] = hwloc_get_obj_inside_cpuset_by_type(topology,remoteCPUset,HWLOC_OBJ_PU,t);
        }    
    }
    IF_DEBUG(2) {
        for (q=0; q<pumap->nDomains; q++) {
            for (t=0; t<totalThreads; t++) {
                DEBUG_LOG(2,"Domain[%d]: %d, t[%d]: %u (log. idx)",q,domains[q],t,pumap->PUs[q][t]->logical_index);
            }
        }
    }


    goto out;
err:
    if (pumap) {
        if (pumap->PUs) {
            int d;
            for (d=0; d<pumap->nDomains; d++) {
                free(pumap->PUs[d]); pumap->PUs[d] = NULL;
            }
        }
        free(pumap->PUs); pumap->PUs = NULL;
        hwloc_bitmap_free(pumap->cpuset); pumap->cpuset = NULL;
        hwloc_bitmap_free(pumap->busy); pumap->busy = NULL;
    }
    free(pumap); pumap = NULL;
    
out:
    hwloc_bitmap_free(nodeset); nodeset = NULL;
    hwloc_bitmap_free(remoteCPUset); remoteCPUset = NULL;
    hwloc_bitmap_free(localCPUset); localCPUset = NULL;
    free(domains); domains = NULL;
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_TASKING|GHOST_FUNCTYPE_SETUP);
    return ret;

}

void ghost_pumap_destroy()
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_TASKING|GHOST_FUNCTYPE_TEARDOWN);
    
    if (pumap) {
        if (pumap->PUs) {
            int d;
            for (d=0; d<pumap->nDomains; d++) {
                free(pumap->PUs[d]); pumap->PUs[d] = NULL;
            }
        }
        free(pumap->PUs); pumap->PUs = NULL;
        hwloc_bitmap_free(pumap->cpuset); pumap->cpuset = NULL;
        hwloc_bitmap_free(pumap->busy); pumap->busy = NULL;
    }
    free(pumap); pumap = NULL;

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_TASKING|GHOST_FUNCTYPE_TEARDOWN);
}

ghost_error_t ghost_pumap_string(char **str)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_TASKING);
    
    int myrank;
    int mynoderank;
    ghost_mpi_comm_t nodecomm;
    
    GHOST_CALL_RETURN(ghost_nodecomm_get(&nodecomm));
    GHOST_CALL_RETURN(ghost_rank(&myrank, MPI_COMM_WORLD));
    GHOST_CALL_RETURN(ghost_rank(&mynoderank, nodecomm));
    GHOST_CALL_RETURN(ghost_malloc((void **)str,1));
    memset(*str,'\0',1);
    

    int nIdle = 0;
    char *cpusetstr = NULL;
    char *busystr = NULL;
    
    hwloc_bitmap_list_asprintf(&cpusetstr,pumap->cpuset);
    hwloc_bitmap_list_asprintf(&busystr,pumap->busy);
    GHOST_CALL_RETURN(ghost_pumap_nidle(&nIdle,GHOST_NUMANODE_ANY));

    ghost_header_string(str,"PU map @ local rank %d (glob %d)",mynoderank,myrank);
    ghost_line_string(str,"Total CPU set",NULL,"%s",cpusetstr);
    ghost_line_string(str,"Busy CPU set",NULL,"%s",cpusetstr);
    ghost_line_string(str,"No. of idle PUs",NULL,"%d",nIdle);
    ghost_footer_string(str);

    free(cpusetstr);
    free(busystr);
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_TASKING);

    return GHOST_SUCCESS;

}
