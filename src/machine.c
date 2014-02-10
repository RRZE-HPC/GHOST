#include "ghost/machine.h"
#include "ghost/log.h"

static hwloc_topology_t ghost_topology = NULL;

ghost_error_t ghost_createTopology()
{
    if (ghost_topology) {
        return GHOST_SUCCESS;
    }

    if (hwloc_topology_init(&ghost_topology)) {
        ERROR_LOG("Could not init topology");
        return GHOST_ERR_HWLOC;
    }

    if (hwloc_topology_load(ghost_topology)) {
        ERROR_LOG("Could not load topology");
        return GHOST_ERR_HWLOC;
    }

    return GHOST_SUCCESS;
}

void ghost_destroyTopology()
{
    hwloc_topology_destroy(ghost_topology);

    return;
}

ghost_error_t ghost_getSizeOfLLC(uint64_t *size)
{
    hwloc_topology_t topology;
    GHOST_CALL_RETURN(ghost_getTopology(&topology));
    hwloc_obj_t obj;
    int depth;

    for (depth=0; depth<(int)hwloc_topology_get_depth(topology); depth++) {
        obj = hwloc_get_obj_by_depth(topology,depth,0);
        if (obj->type == HWLOC_OBJ_CACHE) {
            *size = obj->attr->cache.size;
            break;
        }
    }

#if GHOST_HAVE_MIC
    size = size*ghost_getNumberOfPhysicalCores(); // the cache is shared but not reported so
#endif

    return GHOST_SUCCESS;
}

ghost_error_t ghost_getSizeOfCacheLine(unsigned *size)
{
    hwloc_topology_t topology;
    GHOST_CALL_RETURN(ghost_getTopology(&topology));

    
    hwloc_obj_t obj;
    int depth;

    for (depth=0; depth<(int)hwloc_topology_get_depth(topology); depth++) {
        obj = hwloc_get_obj_by_depth(topology,depth,0);
        if (obj->type == HWLOC_OBJ_CACHE) {
            *size = obj->attr->cache.linesize;
        }
    }

    return GHOST_SUCCESS;
}

ghost_error_t ghost_getTopology(hwloc_topology_t *topo)
{
    if (!topo) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }

    if (!ghost_topology) {
        ERROR_LOG("Topology does not exist!");
        return GHOST_ERR_UNKNOWN;
    }
    
    *topo = ghost_topology;

    return GHOST_SUCCESS;

}

ghost_error_t ghost_getNumberOfCores(int *nCores, int numaNode)
{
    int nPUs;
    int smt;
    GHOST_CALL_RETURN(ghost_getNumberOfPUs(&nPUs, numaNode));
    GHOST_CALL_RETURN(ghost_getSMTlevel(&smt));

    if (smt == 0) {
        ERROR_LOG("The SMT level is zero");
        return GHOST_ERR_UNKNOWN;
    }

    if (nPUs % smt) {
        ERROR_LOG("The number of PUs is not a multiple of the SMT level");
        return GHOST_ERR_UNKNOWN;
    }

    *nCores = nPUs/smt;

    return GHOST_SUCCESS;
}

ghost_error_t ghost_getNumberOfPUs(int *nPUs, int numaNode)
{
    hwloc_topology_t topology;
    GHOST_CALL_RETURN(ghost_getTopology(&topology));
    hwloc_const_cpuset_t cpuset = NULL;
    if (numaNode == GHOST_NUMANODE_ANY) {
        cpuset = hwloc_topology_get_allowed_cpuset(topology);
    } else {
        hwloc_obj_t node;
        ghost_getNumaNode(&node,numaNode);
        cpuset = node->cpuset;
    }

    *nPUs = hwloc_bitmap_weight(cpuset);

    return GHOST_SUCCESS;
}

ghost_error_t ghost_getSMTlevel(int *nLevels)
{
    hwloc_topology_t topology;
    GHOST_CALL_RETURN(ghost_getTopology(&topology));

    int nCores = hwloc_get_nbobjs_by_type(topology,HWLOC_OBJ_CORE);
    if (nCores == 0) {
        ERROR_LOG("Can not compute SMT level because the number of cores is zero");
        return GHOST_ERR_HWLOC;
    }

    hwloc_obj_t firstCore = hwloc_get_obj_by_type(topology,HWLOC_OBJ_CORE,0);
    *nLevels = (int)firstCore->arity;
    
    return GHOST_SUCCESS;
}

ghost_error_t ghost_getNumberOfNumaNodes(int *nNodes)
{
    hwloc_topology_t topology;
    GHOST_CALL_RETURN(ghost_getTopology(&topology));
    int nnodes = hwloc_get_nbobjs_by_type(topology,HWLOC_OBJ_NODE);
    if (nnodes < 0) {
        ERROR_LOG("Could not obtain number of NUMA nodes");
        return GHOST_ERR_HWLOC;
    } else if (nnodes == 0) {
        *nNodes = 1;
    } else {
        *nNodes = nnodes;
    }

    return GHOST_SUCCESS;

}

char ghost_machineIsBigEndian()
{
    int test = 1;
    unsigned char *endiantest = (unsigned char *)&test;

    return (endiantest[0] == 0);
}

ghost_error_t ghost_getNumaNode(hwloc_obj_t *node, int idx)
{
    if (!node) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }

    hwloc_topology_t topology;
    GHOST_CALL_RETURN(ghost_getTopology(&topology));
    
    int nNodes = hwloc_get_nbobjs_by_type(topology,HWLOC_OBJ_NODE);
    
    if (idx < 0 || idx >= nNodes) {
        ERROR_LOG("Index out of range");
        return GHOST_ERR_INVALID_ARG;
    }

    if (nNodes == 0) {    
        *node = hwloc_get_obj_by_type(topology,HWLOC_OBJ_SOCKET,0);
    } else {
        *node = hwloc_get_obj_by_type(topology,HWLOC_OBJ_NODE,idx);
    }

    return GHOST_SUCCESS;
}
/*
ghost_error_t ghost_getNumberOfPUsInLD(int ld)
{
    int i,n = 0;
    hwloc_obj_t obj,runner;
    for (i=0; i<ghost_thpool->nThreads; i++) {    
        obj = ghost_thpool->PUs[i];
        for (runner=obj; runner; runner=runner->parent) {
            if (runner->type <= HWLOC_OBJ_NODE) {
                if ((int)runner->logical_index == ld) {
                    n++;
                }
                break;
            }
        }
    }

    return n;
}*/
