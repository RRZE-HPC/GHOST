#include "ghost/machine.h"
#include "ghost/log.h"


static hwloc_topology_t ghost_topology = NULL;

ghost_error_t ghost_initTopology()
{
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

ghost_error_t ghost_getTopology(hwloc_topology_t *topo){
    if (!ghost_topology) {
        ERROR_LOG("Topology does not exist!");
        return GHOST_ERR_UNKNOWN;
    }
    *topo = ghost_topology;

    return GHOST_SUCCESS;

}

ghost_error_t ghost_getNumberOfPhysicalCores(int *nCores)
{
    hwloc_topology_t topology;
    GHOST_CALL_RETURN(ghost_getTopology(&topology));
    int ncores = hwloc_get_nbobjs_by_type(topology,HWLOC_OBJ_CORE);
    if (ncores < 0) {
        ERROR_LOG("Could not obtain number of physical cores");
        return GHOST_ERR_HWLOC;
    }

    *nCores = ncores;

    return GHOST_SUCCESS;
}

ghost_error_t ghost_getNumberOfHwThreads(int *nThreads)
{
    hwloc_topology_t topology;
    GHOST_CALL_RETURN(ghost_getTopology(&topology));
    *nThreads = hwloc_get_nbobjs_by_type(topology,HWLOC_OBJ_PU);    

    return GHOST_SUCCESS;
}

ghost_error_t ghost_getSMTlevel(int *nLevels)
{
    hwloc_topology_t topology;
    GHOST_CALL_RETURN(ghost_getTopology(&topology));

    int nHwThreads;
    int nCores;

    GHOST_CALL_RETURN(ghost_getNumberOfHwThreads(&nHwThreads));
    GHOST_CALL_RETURN(ghost_getNumberOfPhysicalCores(&nCores));

    if (nCores == 0) {
        ERROR_LOG("Can not compute SMT level because the number of cores is zero");
        return GHOST_ERR_HWLOC;
    }

    *nLevels = nHwThreads/nCores;

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
