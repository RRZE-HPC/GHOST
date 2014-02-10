#include "ghost/cpumap.h"
#include "ghost/machine.h"
#include "ghost/util.h"

//static int** coreidx = NULL;
static ghost_cpumap_t *cpumap = NULL;

/*int coreIdx(int LD, int t)
{

    return coreidx[LD][t];
}*/

int nIdleCoresAtLD(int ld)
{

    hwloc_obj_t node;
    ghost_getNumaNode(&node,ld);
    hwloc_bitmap_t LDBM = node->cpuset;

    hwloc_bitmap_t idle = hwloc_bitmap_alloc();
    hwloc_bitmap_andnot(idle,LDBM,cpumap->busy);

    int w = hwloc_bitmap_weight(idle);

    hwloc_bitmap_free(LDBM);
    hwloc_bitmap_free(idle);
    return w;
}

int nIdleCores()
{
    return hwloc_bitmap_weight(cpumap->cpuset)-hwloc_bitmap_weight(cpumap->busy);
}


int nBusyCoresAtLD(hwloc_bitmap_t bm, int ld)
{
    hwloc_obj_t node;
    ghost_getNumaNode(&node,ld);
    hwloc_bitmap_t LDBM = node->cpuset;

    hwloc_bitmap_t busy = hwloc_bitmap_alloc();
    hwloc_bitmap_and(busy,LDBM,bm);

    int w = hwloc_bitmap_weight(busy);

    hwloc_bitmap_free(LDBM);
    hwloc_bitmap_free(busy);
    return w;
}

ghost_error_t ghost_setCPUidle(int cpu)
{
    hwloc_bitmap_clr(cpumap->busy,cpu);

    return GHOST_SUCCESS;
}

ghost_error_t ghost_setCPUbusy(int cpu)
{
    hwloc_bitmap_set(cpumap->busy,cpu);
    
    return GHOST_SUCCESS;
}

ghost_error_t ghost_getCPUmap(ghost_cpumap_t **map)
{

    if (!map) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }

    *map = cpumap;

    return GHOST_SUCCESS;
}

ghost_error_t ghost_createCPUmap(hwloc_cpuset_t cpuset)
{
    int q,t;

    int totalThreads = hwloc_bitmap_weight(cpuset);
    hwloc_topology_t topology;
    ghost_getTopology(&topology);

    GHOST_CALL_RETURN(ghost_malloc((void **)&cpumap,sizeof(ghost_cpumap_t)));
    GHOST_CALL_RETURN(ghost_malloc((void **)&cpumap->PUs,totalThreads*sizeof(hwloc_obj_t)));
    cpumap->cpuset = hwloc_bitmap_dup(cpuset);
    cpumap->busy = hwloc_bitmap_alloc();

    // get number of NUMA nodes
    hwloc_nodeset_t nodeset = hwloc_bitmap_alloc();
    hwloc_cpuset_to_nodeset(topology,cpuset,nodeset);
    cpumap->nLDs = hwloc_bitmap_weight(nodeset);
    hwloc_bitmap_free(nodeset);

    GHOST_CALL_RETURN(ghost_malloc((void **)&cpumap->PUs,sizeof(hwloc_obj_t *)*cpumap->nLDs));
    
    // TODO error goto
    
    int localthreads;
    hwloc_obj_t numaNode;
    hwloc_cpuset_t remoteCPUset = hwloc_bitmap_alloc();
    // sort the cores according to the locality domain
    for (q=0; q<cpumap->nLDs; q++) {
        GHOST_CALL_RETURN(ghost_getNumberOfPUs(&localthreads,q));
        GHOST_CALL_RETURN(ghost_malloc((void **)&cpumap->PUs[q],sizeof(hwloc_obj_t)*totalThreads));
        GHOST_CALL_RETURN(ghost_getNumaNode(&numaNode,q)); 
        hwloc_bitmap_andnot(remoteCPUset,cpumap->cpuset,numaNode->cpuset);

        for (t=0; t<localthreads; t++) { // my own threads
            cpumap->PUs[q][t] = hwloc_get_obj_inside_cpuset_by_type(topology,numaNode->cpuset,HWLOC_OBJ_PU,t);
        }
        for (t=0; t<totalThreads-localthreads; t++) { // other NUMA nodes
            cpumap->PUs[q][localthreads+t] = hwloc_get_obj_inside_cpuset_by_type(topology,remoteCPUset,HWLOC_OBJ_PU,t);
        }    
    }
    for (q=0; q<cpumap->nLDs; q++) {
        for (t=0; t<totalThreads; t++) {
            INFO_LOG("LD: %d, t[%d] %d",q,t,cpumap->PUs[q][t]->logical_index);
        }
    }
    return GHOST_SUCCESS;

}

void ghost_destroyCPUmap()
{
    if (cpumap) {
        free(cpumap->PUs); cpumap->PUs = NULL;
        hwloc_bitmap_free(cpumap->cpuset); cpumap->cpuset = NULL;
        hwloc_bitmap_free(cpumap->busy); cpumap->busy = NULL;
    }
    free(cpumap); cpumap = NULL;

}

