#include "ghost/pumap.h"
#include "ghost/machine.h"
#include "ghost/util.h"

static ghost_pumap_t *pumap = NULL;

ghost_error_t ghost_pumap_getNumberOfIdlePUs(int *nPUs, int numaNode)
{

    if (numaNode == GHOST_NUMANODE_ANY) {
        *nPUs = hwloc_bitmap_weight(pumap->cpuset)-hwloc_bitmap_weight(pumap->busy);
    } else {
        hwloc_obj_t node;
        ghost_getNumaNode(&node,numaNode);
        hwloc_bitmap_t LDBM = node->cpuset;

        hwloc_bitmap_t idle = hwloc_bitmap_alloc();
        hwloc_bitmap_andnot(idle,LDBM,pumap->busy);

        int w = hwloc_bitmap_weight(idle);

        hwloc_bitmap_free(idle);
        *nPUs = w;
    }

    return GHOST_SUCCESS;
}

ghost_error_t ghost_pumap_setIdle(hwloc_bitmap_t cpuset)
{
    hwloc_bitmap_andnot(pumap->busy,pumap->busy,cpuset);

    return GHOST_SUCCESS;
}

ghost_error_t ghost_pumap_setIdleIdx(int idx)
{
    hwloc_bitmap_clr(pumap->busy,idx);

    return GHOST_SUCCESS;
}

ghost_error_t ghost_pumap_setBusy(hwloc_bitmap_t cpuset)
{
    hwloc_bitmap_or(pumap->busy,pumap->busy,cpuset);
    
    return GHOST_SUCCESS;
}

ghost_error_t ghost_pumap_get(ghost_pumap_t **map)
{

    if (!map) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }

    *map = pumap;

    return GHOST_SUCCESS;
}

ghost_error_t ghost_pumap_create(hwloc_cpuset_t cpuset)
{
    int q,t;

    int totalThreads = hwloc_bitmap_weight(cpuset);
    hwloc_topology_t topology;
    ghost_getTopology(&topology);

    GHOST_CALL_RETURN(ghost_malloc((void **)&pumap,sizeof(ghost_pumap_t)));
    GHOST_CALL_RETURN(ghost_malloc((void **)&pumap->PUs,totalThreads*sizeof(hwloc_obj_t)));
    pumap->cpuset = hwloc_bitmap_dup(cpuset);
    pumap->busy = hwloc_bitmap_alloc();

    // get number of NUMA nodes
    hwloc_nodeset_t nodeset = hwloc_bitmap_alloc();
    hwloc_cpuset_to_nodeset(topology,cpuset,nodeset);
    pumap->nDomains = hwloc_bitmap_weight(nodeset);
    hwloc_bitmap_free(nodeset);

    GHOST_CALL_RETURN(ghost_malloc((void **)&pumap->PUs,sizeof(hwloc_obj_t *)*pumap->nDomains));
    
    // TODO error goto
    
    int localthreads;
    hwloc_obj_t numaNode;
    hwloc_cpuset_t remoteCPUset = hwloc_bitmap_alloc();
    // sort the cores according to the locality domain
    for (q=0; q<pumap->nDomains; q++) {
        GHOST_CALL_RETURN(ghost_getNumberOfPUs(&localthreads,q));
        GHOST_CALL_RETURN(ghost_malloc((void **)&pumap->PUs[q],sizeof(hwloc_obj_t)*totalThreads));
        GHOST_CALL_RETURN(ghost_getNumaNode(&numaNode,q)); 
        hwloc_bitmap_andnot(remoteCPUset,pumap->cpuset,numaNode->cpuset);

        for (t=0; t<localthreads; t++) { // my own threads
            pumap->PUs[q][t] = hwloc_get_obj_inside_cpuset_by_type(topology,numaNode->cpuset,HWLOC_OBJ_PU,t);
        }
        for (t=0; t<totalThreads-localthreads; t++) { // other NUMA nodes
            pumap->PUs[q][localthreads+t] = hwloc_get_obj_inside_cpuset_by_type(topology,remoteCPUset,HWLOC_OBJ_PU,t);
        }    
    }
    hwloc_bitmap_free(remoteCPUset);
    for (q=0; q<pumap->nDomains; q++) {
        for (t=0; t<totalThreads; t++) {
            INFO_LOG("LD: %d, t[%d] %d",q,t,pumap->PUs[q][t]->logical_index);
        }
    }
    return GHOST_SUCCESS;

}

void ghost_pumap_destroy()
{
    if (pumap) {
        int d;
        for (d=0; d<pumap->nDomains; d++) {
            free(pumap->PUs[d]); pumap->PUs[d] = NULL;
        }
        free(pumap->PUs); pumap->PUs = NULL;
        hwloc_bitmap_free(pumap->cpuset); pumap->cpuset = NULL;
        hwloc_bitmap_free(pumap->busy); pumap->busy = NULL;
    }
    free(pumap); pumap = NULL;

}

