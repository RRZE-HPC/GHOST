#include "ghost/bitmap.h"

pthread_mutex_t ghost_bitmap_mutex = PTHREAD_MUTEX_INITIALIZER;


ghost_error_t ghost_bitmap_copy_indices(ghost_bitmap_t dst, ghost_lidx_t *offset, ghost_bitmap_t src, ghost_lidx_t *idx, ghost_lidx_t nidx)
{
    ghost_lidx_t origbit = -1;
    ghost_lidx_t previdx = -1;
    ghost_lidx_t i, j;
    ghost_lidx_t off = 0;
    
    for (i=0; i<nidx; i++) {
        for (j=0; j<idx[i]-previdx; j++) { // skip gaps
            origbit = ghost_bitmap_next(src,origbit);
        }
        if (i==0 && offset) {
            off = origbit;
        }
        ghost_bitmap_set(dst,origbit-off);
        previdx = idx[i];
    }

    if (offset) {
        *offset = off;
    }

    return GHOST_SUCCESS;
} 
    
int ghost_bitmap_first_safe(hwloc_const_bitmap_t bitmap)
{
    int ret = hwloc_bitmap_first(bitmap);
    return ret;

}
    
hwloc_obj_t ghost_get_obj_inside_cpuset_by_type_safe(hwloc_topology_t topology,hwloc_const_cpuset_t cpuset,hwloc_obj_type_t type, unsigned idx)
{
    hwloc_obj_t ret = hwloc_get_obj_inside_cpuset_by_type(topology, cpuset, type, idx);
    return ret;
}
