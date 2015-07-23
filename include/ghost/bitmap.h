/**
 * @file bitmap.h
 * @brief Bitmap used for viewing densemat cols/rows in the leading dimension.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_BITMAP_H
#define GHOST_BITMAP_H

#include "config.h"
#include "types.h"
#include "error.h"
#include <hwloc/bitmap.h>
#include <hwloc.h>

/**
 * @brief ghost_bitmap_t is just an alias for hwloc_bitmap_t
 */
typedef hwloc_bitmap_t ghost_bitmap_t;

#define ghost_bitmap_set(bitmap,idx) hwloc_bitmap_set(bitmap,idx)
#define ghost_bitmap_set_range(bitmap,start,end) hwloc_bitmap_set_range(bitmap,start,end)
#define ghost_bitmap_clr_range(bitmap,start,end) hwloc_bitmap_clr_range(bitmap,start,end)
#define ghost_bitmap_list_asprintf(str,bitmap) hwloc_bitmap_list_asprintf(str,bitmap)
#define ghost_bitmap_copy(dst,src) hwloc_bitmap_copy(dst,src)
#define ghost_bitmap_clr(bitmap,idx) hwloc_bitmap_clr(bitmap,idx)
#define ghost_bitmap_first(bitmap) hwloc_bitmap_first(bitmap)
#define ghost_bitmap_next(bitmap,idx) hwloc_bitmap_next(bitmap,idx)
#define ghost_bitmap_alloc(bitmap) hwloc_bitmap_alloc(bitmap)
#define ghost_bitmap_free(bitmap) hwloc_bitmap_free(bitmap)
#define ghost_bitmap_isset(bitmap,idx) hwloc_bitmap_isset(bitmap,idx)
#define ghost_bitmap_iszero(bitmap) hwloc_bitmap_iszero(bitmap)
#define ghost_bitmap_isequal(bitmap1,bitmap2) hwloc_bitmap_isequal(bitmap1,bitmap2)
#define ghost_bitmap_last(bitmap) hwloc_bitmap_last(bitmap)
#define ghost_bitmap_weight(bitmap) hwloc_bitmap_weight(bitmap)
#define ghost_bitmap_iscompact(bitmap) ((ghost_bitmap_last(bitmap)-ghost_bitmap_first(bitmap)+1) == ghost_bitmap_weight(bitmap))

#ifdef __cplusplus
extern "C" {
#endif
    ghost_error_t ghost_bitmap_copy_indices(ghost_bitmap_t dst, ghost_lidx_t *offset, ghost_bitmap_t src, ghost_lidx_t *idx, ghost_lidx_t nidx);
    int ghost_bitmap_first_safe(hwloc_const_bitmap_t bitmap);
    hwloc_obj_t ghost_get_obj_inside_cpuset_by_type_safe(hwloc_topology_t topology,hwloc_const_cpuset_t cpuset,hwloc_obj_type_t type, unsigned idx);

    extern pthread_mutex_t ghost_bitmap_mutex;

#ifdef __cplusplus
}
#endif

#endif
