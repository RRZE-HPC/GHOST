/**
 * @file bitmap.h
 * @brief Bitmap used for viewing densemat cols/rows in the leading dimension.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_BITMAP_H
#define GHOST_BITMAP_H

#include <hwloc/bitmap.h>

/**
 * @brief ghost_bitmap_t is just an alias for hwloc_bitmap_t
 */
typedef hwloc_bitmap_t ghost_bitmap_t;

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

#endif
