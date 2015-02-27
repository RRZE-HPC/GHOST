#include "ghost/bitmap.h"


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
