#include "ghost/bitmap.h"
#include "ghost/func_util.h"



ghost_error ghost_bitmap_copy_indices(ghost_bitmap dst, ghost_lidx *offset, ghost_bitmap src, ghost_lidx *idx, ghost_lidx nidx)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    ghost_lidx origbit = -1;
    ghost_lidx previdx = -1;
    ghost_lidx i, j;
    ghost_lidx off = 0;
    
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

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
} 
    
