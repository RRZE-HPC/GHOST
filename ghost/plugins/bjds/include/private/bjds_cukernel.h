#ifndef __BJDS_CU_KERNEL_H__
#define __BJDS_CU_KERNEL_H__

#include <ghost.h>

void BJDS_kernel_wrap(ghost_vdat_t *lhs, ghost_vdat_t *rhs, int options, int nrows, int nrowspadded, ghost_midx_t *rowlen, ghost_midx_t *col, ghost_mdat_t *val, ghost_mnnz_t *chunkstart, ghost_midx_t *chunklen);

#endif

