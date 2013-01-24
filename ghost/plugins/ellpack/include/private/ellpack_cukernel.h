#ifndef __ELLPACK_CU_KERNEL_H__
#define __ELLPACK_CU_KERNEL_H__

#include <ghost.h>
#include <ghost_util.h>
#include <ghost_types.h>


void ELLPACK_kernel_wrap(ghost_vdat_t *lhs, ghost_vdat_t *rhs, int options, int nrows, int nrowspadded, ghost_midx_t *rowlen, ghost_midx_t *col, ghost_mdat_t *val);

#endif

