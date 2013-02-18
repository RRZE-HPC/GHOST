#ifndef __ELLPACK_CU_KERNEL_H__
#define __ELLPACK_CU_KERNEL_H__

#include <ghost.h>

void ELLPACK_kernel_wrap(ghost_dt *lhs, ghost_dt *rhs, int options, int nrows, int nrowspadded, ghost_midx_t *rowlen, ghost_midx_t *col, ghost_dt *val);

#endif

