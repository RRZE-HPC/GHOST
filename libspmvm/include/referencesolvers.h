#ifndef _REFERENCESOLVERS_H_
#define _REFERENCESOLVERS_H_

#include "ghost.h"

void ghost_referenceKernel(ghost_mdat_t *res, mat_nnz_t *col, mat_idx_t *rpt, ghost_mdat_t *val, ghost_mdat_t *rhs, mat_idx_t nrows, int spmvmOptions);

#endif
