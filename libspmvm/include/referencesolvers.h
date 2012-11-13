#ifndef _REFERENCESOLVERS_H_
#define _REFERENCESOLVERS_H_

#include "spmvm.h"

void SpMVM_referenceKernel(mat_data_t *res, mat_nnz_t *col, mat_idx_t *rpt, mat_data_t *val, mat_data_t *rhs, mat_idx_t nrows, int spmvmOptions);

#endif
