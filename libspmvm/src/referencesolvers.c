#include "referencesolvers.h"

void SpMVM_referenceKernel(mat_data_t *res, mat_nnz_t *col, mat_idx_t *rpt, mat_data_t *val, mat_data_t *rhs, mat_idx_t nrows, int spmvmOptions)
{

	mat_idx_t i, j;
	mat_data_t hlp1;

#pragma omp	parallel for schedule(runtime) private (hlp1, j)
	for (i=0; i<nrows; i++){
		hlp1 = 0.0;
		for (j=rpt[i]; j<rpt[i+1]; j++){
			hlp1 = hlp1 + val[j] * rhs[col[j]]; 
		}
		if (spmvmOptions & GHOST_OPTION_AXPY) 
			res[i] += hlp1;
		else
			res[i] = hlp1;
	}


}	
