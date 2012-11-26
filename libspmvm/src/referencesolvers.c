#include "referencesolvers.h"

void ghost_referenceKernel(ghost_mdat_t *res, mat_nnz_t *col, mat_idx_t *rpt, ghost_mdat_t *val, ghost_mdat_t *rhs, mat_idx_t nrows, int spmvmOptions)
{

	mat_idx_t i, j;
	ghost_mdat_t hlp1;

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
