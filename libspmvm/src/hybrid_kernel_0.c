#include "kernel_helper.h"
#include "kernel.h"
#include "matricks.h"
#include <stdio.h>

#ifdef LIKWID
#include <likwid.h>
#endif

void hybrid_kernel_0(ghost_vec_t* res, ghost_setup_t* setup, ghost_vec_t* invec, int spmvmOptions)
{
	setup->fullMatrix->kernel(res,invec,spmvmOptions);
}

void kern_glob_CRS_0(ghost_vec_t* res, CR_TYPE* cr, ghost_vec_t* invec, int spmvmOptions)
{
	mat_idx_t i, j;
	mat_data_t hlp1;

#pragma omp	parallel for schedule(runtime) private (hlp1, j)
	for (i=0; i<cr->nrows; i++){
		hlp1 = 0.0;
		for (j=cr->rpt[i]; j<cr->rpt[i+1]; j++){
			hlp1 = hlp1 + cr->val[j] * invec->val[cr->col[j]]; 
		}
		if (spmvmOptions & GHOST_OPTION_AXPY) 
			res->val[i] += hlp1;
		else
			res->val[i] = hlp1;
	}

}

void kern_glob_CRS_CD_0(ghost_vec_t* res, CR_TYPE* cr, ghost_vec_t* invec, int spmvmOptions)
{
	mat_idx_t i, j;
	mat_data_t hlp1;

#pragma omp	parallel for schedule(runtime) private (hlp1, j)
	for (i=0; i<cr->nrows; i++){
		hlp1 = 0.0;
		for (j=cr->rpt[i]; j<cr->rpt[i+1]; j++) {
			hlp1 = hlp1 + cr->val[j] * invec->val[cr->col[j]]; 
		}

		for (j=0; j<cr->nConstDiags; j++) {
			if (i >= cr->constDiags[j].minRow && i <= cr->constDiags[j].maxRow) {
				hlp1 = hlp1 + cr->constDiags[j].val * invec->val[i+cr->constDiags[j].idx];
			}
		}

		if (spmvmOptions & GHOST_OPTION_AXPY) 
			res->val[i] += hlp1;
		else
			res->val[i] = hlp1;
	}

}
