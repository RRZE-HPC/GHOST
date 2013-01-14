#include "kernel_helper.h"
#include "kernel.h"
#include "ghost_mat.h"
#include <stdio.h>

#ifdef LIKWID_PERFMON
#include <likwid.h>
#endif

void hybrid_kernel_0(ghost_vec_t* res, ghost_context_t* context, ghost_vec_t* invec, int spmvmOptions)
{
#ifdef LIKWID_PERFMON
#pragma omp parallel
	LIKWID_MARKER_START("full SpMVM");
#endif
	context->fullMatrix->kernel(context->fullMatrix,res,invec,spmvmOptions);
#ifdef LIKWID_PERFMON
#pragma omp parallel
	LIKWID_MARKER_STOP("full SpMVM");
#endif
}

void kern_glob_CRS_0(ghost_vec_t* res, CR_TYPE* cr, ghost_vec_t* invec, int spmvmOptions)
{
	ghost_midx_t i, j;
	ghost_vdat_t hlp1;

#pragma omp	parallel for schedule(runtime) private (hlp1, j)
	for (i=0; i<cr->nrows; i++){
		hlp1 = 0.0;
		for (j=cr->rpt[i]; j<cr->rpt[i+1]; j++){
			hlp1 = hlp1 + (ghost_vdat_t)cr->val[j] * invec->val[cr->col[j]]; 
		}
		if (spmvmOptions & GHOST_OPTION_AXPY) 
			res->val[i] += hlp1;
		else
			res->val[i] = hlp1;
	}

}

void kern_glob_CRS_CD_0(ghost_vec_t* res, CR_TYPE* cr, ghost_vec_t* invec, int spmvmOptions)
{
	ghost_midx_t i, j;
	ghost_vdat_t hlp1;

#pragma omp	parallel for schedule(runtime) private (hlp1, j)
	for (i=0; i<cr->nrows; i++){
		hlp1 = 0.0;
		for (j=cr->rpt[i]; j<cr->rpt[i+1]; j++) {
			hlp1 = hlp1 + (ghost_vdat_t)cr->val[j] * invec->val[cr->col[j]]; 
		}

		for (j=0; j<cr->nConstDiags; j++) {
			if (i >= cr->constDiags[j].minRow && i <= cr->constDiags[j].maxRow) {
				hlp1 = hlp1 + (ghost_vdat_t)cr->constDiags[j].val * invec->val[i+cr->constDiags[j].idx];
			}
		}

		if (spmvmOptions & GHOST_OPTION_AXPY) 
			res->val[i] += hlp1;
		else
			res->val[i] = hlp1;
	}

}
