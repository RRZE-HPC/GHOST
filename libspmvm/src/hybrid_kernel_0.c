#include "kernel_helper.h"
#include "kernel.h"

#ifdef LIKWID
#include <likwid.h>
#endif
void hybrid_kernel_0(VECTOR_TYPE* res, LCRP_TYPE* lcrp, VECTOR_TYPE* invec, int spmvmOptions)
{


	/***************************************************************************
	 ********            rein OpenMP-paralleler Kernel: ca                ******
	 ********             ausschliesslich fuer den Fall np=1              ******
	 **************************************************************************/

	int me=0;

#ifdef LIKWID_MARKER
#pragma omp parallel
	likwid_markerStartRegion("Kernel 0");
#endif


	spmvmKernAll(lcrp, invec, res, &me, spmvmOptions);

#ifdef LIKWID_MARKER
#pragma omp parallel
	likwid_markerStopRegion("Kernel 0");
#endif
}

void kern_glob_CRS_0(VECTOR_TYPE* res, CR_TYPE* cr, VECTOR_TYPE* invec, int spmvmOptions)
{
int i, j;
data_t hlp1;

#pragma omp	parallel for schedule(runtime) private (hlp1, j)
	for (i=0; i<cr->nRows; i++){
		hlp1 = 0.0;
		for (j=cr->rowOffset[i]; j<cr->rowOffset[i+1]; j++){
			hlp1 = hlp1 + cr->val[j] * invec->val[cr->col[j]]; 
		}
		if (spmvmOptions & SPMVM_OPTION_AXPY) 
			res->val[i] += hlp1;
		else
			res->val[i] = hlp1;
	}

}
