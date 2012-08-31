#include <matricks.h>
#include <mpi.h>
#include "kernel_helper.h"
#include "kernel.h"

#ifdef LIKWID
#include <likwid.h>
#endif
void hybrid_kernel_0(VECTOR_TYPE* res, LCRP_TYPE* lcrp, VECTOR_TYPE* invec)
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

	spmvmKernAll(lcrp, invec, res, &me);

#ifdef LIKWID_MARKER
#pragma omp parallel
	likwid_markerStopRegion("Kernel 0");
#endif
}
