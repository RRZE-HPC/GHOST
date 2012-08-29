#include <matricks.h>
#include <mpi.h>
#include "kernel_helper.h"
#include "kernel.h"

void hybrid_kernel_0(VECTOR_TYPE* res, LCRP_TYPE* lcrp, VECTOR_TYPE* invec)
{


	/***************************************************************************
	 ********            rein OpenMP-paralleler Kernel: ca                ******
	 ********             ausschliesslich fuer den Fall np=1              ******
	 **************************************************************************/

	int me=0;

	spmvmKernAll(lcrp, invec, res, &me);
}
