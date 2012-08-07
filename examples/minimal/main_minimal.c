#include "spmvm_util.h"
#include "matricks.h"
#include "mpihelper.h"
#include "timing.h"

#include <limits.h>
#include <libgen.h>


/** function to initialize the RHS vector **/
real rhsVal (int i) {
	return i+1.0;
}

int main( int argc, char* argv[] ) {

	int ierr, me, it, nIter = 100, kernel;
	double start, end, dummy;

	CR_TYPE         *cr;      // global CRS matrix
	LCRP_TYPE       *lcrp;    // local CRS portion
	VECTOR_TYPE     *nodeLHS; // lhs vector per node
	VECTOR_TYPE     *nodeRHS; // rhs vector node
	HOSTVECTOR_TYPE *globRHS; // global rhs vector
	HOSTVECTOR_TYPE *globLHS; // global lhs vector

	SPMVM_OPTIONS = SPMVM_OPTION_NONE; // performan standard spmvm
	SPMVM_KERNELS_SELECTED = SPMVM_KERNEL_VECTORMODE;            // setup kernels to execute
	SPMVM_KERNELS_SELECTED |= SPMVM_KERNEL_TASKMODE;

	me      = SpMVM_init(argc,argv);    // basic initialization
	cr      = SpMVM_createCRS(argv[1]); // create CRS matrix from given matrix path

	globRHS = SpMVM_createGlobalHostVector(cr->nCols, rhsVal); // create global RHS vector & initialize with function pointer
	globLHS = SpMVM_createGlobalHostVector(cr->nCols, NULL);   // create global LHS vector & initialize with zero

	lcrp    = SpMVM_distributeCRS(cr);    // distribute CRS matrix to nodes

	nodeRHS = SpMVM_distributeVector(lcrp,globRHS); // distribute RHS vector
	nodeLHS = SpMVM_distributeVector(lcrp,globLHS); // distribute LHS vector

	SpMVM_printEnvInfo();
	SpMVM_printMatrixInfo(lcrp,strtok(basename(argv[1]),"_."));

	for (kernel=0; kernel < SPMVM_NUMKERNELS; kernel++){

		if (!SpMVM_kernelValid(kernel,lcrp)) 
			continue; // Skip loop body if kernel does not make sense for used parametes
		
		if (me == 0) timing(&start,&dummy);

		for( it = 0; it < nIter; it++ ) {
			SPMVM_KERNELS[kernel].kernel(nodeLHS, lcrp, nodeRHS); // execute kernel
			MPI_Barrier(MPI_COMM_WORLD);
		}

		if (me == 0) {
			timing(&end,&dummy);
			printf("Kernel %2d @ %7.2f GF/s\n",kernel,2.0e-9*(double)nIter*(double)lcrp->nEnts/(end-start));
		}
	}

	freeVector( nodeLHS );
	freeVector( nodeRHS );
	freeHostVector( globLHS );
	freeHostVector( globRHS );
	freeLcrpType( lcrp );

	MPI_Finalize();

	return EXIT_SUCCESS;
}
