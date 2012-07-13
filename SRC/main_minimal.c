#include "spmvm_util.h"
#include "matricks.h"
#include "mpihelper.h"
#include "timing.h"

#include <limits.h>
#include <libgen.h>


/** function to initialize the RHS vector **/
double rhsVal (int i) {
	return i+1.0;
}

int main( int argc, char* argv[] ) {

	int ierr, me, it;
	int nIter = 100;
	int kernel = 12;
	char matrixPath[PATH_MAX];
	double start, end, dummy;

	CR_TYPE         *cr;      // global CRS matrix
	LCRP_TYPE       *lcrp;    // local CRS portion
	VECTOR_TYPE     *nodeLHS; // lhs vector per node
	VECTOR_TYPE     *nodeRHS; // rhs vector node
	HOSTVECTOR_TYPE *globRHS; // global rhs vector
	HOSTVECTOR_TYPE *globLHS; // global lhs vector


	/** get full path from given matrix **/
	getMatrixPath(argv[1],matrixPath);
	if (!matrixPath)
		myabort("No valid matrix specified! (no absolute file name and not present in $MATHOME)");
	
	
	SPMVM_OPTIONS = SPMVM_OPTION_NONE;     // performan standard spmvm
	JOBMASK       = 0x1<<kernel;           // setup job mask

	me      = SpMVM_init(argc,argv);       // basic initialization
	cr      = SpMVM_createCRS(matrixPath); // create CRS matrix from path

	globRHS = SpMVM_createGlobalHostVector(cr->nCols, rhsVal); // create global RHS vector & initialize with function pointer
	globLHS = SpMVM_createGlobalHostVector(cr->nCols, NULL);   // create global LHS vector & initialize with zero

	lcrp    = SpMVM_distributeCRS(cr);    // distribute CRS matrix to nodes
	
	nodeRHS = SpMVM_distributeVector(lcrp,globRHS); // distribute RHS vector
	nodeLHS = SpMVM_distributeVector(lcrp,globLHS); // distribute LHS vector

	SpMVM_printMatrixInfo(lcrp,basename(matrixPath));     // print matrix information

	if (me == 0) timing(&start,&dummy);

	for( it = 0; it < nIter; it++ ) {
		HyK[kernel].kernel( it, nodeLHS, lcrp, nodeRHS); // execute kernel
		MPI_Barrier(MPI_COMM_WORLD);
	}

	if (me == 0) {
		timing(&end,&dummy);
		printf("Kernel %2d @ %7.2f GF/s\n",kernel,2.0e-9*(double)nIter*(double)lcrp->nEnts/(end-start));
	}

	freeVector( nodeLHS );
	freeVector( nodeRHS );
	freeHostVector( globLHS );
	freeHostVector( globRHS );
	freeLcrpType( lcrp );

	MPI_Finalize();

	return EXIT_SUCCESS;
}
