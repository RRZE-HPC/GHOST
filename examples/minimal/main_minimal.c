#include <spmvm.h>
#include <spmvm_util.h>

#include <limits.h>
#include <libgen.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>


/** function to initialize the RHS vector **/
static real rhsVal (int i) {
	return i+1.0;
}

int main( int argc, char* argv[] ) {

	int me, nIter = 100;
	double time;

	CR_TYPE         *cr;      // global CRS matrix
	LCRP_TYPE       *lcrp;    // local CRS portion
	VECTOR_TYPE     *nodeLHS; // lhs vector per node
	VECTOR_TYPE     *nodeRHS; // rhs vector node
	HOSTVECTOR_TYPE *globRHS; // global rhs vector
	HOSTVECTOR_TYPE *globLHS; // global lhs vector

	int options = SPMVM_OPTION_NONE; // performan standard spmvm
	int kernel = SPMVM_KERNEL_NOMPI;

	me      = SpMVM_init(argc,argv,options);    // basic initialization
	cr      = SpMVM_createCRS(argv[1]); // create CRS matrix from given matrix path

	globRHS = SpMVM_createGlobalHostVector(cr->nCols, rhsVal); // create global RHS vector & initialize with function pointer
	globLHS = SpMVM_createGlobalHostVector(cr->nCols, NULL);   // create global LHS vector & initialize with zero

	lcrp    = SpMVM_distributeCRS(cr,NULL);    // distribute CRS matrix to nodes

	nodeRHS = SpMVM_distributeVector(lcrp,globRHS); // distribute RHS vector
	nodeLHS = SpMVM_distributeVector(lcrp,globLHS); // distribute LHS vector

	SpMVM_printEnvInfo();
	SpMVM_printMatrixInfo(lcrp,strtok(basename(argv[1]),"_."),options);

	time = SpMVM_solve(nodeLHS,lcrp,nodeRHS,kernel,nIter);

	if (me == 0) {
		printf("%s kernel @ %7.2f GF/s\n",SpMVM_kernelName(kernel),2.0e-9*(double)nIter*(double)lcrp->nEnts/(time));
	}

	SpMVM_freeVector( nodeLHS );
	SpMVM_freeVector( nodeRHS );
	SpMVM_freeHostVector( globLHS );
	SpMVM_freeHostVector( globRHS );
	SpMVM_freeLCRP( lcrp );
	SpMVM_freeCRS( cr );

	SpMVM_finish();

	return EXIT_SUCCESS;
}
