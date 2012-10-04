#include <spmvm.h>
#include <spmvm_util.h>

#include <limits.h>
#include <libgen.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>

/** function to initialize the RHS vector **/
static data_t rhsVal (int i) {
	return i+1.0;
}

int main( int argc, char* argv[] ) {

	int me, nIter = 100;
	double time;
	int kernel = SPMVM_KERNEL_TASKMODE;
	int options = SPMVM_OPTION_NONE;

	LCRP_TYPE       *lcrp;    // local CRS portion
	VECTOR_TYPE     *nodeLHS; // lhs vector per node
	VECTOR_TYPE     *nodeRHS; // rhs vector node

	me      = SpMVM_init(argc,argv,options); // basic initialization
	lcrp    = SpMVM_createCRS(argv[1],NULL); // create CRS matrix 
	nodeRHS = SpMVM_createVector(lcrp,VECTOR_TYPE_RHS,rhsVal); // RHS vec
	nodeLHS = SpMVM_createVector(lcrp,VECTOR_TYPE_LHS,NULL);   // LHS vec (=0)

	SpMVM_printEnvInfo();
	SpMVM_printMatrixInfo(lcrp,strtok(basename(argv[1]),"_."),options);

	time = SpMVM_solve(nodeLHS,lcrp,nodeRHS,kernel,nIter);

	if (me == 0){
		printf("%s kernel @ %7.2f GF/s\n",SpMVM_kernelName(kernel),
				2.0e-9*(double)lcrp->nEnts/time);
	}

	SpMVM_freeVector( nodeLHS );
	SpMVM_freeVector( nodeRHS );
	SpMVM_freeLCRP( lcrp );

	SpMVM_finish();

	return EXIT_SUCCESS;
}
