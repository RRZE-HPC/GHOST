#include <spmvm.h>
#include <spmvm_util.h>

#include <limits.h>
#include <libgen.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>

/** function to initialize the RHS vector **/
static mat_data_t rhsVal (int i) {
	return i+1.0;
}

int main( int argc, char* argv[] ) {

	int me, nIter = 100;
	double time;
	int kernel = SPMVM_KERNEL_TASKMODE;
	int options = SPMVM_OPTION_PIN;

	MATRIX_TYPE     *matrix;    // local CRS portion
	VECTOR_TYPE     *nodeLHS; // lhs vector per node
	VECTOR_TYPE     *nodeRHS; // rhs vector node

	me      = SpMVM_init(argc,argv,options); // basic initialization
	matrix  = SpMVM_createMatrix(argv[1],SPM_FORMAT_DIST_CRS,NULL); // create CRS matrix 
	nodeRHS = SpMVM_createVector(matrix,VECTOR_TYPE_RHS,rhsVal); // RHS vec
	nodeLHS = SpMVM_createVector(matrix,VECTOR_TYPE_LHS,NULL);   // LHS vec (=0)

	SpMVM_printEnvInfo();
	SpMVM_printMatrixInfo(matrix,strtok(basename(argv[1]),"_."),options);

	time = SpMVM_solve(nodeLHS,matrix,nodeRHS,kernel,nIter);

	if (me == 0){
		printf("%s kernel @ %7.2f GF/s\n",SpMVM_kernelName(kernel),
				2.0e-9*(double)matrix->nNonz/time);
	}

	SpMVM_freeVector( nodeLHS );
	SpMVM_freeVector( nodeRHS );
	//SpMVM_freeLCRP( lcrp );

	SpMVM_finish();

	return EXIT_SUCCESS;
}
