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

	int nIter = 100;
	double time;
	int kernel = SPMVM_KERNEL_NOMPI;
	int options = SPMVM_OPTION_SERIAL_IO;

	CPU_MATRIX  *mat; // global matrix
	VECTOR_TYPE *lhs; // lhs vector per node
	VECTOR_TYPE *rhs; // rhs vector node

	SpMVM_init(argc,argv,options); // basic initialization
	mat = SpMVM_createGlobalMatrix(argv[1],SPM_FORMAT_MICVEC);  // create CRS matrix 
	rhs = SpMVM_newVector(mat->nRows); // RHS vec
	lhs = SpMVM_newVector(mat->nRows); // LHS vec (=0)

	int i;
	for (i=0; i<mat->nRows; i++) rhs->val[i] = rhsVal(i);

	//SpMVM_printEnvInfo();
	//SpMVM_printMatrixInfo(lcrp,strtok(basename(argv[1]),"_."),options);

	time = SpMVM_solve(lhs,mat,rhs,kernel,nIter);

		printf("%s kernel @ %7.2f GF/s\n",SpMVM_kernelName(kernel),
				2.0e-9*(double)mat->nNz/time);

		for (i=0; i<mat->nRows; i++) printf("%f\n",lhs->val[i]);

	SpMVM_freeVector( lhs );
	SpMVM_freeVector( rhs );
//	SpMVM_freeLCRP( lcrp );

	SpMVM_finish();

	return EXIT_SUCCESS;
}
