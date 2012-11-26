#include <ghost.h>
#include <ghost_util.h>

#include <limits.h>
#include <libgen.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>

/** function to initialize the RHS vector **/
static ghost_mdat_t rhsVal (int i) {
	return i+1.0;
}

int main( int argc, char* argv[] ) {

	int me, nIter = 100;
	double time;
	int kernel = SPMVM_KERNEL_TASKMODE;
	int options = SPMVM_OPTION_PIN;

	MATRIX_TYPE     *matrix;    // local CRS portion
	ghost_vec_t     *nodeLHS; // lhs vector per node
	ghost_vec_t     *nodeRHS; // rhs vector node

	me      = ghost_init(argc,argv,options); // basic initialization
	matrix  = ghost_createMatrix(argv[1],SPM_FORMAT_DIST_CRS,NULL); // create CRS matrix 
	nodeRHS = ghost_createVector(matrix,GHOST_VEC_RHS,rhsVal); // RHS vec
	nodeLHS = ghost_createVector(matrix,GHOST_VEC_LHS,NULL);   // LHS vec (=0)

	ghost_printEnvInfo();
	ghost_printMatrixInfo(matrix,strtok(basename(argv[1]),"_."),options);

	time = ghost_solve(nodeLHS,matrix,nodeRHS,kernel,nIter);

	if (me == 0){
		printf("%s kernel @ %7.2f GF/s\n",ghost_kernelName(kernel),
				2.0e-9*(double)matrix->nnz/time);
	}

	ghost_freeVector( nodeLHS );
	ghost_freeVector( nodeRHS );
	//ghost_freeLCRP( lcrp );

	ghost_finish();

	return EXIT_SUCCESS;
}
