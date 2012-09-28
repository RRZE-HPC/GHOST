/*
   This is a simple program which executes a couple of tests with self-created
   matrices.
   The main function logs to STDOUT and it returns the number of errattic kernels.
 */


#include <spmvm.h>
#include <spmvm_util.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <libgen.h>

static data_t rhsVal (int i) 
{
#ifdef COMPLEX
	return i+1.f + I*(i+1.5f);
#else
	return i+1.f ;
#endif
}

int main( int argc, char* argv[] ) 
{

	int me, kernel, nIter = 100;
	double time;

	int i;
	double mytol;

	int options;
	int kernels[] = {SPMVM_KERNEL_NOMPI,
		SPMVM_KERNEL_VECTORMODE,
		SPMVM_KERNEL_GOODFAITH,
		SPMVM_KERNEL_TASKMODE};
	int nKernels = sizeof(kernels)/sizeof(int);

	VECTOR_TYPE*     nodeLHS; // lhs vector per node
	VECTOR_TYPE*     nodeRHS; // rhs vector node

	LCRP_TYPE *lcrp;

	if (argc!=2) {
		fprintf(stderr,"Usage: test.x <matrixPath>\n");
		exit(SPMVM_NUMKERNELS);
	}

	char *matrixPath = argv[1];

	SPM_GPUFORMATS *matrixFormats = NULL;

#ifdef OPENCL
	matrixFormats = (SPM_GPUFORMATS *)malloc(sizeof(SPM_GPUFORMATS));
	matrixFormats->format[0] = SPM_GPUFORMAT_ELR;
	matrixFormats->format[1] = SPM_GPUFORMAT_ELR;
	matrixFormats->format[2] = SPM_GPUFORMAT_ELR;
	matrixFormats->T[0] = 1;
	matrixFormats->T[1] = 1;
	matrixFormats->T[2] = 1;
#endif

	for (options=0; options<1<<SPMVM_NUMOPTIONS; options++) {
	me   = SpMVM_init(argc,argv,options);       // basic initialization
	lcrp    = SpMVM_createCRS (matrixPath,matrixFormats);
	nodeLHS = SpMVM_createVector(lcrp,VECTOR_TYPE_LHS,NULL);
	nodeRHS = SpMVM_createVector(lcrp,VECTOR_TYPE_RHS,rhsVal);

	CR_TYPE *cr;
	HOSTVECTOR_TYPE *goldLHS; // reference result
	HOSTVECTOR_TYPE *globLHS; // global lhs vector
	HOSTVECTOR_TYPE *globRHS; // global rhs vector
	cr   = SpMVM_createGlobalCRS (matrixPath);
	goldLHS = SpMVM_createGlobalHostVector(cr->nRows,NULL);
	globRHS = SpMVM_createGlobalHostVector(lcrp->nRows,rhsVal);
	globLHS = SpMVM_createGlobalHostVector(cr->nRows,NULL);
	if (me==0)
		SpMVM_referenceSolver(cr,globRHS->val,goldLHS->val,nIter,options);	



	for (kernel=0; kernel < nKernels; kernel++){

		time = SpMVM_solve(nodeLHS,lcrp,nodeRHS,kernels[kernel],nIter);

		SpMVM_collectVectors(lcrp,nodeLHS,globLHS,kernel);

		if (me==0 && time>1e-16) {
			for (i=0; i<cr->nRows; i++){
				mytol = EPSILON * ABS(goldLHS->val[i]) * 
					(cr->rowOffset[i+1]-cr->rowOffset[i]);
				if (REAL(ABS(goldLHS->val[i]-globLHS->val[i])) > mytol || 
						IMAG(ABS(goldLHS->val[i]-globLHS->val[i])) > mytol){
					printf("0 %s failed with options %d\n",SpMVM_kernelName(kernels[kernel]),options);
					break;
				}
			}
			printf("1 %s succeeded with options %d\n",SpMVM_kernelName(kernels[kernel]),options);
		}

		SpMVM_zeroVector(nodeLHS);

	}


	SpMVM_freeVector( nodeLHS );
	SpMVM_freeVector( nodeRHS );
	SpMVM_freeLCRP( lcrp );

	SpMVM_freeHostVector( globRHS );
	SpMVM_freeHostVector( goldLHS );
	SpMVM_freeHostVector( globLHS );
	SpMVM_freeCRS( cr );

	}
	SpMVM_finish();


	return EXIT_SUCCESS;;

}

