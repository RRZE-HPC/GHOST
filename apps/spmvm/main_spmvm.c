
#include <spmvm.h>
#include <spmvm_util.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <libgen.h>

#define CHECK // compare with reference solution


static mat_data_t rhsVal (int i) 
{
#ifdef COMPLEX
	return i+1.0 + I*(i+1.5);
#else
	return i+1.0 ;
#endif
}

int main( int argc, char* argv[] ) 
{

	int me, kernel, nIter = 100;
	double time;

#ifdef CHECK
	unsigned int i, errcount = 0;
	double mytol;
#endif

	int options = SPMVM_OPTION_AXPY;
	int kernels[] = {SPMVM_KERNEL_NOMPI,
		SPMVM_KERNEL_VECTORMODE,
		SPMVM_KERNEL_GOODFAITH,
		SPMVM_KERNEL_TASKMODE};
	int nKernels = sizeof(kernels)/sizeof(int);
	
	VECTOR_TYPE *nodeLHS; // lhs vector per node
	VECTOR_TYPE *nodeRHS; // rhs vector node

	MATRIX_TYPE *matrix;

	if (argc!=2) {
		fprintf(stderr,"Usage: spmvm.x <matrixPath>\n");
		exit(EXIT_FAILURE);
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

	me     = SpMVM_init(argc,argv,options);       // basic initialization
	matrix = SpMVM_createMatrix(matrixPath,SPM_FORMAT_GLOB_BJDS,matrixFormats);
	nodeLHS= SpMVM_createVector(matrix,VECTOR_TYPE_LHS,NULL);
	nodeRHS= SpMVM_createVector(matrix,VECTOR_TYPE_RHS,rhsVal);

#ifdef CHECK	
	MATRIX_TYPE *goldMatrix;
	HOSTVECTOR_TYPE *goldLHS; // reference result
	HOSTVECTOR_TYPE *globLHS; // global lhs vector
	HOSTVECTOR_TYPE *globRHS; // global rhs vector
	goldMatrix = SpMVM_createMatrix (matrixPath,SPM_FORMAT_GLOB_CRS,NULL);
	goldLHS = SpMVM_createVector(goldMatrix,VECTOR_TYPE_LHS|VECTOR_TYPE_HOSTONLY,NULL);
	globRHS = SpMVM_createVector(goldMatrix,VECTOR_TYPE_RHS|VECTOR_TYPE_HOSTONLY,rhsVal);
	globLHS = SpMVM_createVector(goldMatrix,VECTOR_TYPE_LHS|VECTOR_TYPE_HOSTONLY,NULL);
	if (me==0)
		SpMVM_referenceSolver((CR_TYPE *)goldMatrix->matrix,globRHS->val,goldLHS->val,nIter,options);	
#endif	


	SpMVM_printEnvInfo();
	SpMVM_printMatrixInfo(matrix,strtok(basename(argv[optind]),"_."),options);

	for (kernel=0; kernel < nKernels; kernel++){

		time = SpMVM_solve(nodeLHS,matrix,nodeRHS,kernels[kernel],nIter);

#ifdef CHECK
		if (time >= 0.)
			SpMVM_collectVectors(matrix,nodeLHS,globLHS,kernel);

		if (me==0) {
			if (time < 0.) {
				printf("%11s: SKIPPED\n",
						SpMVM_kernelName(kernels[kernel]));
				continue;
			}
			errcount=0;
			for (i=0; i<matrix->nRows; i++){
				mytol = EPSILON * ABS(goldLHS->val[i]) * 
					(((CR_TYPE *)(goldMatrix->matrix))->rowOffset[i+1]-((CR_TYPE *)(goldMatrix->matrix))->rowOffset[i]);
				if (REAL(ABS(goldLHS->val[i]-globLHS->val[i])) > mytol || 
						IMAG(ABS(goldLHS->val[i]-globLHS->val[i])) > mytol){
					printf( "PE%d: error in row %u: %.2e + %.2ei vs. %.2e +"
							"%.2ei (tol: %e, diff: %e)\n", me, i, REAL(goldLHS->val[i]),
							IMAG(goldLHS->val[i]),
							REAL(globLHS->val[i]),
							IMAG(globLHS->val[i]),
							mytol,REAL(ABS(goldLHS->val[i]-globLHS->val[i])));
					errcount++;
				}
			}
			printf("%11s: %s @ %5.2f GF/s | %5.2f ms/it\n",
					SpMVM_kernelName(kernels[kernel]),
					errcount?"FAILURE":"SUCCESS",
					FLOPS_PER_ENTRY*1.e-9*
					(double)matrix->nNonz/time,
					time*1.e3);
		}
#else
		if (me==0) {
			printf("%11s: %5.2f GF/s | %5.2f ms/it\n",
					SpMVM_kernelName(kernels[kernel]),
					FLOPS_PER_ENTRY*1.e-9*
					(double)matrix->nNonz/time,
					time*1.e3);
		}
#endif

		SpMVM_zeroVector(nodeLHS);

	}


	SpMVM_freeVector( nodeLHS );
	SpMVM_freeVector( nodeRHS );
//	SpMVM_freeLCRP( lcrp );
	
#ifdef CHECK
	SpMVM_freeHostVector( globRHS );
	SpMVM_freeHostVector( goldLHS );
	SpMVM_freeHostVector( globLHS );
	//SpMVM_freeCRS( cr );
#endif

	SpMVM_finish();

	return EXIT_SUCCESS;

}
