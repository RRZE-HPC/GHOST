#include <spmvm.h>
#include <spmvm_util.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <libgen.h>




static real rhsVal (int i) 
{
#ifdef COMPLEX
	return i+1.0 + I*(i+1.5);
#else
	return i+1.0 ;
#endif
}

int main( int argc, char* argv[] ) 
{

	int me;

	int kernel;
	int errcount = 0;
	double mytol, time;
	int i; 

	VECTOR_TYPE*     nodeLHS; // lhs vector per node
	VECTOR_TYPE*     nodeRHS; // rhs vector node
	HOSTVECTOR_TYPE *goldLHS; // reference result
	HOSTVECTOR_TYPE *globRHS; // global rhs vector
	HOSTVECTOR_TYPE *globLHS; // global lhs vector

	CR_TYPE *cr;
	LCRP_TYPE *lcrp;

	int options = SPMVM_OPTION_NONE;
	int kernels[] = {SPMVM_KERNEL_VECTORMODE,SPMVM_KERNEL_GOODFAITH,SPMVM_KERNEL_TASKMODE};
	int nKernels = sizeof(kernels)/sizeof(int);

	if (argc!=2) {
		fprintf(stderr,"Usage: spmvm.x <matrixPath>\n");
		exit(EXIT_FAILURE);
	}

	char *matrixPath = argv[1];
	int nIter = 100;
	SPM_GPUFORMATS *matrixFormats = (SPM_GPUFORMATS *)malloc(sizeof(SPM_GPUFORMATS));;

#ifdef OPENCL
	matrixFormats->format[0] = SPM_GPUFORMAT_ELR;
	matrixFormats->format[1] = SPM_GPUFORMAT_ELR;
	matrixFormats->format[2] = SPM_GPUFORMAT_ELR;
	matrixFormats->T[0] = 1;
	matrixFormats->T[1] = 1;
	matrixFormats->T[2] = 1;
#else
	matrixFormats = NULL;
#endif


	me   = SpMVM_init(argc,argv,options);       // basic initialization
	cr   = SpMVM_createCRS (matrixPath);
	lcrp = SpMVM_distributeCRS (cr,matrixFormats);

	globRHS = SpMVM_createGlobalHostVector(cr->nCols,rhsVal);
	globLHS = SpMVM_createGlobalHostVector(cr->nCols,NULL);
	goldLHS = SpMVM_createGlobalHostVector(cr->nCols,NULL);
	nodeRHS = SpMVM_distributeVector(lcrp,globRHS);
	nodeLHS = SpMVM_newVector(lcrp->lnRows[me]);

	if (me==0)
		SpMVM_referenceSolver(cr,globRHS->val,goldLHS->val,nIter,options);	

	SpMVM_printEnvInfo();
	SpMVM_printMatrixInfo(lcrp,strtok(basename(argv[optind]),"_."),options);

	MPI_Barrier(MPI_COMM_WORLD);


	for (kernel=0; kernel < nKernels; kernel++){


		time = SpMVM_solve(nodeLHS,lcrp,nodeRHS,kernels[kernel],nIter);

		SpMVM_collectVectors(lcrp,nodeLHS,globLHS,kernel);

		if (me==0) {
			for (i=0; i<lcrp->nRows; i++){
				mytol = EPSILON * ABS(goldLHS->val[i]) * 
					(cr->rowOffset[i+1]-cr->rowOffset[i]);
				if (REAL(ABS(goldLHS->val[i]-globLHS->val[i])) > mytol || 
						IMAG(ABS(goldLHS->val[i]-globLHS->val[i])) > mytol){
					printf( "PE%d: error in row %i: %.2f + %.2fi vs. %.2f +"
							"%.2fi\n", me, i, REAL(goldLHS->val[i]),
							IMAG(goldLHS->val[i]),
							REAL(globLHS->val[i]),
							IMAG(globLHS->val[i]));
					errcount++;
				}
			}
			printf("Kernel %d %s @ %6.2f GF/s | %6.2f ms/it\n",
					kernel,errcount?"FAILED   ":"SUCCEEDED",
					FLOPS_PER_ENTRY*1.e-9*(double)nIter*
					(double)lcrp->nEnts/(time),(time)*1.e3/
					nIter);
		}

		SpMVM_zeroVector(nodeLHS);

	}


	SpMVM_freeVector( nodeLHS );
	SpMVM_freeVector( nodeRHS );
	SpMVM_freeHostVector( goldLHS );
	SpMVM_freeHostVector( globLHS );
	SpMVM_freeHostVector( globRHS );
	SpMVM_freeLCRP( lcrp );
	SpMVM_freeCRS( cr );

	SpMVM_finish();

	return EXIT_SUCCESS;

}

