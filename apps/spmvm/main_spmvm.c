
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
	mat_idx_t i, errcount = 0;
	double mytol;
#endif

	int options = SPMVM_OPTION_AXPY; // TODO remote kernel immer axpy
	int kernels[] = {SPMVM_KERNEL_NOMPI,
		SPMVM_KERNEL_VECTORMODE,
		SPMVM_KERNEL_GOODFAITH,
		SPMVM_KERNEL_TASKMODE};
	int nKernels = sizeof(kernels)/sizeof(int);
	
	VECTOR_TYPE *nodeLHS; // lhs vector per node
	VECTOR_TYPE *nodeRHS; // rhs vector node

	SETUP_TYPE *setup;

	if (argc!=3) {
		fprintf(stderr,"Usage: spmvm.x <matrixPath> <matrixFormat>\n");
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

	mat_trait_t trait = SpMVM_stringToMatrixTrait(argv[2]);
	trait.flags |= SPM_PERMUTECOLUMNS;

	me     = SpMVM_init(argc,argv,options);       // basic initialization
	setup  = SpMVM_createSetup(matrixPath,&trait,1,SPM_GLOBAL,matrixFormats);
	nodeLHS= SpMVM_createVector(setup,VECTOR_TYPE_LHS,NULL);
	nodeRHS= SpMVM_createVector(setup,VECTOR_TYPE_RHS,rhsVal);

#ifdef CHECK	
	SETUP_TYPE *goldSetup;
	mat_trait_t goldTrait = SpMVM_createMatrixTrait(SPM_FORMAT_CRS,0,NULL);
	HOSTVECTOR_TYPE *goldLHS; // reference result
	HOSTVECTOR_TYPE *globLHS; // global lhs vector
	HOSTVECTOR_TYPE *globRHS; // global rhs vector
	goldSetup = SpMVM_createSetup (matrixPath,&goldTrait,1,SPM_GLOBAL,NULL);
	goldLHS = SpMVM_createVector(goldSetup,VECTOR_TYPE_LHS|VECTOR_TYPE_HOSTONLY,NULL);
	globRHS = SpMVM_createVector(goldSetup,VECTOR_TYPE_RHS|VECTOR_TYPE_HOSTONLY,rhsVal);
	globLHS = SpMVM_createVector(goldSetup,VECTOR_TYPE_LHS|VECTOR_TYPE_HOSTONLY,NULL);
	if (me==0)
		SpMVM_referenceSolver((CR_TYPE *)goldSetup->fullMatrix->data,globRHS->val,goldLHS->val,nIter,options);	
#endif	


	SpMVM_printEnvInfo();
	SpMVM_printMatrixInfo(setup->fullMatrix,strtok(basename(argv[optind]),"."),options);
	SpMVM_printHeader("Performance");

	for (kernel=0; kernel < nKernels; kernel++){

		time = SpMVM_solve(nodeLHS,setup,nodeRHS,kernels[kernel],nIter);

#ifdef CHECK
		if (time >= 0.)
			SpMVM_collectVectors(setup,nodeLHS,globLHS,kernel);
#endif

		if (me==0) {
			if (time < 0.) {
				SpMVM_printLine(SpMVM_kernelName(kernels[kernel]),NULL,"SKIPPED");
				continue;
			}
#ifdef CHECK
			errcount=0;
			for (i=0; i<setup->nRows; i++){
				mytol = EPSILON * ABS(goldLHS->val[i]) * 
					(((CR_TYPE *)(goldSetup->fullMatrix->data))->rowOffset[i+1]-((CR_TYPE *)(goldSetup->fullMatrix->data))->rowOffset[i]);
				if (REAL(ABS(goldLHS->val[i]-globLHS->val[i])) > mytol || 
						IMAG(ABS(goldLHS->val[i]-globLHS->val[i])) > mytol){
					printf( "PE%d: error in row %d: %.2e + %.2ei vs. %.2e +"
							"%.2ei (tol: %e, diff: %e)\n", me, i, REAL(goldLHS->val[i]),
							IMAG(goldLHS->val[i]),
							REAL(globLHS->val[i]),
							IMAG(globLHS->val[i]),
							mytol,REAL(ABS(goldLHS->val[i]-globLHS->val[i])));
					errcount++;
				}
			}
			if (errcount)
				SpMVM_printLine(SpMVM_kernelName(kernels[kernel]),NULL,"FAILED");
			else
				SpMVM_printLine(SpMVM_kernelName(kernels[kernel]),"GF/s","%f",
						FLOPS_PER_ENTRY*1.e-9*
						(double)setup->nNz/time,
						time*1.e3);
#else
			printf("%11s: %5.2f GF/s | %5.2f ms/it\n",
					SpMVM_kernelName(kernels[kernel]),
					FLOPS_PER_ENTRY*1.e-9*
					(double)setup->nNz/time,
					time*1.e3);
#endif
		}

		SpMVM_zeroVector(nodeLHS);

	}
	SpMVM_printFooter();


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
