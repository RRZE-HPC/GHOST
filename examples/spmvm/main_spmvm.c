#include <spmvm_util.h>
#include <matricks.h>
#include <mpihelper.h>
#include <string.h>
#include <timing.h>
#include <math.h>
#include <stdlib.h>
#include <sys/times.h>
#include <unistd.h>
#include <omp.h>
#include <sched.h>

#ifdef OPENCL
#include <oclfun.h>
#endif

#ifdef LIKWID
#include <likwid.h>
#endif

#include <limits.h>
#include <getopt.h>
#include <libgen.h>

typedef struct {
	char matrixPath[PATH_MAX];
	char matrixName[PATH_MAX];
	int nIter;
#ifdef OPENCL
	SPM_GPUFORMATS matrixFormats;
	int devType;
#endif
} PROPS;


void usage() {
}


void getOptions(int argc,  char * const *argv, PROPS *p) {

	while (1) {
		static struct option long_options[] =
		{
			{"help", no_argument, 0, 'h'},
			{"matrixFormat",    required_argument, 0, 'f'},
			//{"workGroupSize",  required_argument, 0, 'w'},
			//{"nEigs",  required_argument, 0, 'e'},
			{"nIter",  required_argument, 0, 'i'},
			//{"dev",  required_argument, 0, 'd'},
			//{"nThreads",  required_argument, 0, 'T'},
			{0, 0, 0, 0}
		};
		/* getopt_long stores the option index here. */
		int option_index = 0;

		int c = getopt_long (argc, argv, "hf:w:e:i:d:T:",
				long_options, &option_index);

		/* Detect the end of the options. */
		if (c == -1)
			break;

		switch (c) {
			case 0:
				if (long_options[option_index].flag != 0)
					break;

			case 'h':
				usage();
				exit(0);
				break;


			case 'f':
				{
#ifdef OPENCL
					char *format;
					format = strtok(optarg,",");
					int i=0;

					while(format != NULL) {
						if (!strncasecmp(format,"ELR",3)) {
							p->matrixFormats.format[i] = SPM_GPUFORMAT_ELR;
							p->matrixFormats.T[i] = atoi(format+4);
						}
						if (!strncasecmp(format,"PJDS",4)) {
							p->matrixFormats.format[i] = SPM_GPUFORMAT_PJDS;
							p->matrixFormats.T[i] = atoi(format+5);
						}
						format = strtok(NULL,",");
						i++;
					}
#endif

					break;
				}

			case 'i':
				p->nIter = atoi(optarg);
				break;

			case '?':
				/* getopt_long already printed an error message. */
				break;

			default:
				abort ();
		}
	}
}

real rhsVal (int i) {
#ifdef COMPLEX
	return i+1.0 + I*(i+1.5);
#else
	return i+1.0 ;
#endif
}

int main( int argc, char* argv[] ) {

	int ierr;
	int me;

	int iteration;
	double start, end, dummy;
	int kernel;
	int errcount = 0;
	double mytol;
	int i,j; 

	VECTOR_TYPE*     nodeLHS; // lhs vector per node
	VECTOR_TYPE*     nodeRHS; // rhs vector node
	HOSTVECTOR_TYPE *goldLHS; // reference result
	HOSTVECTOR_TYPE *globRHS; // global rhs vector
	HOSTVECTOR_TYPE *globLHS; // global lhs vector

	CR_TYPE *cr;
	LCRP_TYPE *lcrp;

	SPMVM_KERNELS_SELECTED= 0;	
	SPMVM_KERNELS_SELECTED |= SPMVM_KERNEL_VECTORMODE;
	SPMVM_KERNELS_SELECTED |= SPMVM_KERNEL_GOODFAITH;
	SPMVM_KERNELS_SELECTED |= SPMVM_KERNEL_TASKMODE;
	
	SPMVM_OPTIONS = SPMVM_OPTION_NONE;


	PROPS props;
#ifdef OPENCL
	props.matrixFormats.format[0] = SPM_GPUFORMAT_ELR;
	props.matrixFormats.format[1] = SPM_GPUFORMAT_PJDS;
	props.matrixFormats.format[2] = SPM_GPUFORMAT_ELR;
	props.matrixFormats.T[0] = 1;
	props.matrixFormats.T[1] = 1;
	props.matrixFormats.T[2] = 1;
	props.devType = CL_DEVICE_TYPE_GPU;
#endif
	props.nIter = 100;

	getOptions(argc,argv,&props);
	if (argc==optind) {
		usage();
		exit(EXIT_FAILURE);
	}

	me   = SpMVM_init(argc,argv);       // basic initialization
	cr   = SpMVM_createCRS (argv[optind]);
	lcrp = SpMVM_distributeCRS (cr);

#ifdef OPENCL
	CL_uploadCRS ( lcrp, &props.matrixFormats);
#endif

	
	globRHS = SpMVM_createGlobalHostVector(cr->nCols,rhsVal);
	globLHS = SpMVM_createGlobalHostVector(cr->nCols,NULL);
	goldLHS = SpMVM_createGlobalHostVector(cr->nCols,NULL);
	nodeRHS = SpMVM_distributeVector(lcrp,globRHS);
	nodeLHS = newVector(lcrp->lnRows[me]);

	if (me==0)
	   SpMVM_referenceSolver(cr,globRHS->val,goldLHS->val,props.nIter);	

	SpMVM_printEnvInfo();
	SpMVM_printMatrixInfo(lcrp,strtok(basename(argv[optind]),"_."));

	MPI_Barrier(MPI_COMM_WORLD);


	for (kernel=0; kernel < SPMVM_NUMKERNELS; kernel++){

		if (!SpMVM_kernelValid(kernel,lcrp)) 
			continue; // Skip loop body if kernel does not make sense for used parametes

		MPI_Barrier(MPI_COMM_WORLD);
		if (me == 0)
			timing(&start,&dummy);


#ifdef LIKWID_MARKER
		char regionName[9];
		sprintf(regionName,"kernel %d",kernel);
#pragma omp parallel
		likwid_markerStartRegion(regionName);
#endif

		for( iteration = 0; iteration < props.nIter; iteration++ ) {
			SPMVM_KERNELS[kernel].kernel(nodeLHS, lcrp, nodeRHS);
			MPI_Barrier(MPI_COMM_WORLD);
		}
#ifdef LIKWID_MARKER
#pragma omp parallel
		likwid_markerStopRegion(regionName);
#endif

		if (me == 0)
			timing(&end,&dummy);

		if ( 0x1<<kernel & SPMVM_KERNELS_COMBINED)  {
			permuteVector(nodeLHS->val,lcrp->fullInvRowPerm,lcrp->lnRows[me]);
		} else if ( 0x1<<kernel & SPMVM_KERNELS_SPLIT ) {
			permuteVector(nodeLHS->val,lcrp->splitInvRowPerm,lcrp->lnRows[me]);
		}

		SpMVM_collectVectors(lcrp,nodeLHS,globLHS);

		if (me==0) {
			for (i=0; i<lcrp->nRows; i++){
				mytol = EPSILON * ABS(goldLHS->val[i]) * 
					(cr->rowOffset[i+1]-cr->rowOffset[i]);
				if (REAL(ABS(goldLHS->val[i]-globLHS->val[i])) > mytol || 
						IMAG(ABS(goldLHS->val[i]-globLHS->val[i])) > mytol){
					IF_DEBUG(1) {
						printf( "PE%d: error in row %i: %.2f + %.2fi vs. %.2f +"
							   "%.2fi\n", me, i, REAL(goldLHS->val[i]),
							   IMAG(goldLHS->val[i]),
							   REAL(globLHS->val[i]),
							   IMAG(globLHS->val[i]));
					}
					errcount++;
				}
			}
			printf("Kernel %2d: result is %s @ %7.2f GF/s | %7.2f ms/it\n",
					kernel,errcount?"WRONG  ":"CORRECT",
					FLOPS_PER_ENTRY*1.e-9*(double)props.nIter*
					(double)lcrp->nEnts/(end-start),(end-start)*1.e3/
					props.nIter);
		}

		zeroVector(nodeLHS);

	}


	freeVector( nodeLHS );
	freeVector( nodeRHS );
	freeHostVector( goldLHS );
	freeHostVector( globLHS );
	freeHostVector( globRHS );
	freeLcrpType( lcrp );
	freeCRMatrix( cr );

	SpMVM_finish();

	return EXIT_SUCCESS;

}
