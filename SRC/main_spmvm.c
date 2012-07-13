#include "spmvm_util.h"
#include "matricks.h"
#include "mpihelper.h"
#include <string.h>
#include "timing.h"
#include <math.h>
#include <stdlib.h>
#include <sys/times.h>
#include <unistd.h>
#include <omp.h>
#include <sched.h>
#include <oclfun.h>

#include <limits.h>
#include <getopt.h>
#include <libgen.h>

/* Global variables */
const char* SPM_FORMAT_NAME[]= {"ELR", "pJDS"};
//int SPMVM_OPTIONS = 0;

typedef struct {
	char matrixPath[PATH_MAX];
	char matrixName[PATH_MAX];
	MATRIX_FORMATS matrixFormats;
	//int wgSize;
	//int nEigs;
	int nIter;
	int devType;
	//int dev;
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
					char *format;
					format = strtok(optarg,",");
					int i=0;

					while(format != NULL) {
						if (!strncasecmp(format,"ELR",3)) {
							p->matrixFormats.format[i] = SPM_FORMAT_ELR;
							p->matrixFormats.T[i] = atoi(format+4);
						}
						if (!strncasecmp(format,"PJDS",4)) {
							p->matrixFormats.format[i] = SPM_FORMAT_PJDS;
							p->matrixFormats.T[i] = atoi(format+5);
						}
						format = strtok(NULL,",");
						i++;
					}

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


int main( int argc, char* argv[] ) {

	int ierr;
	int me;

	int required_threading_level;
	int provided_threading_level;

	int i,j; 

	int kernels[] = {2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17};
	int numKernels = sizeof(kernels)/sizeof(int);
	JOBMASK = 0;

	for (i=0; i<numKernels; i++)
		JOBMASK |= 0x1<<kernels[i];

	VECTOR_TYPE*     nodeLHS; // lhs vector per node
	VECTOR_TYPE*     nodeRHS; // rhs vector node
	HOSTVECTOR_TYPE *goldLHS; // reference result
	HOSTVECTOR_TYPE *globRHS; // global rhs vector
	HOSTVECTOR_TYPE *globLHS; // global lhs vector

	int iteration;

	double start, end, dummy, time_it_took;
	int kernelIdx, kernel;
	int errcount = 0;
	double mytol;


	PROPS props;
	props.matrixFormats.format[0] = SPM_FORMAT_ELR;
	props.matrixFormats.format[1] = SPM_FORMAT_PJDS;
	props.matrixFormats.format[2] = SPM_FORMAT_ELR;
	props.matrixFormats.T[0] = 2;
	props.matrixFormats.T[1] = 2;
	props.matrixFormats.T[2] = 1;
	props.nIter = 1;
	props.devType = CL_DEVICE_TYPE_GPU;

	getOptions(argc,argv,&props);
	if (argc==optind) {
		usage();
		exit(EXIT_FAILURE);
	}

	getMatrixPath(argv[optind],props.matrixPath);
	if (!props.matrixPath)
		myabort("No correct matrix specified! (no absolute file name and not present in $MATHOME)");
	strcpy(props.matrixName,basename(props.matrixPath));

	required_threading_level = MPI_THREAD_MULTIPLE;
	ierr = MPI_Init_thread(&argc, &argv, required_threading_level, 
			&provided_threading_level );

	ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &me );

	SPMVM_OPTIONS = 0;
	SPMVM_OPTIONS |= SPMVM_OPTION_AXPY;


	CR_TYPE *cr = SpMVM_createCRS ( props.matrixPath);



	if (me==0) {
		globRHS = newHostVector( cr->nCols );
		goldLHS = newHostVector( cr->nCols );
		globLHS = newHostVector( cr->nCols );

		for (i=0; i<cr->nCols; i++) { 
			globRHS->val[i] = i+1;
			globLHS->val[i] = 0.;
			goldLHS->val[i] = 0.;
		}

		if (SPMVM_OPTIONS & SPMVM_OPTION_AXPY)
			for (iteration=0; iteration<props.nIter; iteration++)
				fortrancrsaxpy_(&(cr->nRows), &(cr->nEnts), goldLHS->val, globRHS->val, cr->val , cr->col, cr->rowOffset);
		else
			fortrancrs_(&(cr->nRows), &(cr->nEnts), goldLHS->val, globRHS->val, cr->val , cr->col, cr->rowOffset);

	} else {
		goldLHS = newHostVector(0);
		globRHS = newHostVector(0);
		globLHS = newHostVector(0);
	}

	LCRP_TYPE *lcrp = SpMVM_distributeCRS ( cr);
#ifdef OCLKERNEL
	CL_uploadCRS ( lcrp, &props.matrixFormats);
#endif

	nodeRHS = SpMVM_distributeVector(lcrp,globRHS);
	nodeLHS = newVector( lcrp->lnRows[me] );

	SpMVM_printMatrixInfo(lcrp,props.matrixName);


	MPI_Barrier(MPI_COMM_WORLD);

	for (kernelIdx=0; kernelIdx<numKernels; kernelIdx++){
		kernel = kernels[kernelIdx];

		/* Skip loop body if kernel does not make sense for used parametes */
		if (kernel==0 && lcrp->nodes>1) continue;      /* no MPI available */
		if (kernel>10 && kernel < 17 && lcrp->threads==1) continue; /* not enough threads */

		if (me == 0)
			timing(&start,&dummy);

		for( iteration = 0; iteration < props.nIter; iteration++ ) {
			HyK[kernel].kernel( iteration, nodeLHS, lcrp, nodeRHS);

			MPI_Barrier(MPI_COMM_WORLD);
		}

		if (me == 0) {
			timing(&end,&dummy);
			time_it_took = end-start;
		}

		if ( ((0x1<<kernel) & 503) ) {
			permuteVector(nodeLHS->val,lcrp->fullInvRowPerm,lcrp->lnRows[me]);
		} else if ( ((0x1<<kernel) & 261640) ) {
			permuteVector(nodeLHS->val,lcrp->splitInvRowPerm,lcrp->lnRows[me]);
		}

		MPI_Gatherv(nodeLHS->val,lcrp->lnRows[me],MPI_DOUBLE,globLHS->val,lcrp->lnRows,lcrp->lfRow,MPI_DOUBLE,0,MPI_COMM_WORLD);

		if (me==0) {
			for (i=0; i<lcrp->lnRows[me]; i++){
				mytol = EPSILON * (1.0 + fabs(goldLHS->val[i]) ) ;
				if (fabs(goldLHS->val[i]-globLHS->val[i]) > mytol){
					IF_DEBUG(1) {
						printf( "PE%d: error in row %i: (|%e-%e|=%e)\n", me, i, goldLHS->val[i], globLHS->val[i],goldLHS->val[i]-globLHS->val[i]);
					}
					errcount++;
				}
			}
			printf("Kernel %2d: result is %s @ %7.2f GF/s\n",kernel,errcount?"WRONG":"CORRECT",2.0e-9*(double)props.nIter*(double)lcrp->nEnts/time_it_took);
		}
		zeroVector(nodeLHS);

	}


	freeVector( nodeLHS );
	freeVector( nodeRHS );
	freeHostVector( goldLHS );
	freeHostVector( globLHS );
	freeHostVector( globRHS );
	freeLcrpType( lcrp );

	MPI_Finalize();

#ifdef OCLKERNEL
	CL_finish();
#endif

	return EXIT_SUCCESS;

}
