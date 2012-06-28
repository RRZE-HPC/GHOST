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

#include <likwid.h>
#include <limits.h>
#include <getopt.h>

/* Global variables */
const char* SPM_FORMAT_NAME[]= {"ELR", "pJDS"};

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
	int kernels[] = {5,10,12};

	VECTOR_TYPE* hlpvec_out;    // lhs vector
	VECTOR_TYPE* hlpvec_in;     // rhs vector
	VECTOR_TYPE* resCR  = NULL; // reference result

	int iteration;
	
	double start, end, dummy, time_it_took;
	int numKernels = sizeof(kernels)/sizeof(int);
	int kernelIdx, kernel;

	jobmask = 5152;

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

	getMatrixPathAndName(argv[optind],props.matrixPath,props.matrixName);
	if (!props.matrixPath)
		myabort("No correct matrix specified! (no absolute file name and not present in $MATHOME)");

	required_threading_level = MPI_THREAD_MULTIPLE;
	ierr = MPI_Init_thread(&argc, &argv, required_threading_level, 
			&provided_threading_level );

	ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &me );


	LCRP_TYPE *lcrp = SpMVM_init ( props.matrixPath, &props.matrixFormats, &hlpvec_out, &hlpvec_in, &resCR);
	printMatrixInfo(lcrp,props.matrixName);



	MPI_Barrier(MPI_COMM_WORLD);
	for (kernelIdx=0; kernelIdx<numKernels; kernelIdx++){
		kernel = kernels[kernelIdx];

		/* Skip loop body if kernel does not make sense for used parametes */
		if (kernel==0 && lcrp->nodes>1) continue;      /* no MPI available */
		if (kernel>10 && kernel < 17 && lcrp->threads==1) continue; /* not enough threads */

		if (me == 0)
			timing(&start,&dummy);

		for( iteration = 0; iteration < props.nIter; iteration++ ) {
			HyK[kernel].kernel( iteration, hlpvec_out, lcrp, hlpvec_in);

			MPI_Barrier(MPI_COMM_WORLD);
		}

		if (me == 0) {
			timing(&end,&dummy);
			time_it_took = end-start;
		}

		if ( ((0x1<<kernel) & 503) ) {
			permuteVector(hlpvec_out->val,lcrp->fullInvRowPerm,lcrp->lnRows[me]);
		} else if ( ((0x1<<kernel) & 261640) ) {
			permuteVector(hlpvec_out->val,lcrp->splitInvRowPerm,lcrp->lnRows[me]);
		}

		int correct = Correctness_check( resCR, lcrp, hlpvec_out->val );
		if (me==0){
			printf("Kernel %2d: result is %s, %7.2f GF/s\n",kernel,correct?"CORRECT":"WRONG",2.0e-6*(double)props.nIter*(double)lcrp->nEnts/time_it_took);
		}

	}

	MPI_Barrier(MPI_COMM_WORLD);

	freeVector( hlpvec_out );
	freeVector( hlpvec_in );
	freeLcrpType( lcrp );
	freeVector( resCR );

	MPI_Finalize();

#ifdef OCLKERNEL
	CL_finish();
#endif

	return EXIT_SUCCESS;

}
