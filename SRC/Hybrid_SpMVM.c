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

int error_count, acc_error_count;

int coreId=2;

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

	double acc_cycles, acc_time;

	int i,j; 
	size_t ws;
	int kernels[] = {5,10,12};


	VECTOR_TYPE* hlpvec_out = NULL;
	VECTOR_TYPE* hlpvec_in  = NULL;


	int iteration;

	double time_it_took;
	int version;
	int numthreads;

	int ierr;
	int me;


	int job_flag;
	int this_one;
	jobmask = 5152;

	int required_threading_level;
	int provided_threading_level;


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

	MM_TYPE* mm = NULL;
	CR_TYPE* cr = NULL;

	VECTOR_TYPE* rhsVec = NULL;
	VECTOR_TYPE* resCR  = NULL;


	SpMVM_init (argc,argv, props.matrixPath, &props.matrixFormats, jobmask);
	if (me == 0){

		if (!isMMfile(props.matrixPath)){
			/* binary format *************************************/
			cr = (CR_TYPE*) allocateMemory( sizeof( CR_TYPE ), "cr" );
			bin_read_cr(cr, props.matrixPath);
		} else{
			/* ascii format *************************************/
			mm = readMMFile( props.matrixPath, 0.0 );
			cr = convertMMToCRMatrix( mm );
			bin_write_cr(cr, props.matrixName);
			freeMMMatrix(mm);
		}


		/* convert column indices in CRS format to FORTRAN-numbering, required for CPU kernel */
		crColIdToFortran(cr);


		rhsVec = newVector( cr->nCols );
		resCR = newVector( cr->nCols );

		/* Initialisiere invec */
		for (i=0; i<cr->nCols; i++) rhsVec->val[i] = i+1;

		/* Serial CRS-multiplication to get reference result */
		fortrancrs_(&(cr->nRows), &(cr->nEnts), 
				resCR->val, rhsVec->val, cr->val , cr->col, cr->rowOffset);


	} else{

		/* Allokiere minimalen Speicher fuer Dummyversion der globalen Matrix */
		mm            = (MM_TYPE*) allocateMemory( sizeof(MM_TYPE), "mm" );
		cr            = (CR_TYPE*) allocateMemory( sizeof(CR_TYPE), "cr" );
		cr->nRows     = 0;
		cr->nEnts     = 1;
		cr->rowOffset = (int*)     allocateMemory( sizeof(int),     "rowOffset" );
		cr->col       = (int*)     allocateMemory( sizeof(int),     "col" );
		cr->val       = (double*)  allocateMemory( sizeof(double),  "val" );
		rhsVec = newVector( 1 );
		resCR  = newVector( 1 );
	}


	LCRP_TYPE *lcrp = setup_communication(cr, 1,&props.matrixFormats);
	MPI_Barrier(MPI_COMM_WORLD);

#ifdef OCLKERNEL
	if( jobmask & 503 ) { 
		CL_bindMatrixToKernel(lcrp->fullMatrix,lcrp->fullFormat,props.matrixFormats.T[SPM_KERNEL_FULL],SPM_KERNEL_FULL);
	} 
	if( jobmask & 261640 ) { // only if jobtype requires split computation
		CL_bindMatrixToKernel(lcrp->localMatrix,lcrp->localFormat,props.matrixFormats.T[SPM_KERNEL_LOCAL],SPM_KERNEL_LOCAL);
		CL_bindMatrixToKernel(lcrp->remoteMatrix,lcrp->remoteFormat,props.matrixFormats.T[SPM_KERNEL_REMOTE],SPM_KERNEL_REMOTE);
	}
#endif


	int pseudo_ldim = lcrp->lnRows[me]+lcrp->halo_elements ;


	hlpvec_out = newVector( lcrp->lnRows[me] );
	hlpvec_in = newVector( pseudo_ldim );  

#pragma omp parallel for schedule(static)
	for (i=0; i<lcrp->lnRows[me]; i++) 
		hlpvec_out->val[i] = -63.5;


	/* Placement of RHS Vector */
#pragma omp parallel for schedule(runtime)
	for( i = 0; i < pseudo_ldim; i++ ) hlpvec_in->val[i] = 0.0;
	
		/* Fill up halo with some markers */
	for (i=lcrp->lnRows[me]; i< pseudo_ldim; i++) 
		hlpvec_in->val[i] = 77.0;

	/* Scatter the input vector from the master node to all others */
	ierr = MPI_Scatterv ( rhsVec->val, lcrp->lnRows, lcrp->lfRow, MPI_DOUBLE, 
			hlpvec_in->val, lcrp->lnRows[me], MPI_DOUBLE, 0, MPI_COMM_WORLD );


#ifdef OCLKERNEL	
	size_t fullMemSize, localMemSize, remoteMemSize, 
		   totalFullMemSize = 0, totalLocalMemSize = 0, totalRemoteMemSize = 0;
	if( jobmask & 503 ) { 
		fullMemSize = getBytesize(lcrp->fullMatrix,lcrp->fullFormat)/(1024*1024);
		MPI_Reduce(&fullMemSize, &totalFullMemSize,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);

	} 
	if( jobmask & 261640 ) { // only if jobtype requires split computation
		localMemSize = getBytesize(lcrp->localMatrix,lcrp->localFormat)/(1024*1024);
		remoteMemSize = getBytesize(lcrp->remoteMatrix,lcrp->remoteFormat)/(1024*1024);
		MPI_Reduce(&localMemSize, &totalLocalMemSize,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(&remoteMemSize, &totalRemoteMemSize,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
	}
#endif	

	if(me==0){
		ws = ((lcrp->nRows+1)*sizeof(int) + lcrp->nEnts*(sizeof(double)+sizeof(int)))/(1024*1024);
		printf("-----------------------------------------------------\n");
		printf("-------         Statistics about matrix       -------\n");
		printf("-----------------------------------------------------\n");
		printf("Investigated matrix         : %12s\n", props.matrixName); 
		printf("Dimension of matrix         : %12.0f\n", (float)lcrp->nRows); 
		printf("Non-zero elements           : %12.0f\n", (float)lcrp->nEnts); 
		printf("Average elements per row    : %12.3f\n", (float)lcrp->nEnts/(float)lcrp->nRows); 
		printf("Working set             [MB]: %12lu\n", ws);
#ifdef OCLKERNEL	
		if( jobmask & 503 ) 
			printf("Device matrix (combin.) [MB]: %12lu\n", totalFullMemSize); 
		if( jobmask & 261640 ) {
			printf("Device matrix (local)   [MB]: %12lu\n", totalLocalMemSize); 
			printf("Device matrix (remote)  [MB]: %12lu\n", totalRemoteMemSize); 
			printf("Device matrix (loc+rem) [MB]: %12lu\n", totalLocalMemSize+totalRemoteMemSize); 
		}
#endif
		printf("-----------------------------------------------------\n");
		fflush(stdout);
	}





	MPI_Barrier(MPI_COMM_WORLD);
	double start, end, dummy;


	int numKernels = sizeof(kernels)/sizeof(int);
	int kernelIdx, kernel;
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
	freeVector( rhsVec );
	freeVector( resCR );

	MPI_Finalize();

#ifdef OCLKERNEL
	CL_finish();
#endif

	return 0;

}
