#include "spmvm_util.h"
#include "spmvm_globals.h"

#ifdef OPENCL
#include "oclfun.h"
#endif

#include <likwid.h>

#include <mpi.h>
#include <limits.h>
#include <getopt.h>
#include <libgen.h>
#include <stdlib.h>
#include <stdio.h>

/* Global variables */
const char* SPM_FORMAT_NAME[]= {"ELR", "pJDS"};

typedef struct {
	char matrixPath[PATH_MAX];
	char matrixName[PATH_MAX];
	int nIter;
#ifdef OPENCL
	MATRIX_FORMATS matrixFormats;
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

void lanczosStep(LCRP_TYPE *lcrp, int me, VECTOR_TYPE *vnew, VECTOR_TYPE *vold, double *alpha, double *beta, int kernel,  int iteration) {
	vecscal(vnew,-*beta);
	HyK[kernel].kernel( iteration, vnew, lcrp, vold);
	dotprod(vnew,vold,alpha,lcrp->lnRows[me]);
	MPI_Allreduce(MPI_IN_PLACE, alpha,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	axpy(vnew,vold,-(*alpha));
	dotprod(vnew,vnew,beta,lcrp->lnRows[me]);
	MPI_Allreduce(MPI_IN_PLACE, beta,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	*beta=sqrt(*beta);
	vecscal(vnew,1./(*beta));
}

double rhsVal (int i) {
	return i+1.0;
}

int main( int argc, char* argv[] ) {

	int ierr;
	int me;

	int i,j; 

	VECTOR_TYPE *vold;
	VECTOR_TYPE *vnew;
	VECTOR_TYPE *evec;

	HOSTVECTOR_TYPE *r0;

	int iteration;

	double start, end, dummy, tstart, tend, time_it_took, tacc = 0;
	int kernelIdx, kernel;

	int kernels[] = {5,12/*5,10,12*/};
	int numKernels = sizeof(kernels)/sizeof(int);
	JOBMASK = 0;

	for (i=0; i<numKernels; i++)
		JOBMASK |= 0x1<<kernels[i];

	PROPS props;
	props.nIter = 100;
#ifdef OPENCL
	props.matrixFormats.format[0] = SPM_FORMAT_ELR;
	props.matrixFormats.format[1] = SPM_FORMAT_PJDS;
	props.matrixFormats.format[2] = SPM_FORMAT_ELR;
	props.matrixFormats.T[0] = 2;
	props.matrixFormats.T[1] = 2;
	props.matrixFormats.T[2] = 1;
	props.devType = CL_DEVICE_TYPE_GPU;
	cl_mem tmp;
	cl_event event;
#endif

	getOptions(argc,argv,&props);
	if (argc==optind) {
		usage();
		exit(EXIT_FAILURE);
	}

	getMatrixPath(argv[optind],props.matrixPath);
	if (!props.matrixPath)
		myabort("No correct matrix specified! (no absolute file name and not present in $MATHOME)");
	strcpy(props.matrixName,basename(props.matrixPath));

	me      = SpMVM_init(argc,argv);       // basic initialization

#ifdef LIKDIW_MARKER
	likwid_markerInit();
#endif

#pragma omp parallel
#pragma omp master
	printf("Running with %d threads.\n",omp_get_num_threads());


	SPMVM_OPTIONS |= SPMVM_OPTION_KEEPRESULT; // keep result vector on device after spmvm
	SPMVM_OPTIONS |= SPMVM_OPTION_AXPY;       // performa y <- y + A*x

	//LCRP_TYPE *lcrp = SpMVM_init ( props.matrixPath, &props.matrixFormats, &hlpvec_in, &resCR);
	CR_TYPE *cr = SpMVM_createCRS ( props.matrixPath);
	int nnz = cr->nEnts;

	r0 = SpMVM_createGlobalHostVector(cr->nCols,rhsVal);
	normalize(r0->val,r0->nRows);

	LCRP_TYPE *lcrp = SpMVM_distributeCRS ( cr);

#ifdef OPENCL
	CL_uploadCRS ( lcrp, &props.matrixFormats);
#endif

	double *zero = (double *)malloc(lcrp->lnRows[me]*sizeof(double));
	for (i=0; i<lcrp->lnRows[me]; i++)
		zero[i] = 0.;

	vnew = newVector(lcrp->lnRows[me]+lcrp->halo_elements); // = 0
	vold = newVector(lcrp->lnRows[me]+lcrp->halo_elements); // = r0
	evec = newVector(lcrp->lnRows[me]); // = r0

	memcpy(vnew->val,zero,sizeof(double)*lcrp->lnRows[me]);

	vold = SpMVM_distributeVector(lcrp,r0);
	evec = SpMVM_distributeVector(lcrp,r0);

	SpMVM_printMatrixInfo(lcrp,props.matrixName);

	MPI_Barrier(MPI_COMM_WORLD);
	for (kernelIdx=0; kernelIdx<numKernels; kernelIdx++){

		kernel = kernels[kernelIdx];

		/* Skip loop body if kernel does not make sense for used parametes */
		if (kernel==0 && lcrp->nodes>1) continue;      /* no MPI available */
		if (kernel>10 && kernel < 17 && lcrp->threads==1) continue; /* not enough threads */


		//double *z = (double *)malloc(sizeof(double)*props.nIter*props.nIter,"z");
		double *alphas  = (double *)malloc(sizeof(double)*props.nIter);
		double *betas   = (double *)malloc(sizeof(double)*props.nIter);
		double *falphas = (double *)malloc(sizeof(double)*props.nIter);
		double *fbetas  = (double *)malloc(sizeof(double)*props.nIter);

		double *dtmp;
		int ferr;

		tacc = 0;
		double alpha=0., beta=0.;
		betas[0] = beta;
		int n;


		lanczosStep(lcrp,me,vnew,vold,&alpha,&beta,kernel,0);

		alphas[0] = alpha;
		betas[1] = beta;


		for( iteration = 1, n=1; iteration < props.nIter; iteration++, n++ ) {
			if (me == 0) { 
				printf("\r");
				timing(&start,&dummy);
			}
#ifdef OPENCL	
			event = CL_copyDeviceToHostNonBlocking( vnew->val, vnew->CL_val_gpu, lcrp->lnRows[me]*sizeof(double) );
#endif
			swapVectors(vnew,vold);


			memcpy(falphas,alphas,(iteration)*sizeof(double));
			memcpy(fbetas,betas,(iteration)*sizeof(double));
			imtql1_(&n,falphas,fbetas,&ferr); // TODO overlap

			if(ferr != 0) {
				printf("> Error: the %d. eigenvalue could not be determined\n",ferr);
			}
			if (me == 0) {
				timing(&end,&dummy);
				time_it_took = end-start;
				printf("imtql: %6.2f ms, ",time_it_took*1e3);
			}

#ifdef OPENCL
			clWaitForEvents(1,&event);
#endif

			if (me == 0) {
				timing(&start,&dummy);
				timing(&tstart,&dummy);
			
			}

			lanczosStep(lcrp,me,vnew,vold,&alpha,&beta,kernel,iteration);
			if (me == 0) {
				timing(&end,&dummy);
				time_it_took = end-start;
				printf("lcz: %6.2f ms, ",time_it_took*1e3);
			}

			alphas[iteration] = alpha;
			betas[iteration+1] = beta;

			if (me==0) {
				timing(&tend,&dummy);
				tacc += tend-tstart;
				printf("it: %6.2f ms, e: %6.2f ",(tend-tstart)*1e3,falphas[0]);
				fflush(stdout);
			}

		}
		if (me==0) {
			printf("| total: %.2f ms\n",tacc*1e3);
		}

	}


	MPI_Barrier(MPI_COMM_WORLD);

	freeHostVector( r0 );
	freeVector( vold );
	freeVector( vnew );
	freeVector( evec );
	freeLcrpType( lcrp );

#ifdef LIKWID_MARKER
	likwid_markerClose();
#endif

	MPI_Finalize();

#ifdef OPENCL
	CL_finish();
#endif

	return EXIT_SUCCESS;

}
