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

	VECTOR_TYPE* hlpvec_out;    // lhs vector
	VECTOR_TYPE* hlpvec_in;     // rhs vector
	VECTOR_TYPE* resCR  = NULL; // reference result

	VECTOR_TYPE *vold;
	VECTOR_TYPE *vnew;
	VECTOR_TYPE *evec;

	HOSTVECTOR_TYPE *r0;



	

	int iteration;
	
	double start, end, dummy, time_it_took;
	int kernelIdx, kernel;

	int kernels[] = {12};
	int numKernels = sizeof(kernels)/sizeof(int);
	jobmask = 0;

	for (i=0; i<numKernels; i++)
		jobmask |= 0x1<<kernels[i];

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


	LCRP_TYPE *lcrp = SpMVM_init ( props.matrixPath, &props.matrixFormats, &hlpvec_in, &resCR);
	printMatrixInfo(lcrp,props.matrixName);

	hlpvec_out = newVector( lcrp->lnRows[me] );

	if (me == 0) {
		r0 = newHostVector(lcrp->nRows);

		for (i=0; i<lcrp->nRows; i++) {
			r0->val[i] = (double)rand()/RAND_MAX;
		}

		normalize(r0->val,r0->nRows);
	} else {
		r0 = newHostVector(1);
	}
	double *zero = (double *)allocateMemory(lcrp->lnRows[me]*sizeof(double),"zero");
	for (i=0; i<lcrp->lnRows[me]; i++)
		zero[i] = 0;

	vnew = newVector(lcrp->lnRows[me]+lcrp->halo_elements); // = 0
	vold = newVector(lcrp->lnRows[me]+lcrp->halo_elements); // = r0
	evec = newVector(lcrp->lnRows[me]); // = r0

	memcpy(vnew->val,zero,sizeof(double)*lcrp->lnRows[me]);

	ierr = MPI_Scatterv ( r0->val, lcrp->lnRows, lcrp->lfRow, MPI_DOUBLE, 
			vold->val, lcrp->lnRows[me], MPI_DOUBLE, 0, MPI_COMM_WORLD );
	ierr = MPI_Scatterv ( r0->val, lcrp->lnRows, lcrp->lfRow, MPI_DOUBLE, 
			evec->val, lcrp->lnRows[me], MPI_DOUBLE, 0, MPI_COMM_WORLD );

	CL_copyHostToDevice(vnew->CL_val_gpu,vnew->val,lcrp->lnRows[me]*sizeof(double));
	CL_copyHostToDevice(vold->CL_val_gpu,vold->val,lcrp->lnRows[me]*sizeof(double));
	CL_copyHostToDevice(evec->CL_val_gpu,evec->val,lcrp->lnRows[me]*sizeof(double));

	double res=0.,totalRes=0.;
	CL_dotprod(vold->CL_val_gpu,vold->CL_val_gpu,&res,lcrp->lnRows[me]);

	MPI_Reduce(&res, &totalRes,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

	if (me==0)
		printf("should be 1: %f\n",totalRes);


	double alpha=0., beta=0.;

	MPI_Barrier(MPI_COMM_WORLD);
	for (kernelIdx=0; kernelIdx<numKernels; kernelIdx++){
		kernel = kernels[kernelIdx];

		/* Skip loop body if kernel does not make sense for used parametes */
		if (kernel==0 && lcrp->nodes>1) continue;      /* no MPI available */
		if (kernel>10 && kernel < 17 && lcrp->threads==1) continue; /* not enough threads */

	/*vecScalOCL(bufs.vnew, -(*beta), bufs.vnew, CLmatrix.nRows, CLinfo);
	SpMVOCL(CLmatrix, bufs.vold, bufs.vnew, CLinfo);
	scalarProdOCL(bufs.vold, bufs.vnew, alpha, CLmatrix.nRows,CLmatrix.nRowsPadded,CLinfo);
	axpyOCL(bufs.vnew, bufs.vold, -(*alpha), CLmatrix.nRows, CLinfo);
	scalarProdOCL(bufs.vnew, bufs.vnew, beta, CLmatrix.nRows, CLmatrix.nRowsPadded,CLinfo);
	*beta = sqrt(*beta);
	vecScalOCL(bufs.vnew, ((real)1.0/(*beta)), bufs.vnew, CLmatrix.nRows, CLinfo);*/

		//double *z = (double *)allocateMemory(sizeof(double)*props.nIter*props.nIter,"z");
		double *alphas = (double *)allocateMemory(sizeof(double)*props.nIter,"alphas");
		double *betas = (double *)allocateMemory(sizeof(double)*props.nIter,"betas");
		double *falphas = (double *)allocateMemory(sizeof(double)*props.nIter,"falphas");
		double *fbetas = (double *)allocateMemory(sizeof(double)*props.nIter,"fbetas");
		
		cl_mem tmp;
		double *dtmp;
		int ferr;

		betas[0] = beta;	
		for( iteration = 0; iteration < props.nIter; iteration++ ) {
			if (me == 0)
				timing(&start,&dummy);

			CL_vecscal(vnew->CL_val_gpu,-beta,lcrp->lnRows[me]);	
			HyK[kernel].kernel( iteration, vnew, lcrp, vold);

			CL_dotprod(vold->CL_val_gpu,vnew->CL_val_gpu,&alpha,lcrp->lnRows[me]);

			MPI_Allreduce(MPI_IN_PLACE, &alpha,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

			CL_axpy(vnew->CL_val_gpu,vold->CL_val_gpu,-alpha,lcrp->lnRows[me]);
			CL_dotprod(vnew->CL_val_gpu,vnew->CL_val_gpu,&beta,lcrp->lnRows[me]);
			
			MPI_Allreduce(MPI_IN_PLACE, &beta,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

			beta=sqrt(beta);
			CL_vecscal(vnew->CL_val_gpu,1./beta,lcrp->lnRows[me]);
			CL_copyDeviceToHost(vnew->val,vnew->CL_val_gpu,vnew->nRows*sizeof(double));

			alphas[iteration] = alpha;
			betas[iteration+1] = beta;

			if (me == 0) {
				timing(&end,&dummy);
				time_it_took = end-start;
				printf("\rlanczos: %f ms, ",time_it_took*1e3);
			}
			if (me == 0)
				timing(&start,&dummy);

			memcpy(falphas,alphas,(iteration+1)*sizeof(double));
			memcpy(fbetas,betas,(iteration+1)*sizeof(double));

			int n = iteration+1;
			imtql1_(&n,falphas,fbetas,&ferr);
			if(ferr != 0)
				fprintf(stderr,"> Error: the %d. eigenvalue could not be determined\n",ferr);
			if (me == 0) {
				timing(&end,&dummy);
				time_it_took = end-start;
				printf("imtql: %f ms, evmin: %f",time_it_took*1e3,falphas[0]);
			}

			dtmp = vnew->val;
			vnew->val = vold->val;
			vold->val = dtmp;
			tmp = vnew->CL_val_gpu;
			vnew->CL_val_gpu = vold->CL_val_gpu;
			vold->CL_val_gpu = tmp;
		}
		printf("\n");

		/*if ( ((0x1<<kernel) & 503) ) {
			permuteVector(hlpvec_out->val,lcrp->fullInvRowPerm,lcrp->lnRows[me]);
		} else if ( ((0x1<<kernel) & 261640) ) {
			permuteVector(hlpvec_out->val,lcrp->splitInvRowPerm,lcrp->lnRows[me]);
		}

		int correct = Correctness_check( resCR, lcrp, hlpvec_out->val );
		if (me==0){
			printf("Kernel %2d: result is %s, %7.2f GF/s\n",kernel,correct?"CORRECT":"WRONG",2.0e-6*(double)props.nIter*(double)lcrp->nEnts/time_it_took);
		}*/

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
