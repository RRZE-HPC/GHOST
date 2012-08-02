#include "spmvm_util.h"
#include "spmvm_globals.h"
#include "matricks.h"
#include "mpihelper.h"

#ifdef OPENCL
#include "oclfun.h"
#include "oclmacros.h"
#endif

#include <likwid.h>

#include <mpi.h>
#include <limits.h>
#include <getopt.h>
#include <libgen.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef OPENCL
static cl_kernel axpyKernel;
static cl_kernel vecscalKernel;
static cl_kernel dotprodKernel;
static int localSz = 256;
#endif

typedef struct {
	char matrixPath[PATH_MAX];
	char matrixName[PATH_MAX];
	int nIter;
#ifdef OPENCL
	SPM_GPUFORMATS matrixFormats;
	int devType;
#endif
} PROPS;

void usage()
{
}

void getOptions(int argc,  char * const *argv, PROPS *p)
{

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

void vecscal(VECTOR_TYPE *vec, real s)
{
	
#ifdef OPENCL
	CL_safecall(clSetKernelArg(vecscalKernel,0,sizeof(cl_mem),
				&vec->CL_val_gpu));
	CL_safecall(clSetKernelArg(vecscalKernel,1,sizeof(real),&s));
	CL_safecall(clSetKernelArg(vecscalKernel,2,sizeof(int),&vec->nRows));

	CL_enqueueKernel(vecscalKernel,256);	
#else
	int i;
#pragma omp parallel
	{

#ifdef LIKWID_MARKER_FINE
		likwid_markerStartRegion("vecscal");
#endif

#pragma omp for private(i)
		for (i=0; i<vec->nRows; i++)
			vec->val[i] = s*vec->val[i];

#ifdef LIKWID_MARKER_FINE
		likwid_markerStopRegion("vecscal");
#endif
	}

#endif
}

void dotprod(VECTOR_TYPE *v1, VECTOR_TYPE *v2, real *res, int n)
{

#ifdef OPENCL
	int localSize = 256;
	int resVecSize = v1->nRows/localSize; 
	int i;
	*res = 0.0;

	VECTOR_TYPE *tmp = newVector(resVecSize*sizeof(real));

	CL_safecall(clSetKernelArg(dotprodKernel,0,sizeof(cl_mem),&v1->CL_val_gpu));
	CL_safecall(clSetKernelArg(dotprodKernel,1,sizeof(cl_mem),&v2->CL_val_gpu));
	CL_safecall(clSetKernelArg(dotprodKernel,2,sizeof(cl_mem),
				&tmp->CL_val_gpu));
	CL_safecall(clSetKernelArg(dotprodKernel,3,sizeof(int),&n));
	CL_safecall(clSetKernelArg(dotprodKernel,4,sizeof(real)*localSize,NULL));

	CL_enqueueKernel(dotprodKernel,localSize);

	CL_copyDeviceToHost(tmp->val,tmp->CL_val_gpu,resVecSize*sizeof(real));

	for(i = 0; i < resVecSize; ++i) {
		*res += tmp->val[i];
	}
	freeVector(tmp);
#else
	int i;
	real sum = 0;
#pragma omp parallel 
	{

#ifdef LIKWID_MARKER_FINE
		likwid_markerStartRegion("dotprod");
#endif
#pragma omp for private(i) reduction(+:sum)
		for (i=0; i<n; i++)
			sum += v1->val[i]*v2->val[i];
#ifdef LIKWID_MARKER_FINE
		likwid_markerStopRegion("dotprod");
#endif

	}
	*res = sum;
#endif
}

void axpy(VECTOR_TYPE *v1, VECTOR_TYPE *v2, real s)
{

#ifdef OPENCL
	CL_safecall(clSetKernelArg(axpyKernel,0,sizeof(cl_mem),&v1->CL_val_gpu));
	CL_safecall(clSetKernelArg(axpyKernel,1,sizeof(cl_mem),&v2->CL_val_gpu));
	CL_safecall(clSetKernelArg(axpyKernel,2,sizeof(real),&s));
	CL_safecall(clSetKernelArg(axpyKernel,3,sizeof(int),&v1->nRows));

	CL_enqueueKernel(axpyKernel,256);
#else
	int i;
#pragma omp parallel
	{
#ifdef LIKWID_MARKER_FINE
		likwid_markerStartRegion("axpy");
#endif

   
#pragma omp for private(i)
	for (i=0; i<v1->nRows; i++)
		v1->val[i] = v1->val[i] + s*v2->val[i];

#ifdef LIKWID_MARKER_FINE
		likwid_markerStopRegion("axpy");
#endif

	}
#endif
}

void lanczosStep(LCRP_TYPE *lcrp, int me, VECTOR_TYPE *vnew, VECTOR_TYPE *vold,
	   	real *alpha, real *beta, int kernel,  int iteration)
{
	vecscal(vnew,-*beta);
	HyK[kernel].kernel( iteration, vnew, lcrp, vold);
	dotprod(vnew,vold,alpha,lcrp->lnRows[me]);
	MPI_Allreduce(MPI_IN_PLACE,alpha,1,MPI_MYDATATYPE,MPI_MYSUM,MPI_COMM_WORLD);
	axpy(vnew,vold,-(*alpha));
	dotprod(vnew,vnew,beta,lcrp->lnRows[me]);
	MPI_Allreduce(MPI_IN_PLACE, beta,1,MPI_MYDATATYPE,MPI_MYSUM,MPI_COMM_WORLD);
	*beta=sqrt(*beta);
	vecscal(vnew,1./(*beta));
}

real rhsVal (int i)
{
	return i+1.0;
}

int main( int argc, char* argv[] )
{

	int ierr;
	int me;

	int i,j; 

	VECTOR_TYPE *vold;
	VECTOR_TYPE *vnew;
	VECTOR_TYPE *evec;

	HOSTVECTOR_TYPE *r0;

	int iteration;

	double start, end, dummy, tstart, tend, time_it_took, tacc = 0;
	int kernel;


	SPMVM_KERNELS = 0;	
	//SPMVM_KERNELS |= SPMVM_KERNEL_NOMPI;
	SPMVM_KERNELS |= SPMVM_KERNEL_VECTORMODE;
	SPMVM_KERNELS |= SPMVM_KERNEL_GOODFAITH;
	SPMVM_KERNELS |= SPMVM_KERNEL_TASKMODE;

	PROPS props;
	props.nIter = 100;
#ifdef OPENCL
	props.matrixFormats.format[0] = SPM_GPUFORMAT_ELR;
	props.matrixFormats.format[1] = SPM_GPUFORMAT_PJDS;
	props.matrixFormats.format[2] = SPM_GPUFORMAT_ELR;
	props.matrixFormats.T[0] = 1;
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
		myabort("No correct matrix specified! \
				(no absolute file name and not present in $MATHOME)");
	strcpy(props.matrixName,basename(props.matrixPath));

	me      = SpMVM_init(argc,argv);       // basic initialization
	CR_TYPE *cr = SpMVM_createCRS ( props.matrixPath);
	LCRP_TYPE *lcrp = SpMVM_distributeCRS ( cr);


	SPMVM_OPTIONS = SPMVM_OPTION_NONE;
	SPMVM_OPTIONS |= SPMVM_OPTION_KEEPRESULT; // keep result vector on device
	SPMVM_OPTIONS |= SPMVM_OPTION_AXPY;       // perform y <- y + A*x

	r0 = SpMVM_createGlobalHostVector(cr->nCols,rhsVal);
	normalize(r0->val,r0->nRows);


#ifdef OPENCL
	CL_uploadCRS ( lcrp, &props.matrixFormats);

#ifdef DOUBLE
#ifdef COMPLEX
	char *opt = " -DDOUBLE -DCOMPLEX ";
#else
	char *opt = " -DDOUBLE ";
#endif
#endif

#ifdef SINGLE
#ifdef COMPLEX
	char *opt = " -DSINGLE -DCOMPLEX ";
#else
	char *opt = " -DSINGLE ";
#endif
#endif
	cl_program program = CL_registerProgram("SRC/lanczoskernels.cl",opt);

	int err;
	axpyKernel = clCreateKernel(program,"axpyKernel",&err);
	dotprodKernel = clCreateKernel(program,"dotprodKernel",&err);
	vecscalKernel = clCreateKernel(program,"vecscalKernel",&err);
#endif

	real *zero = (real *)malloc(lcrp->lnRows[me]*sizeof(real));
	for (i=0; i<lcrp->lnRows[me]; i++)
		zero[i] = 0.;

	vnew = newVector(lcrp->lnRows[me]+lcrp->halo_elements); // = 0
	vold = newVector(lcrp->lnRows[me]+lcrp->halo_elements); // = r0
	evec = newVector(lcrp->lnRows[me]); // = r0

	memcpy(vnew->val,zero,sizeof(real)*lcrp->lnRows[me]);

	vold = SpMVM_distributeVector(lcrp,r0);
	evec = SpMVM_distributeVector(lcrp,r0);

	SpMVM_printMatrixInfo(lcrp,props.matrixName);

	MPI_Barrier(MPI_COMM_WORLD);
	for (kernel=0; kernel < SPMVM_NUMKERNELS; kernel++){

		/* Skip loop body if kernel does not make sense for used parametes */
		if (!(0x1<<kernel & SPMVM_KERNELS)) {
			continue; // kernel not selected
		}
		if ((0x1<<kernel & SPMVM_KERNEL_NOMPI)  && lcrp->nodes>1) {
			continue; // non-MPI kernel
		}
		if ((0x1<<kernel & SPMVM_KERNEL_TASKMODE) &&  lcrp->threads==1) {
			continue; // not enough threads
		}

		//real *z = (real *)malloc(sizeof(real)*props.nIter*props.nIter,"z");
		real *alphas  = (real *)malloc(sizeof(real)*props.nIter);
		real *betas   = (real *)malloc(sizeof(real)*props.nIter);
		real *falphas = (real *)malloc(sizeof(real)*props.nIter);
		real *fbetas  = (real *)malloc(sizeof(real)*props.nIter);

		real *dtmp;
		int ferr;

		tacc = 0;
		real alpha=0., beta=0.;
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
			event = CL_copyDeviceToHostNonBlocking(vnew->val, vnew->CL_val_gpu,
				   	lcrp->lnRows[me]*sizeof(real));
#endif
			swapVectors(vnew,vold);


			memcpy(falphas,alphas,(iteration)*sizeof(real));
			memcpy(fbetas,betas,(iteration)*sizeof(real));
			imtql1_(&n,falphas,fbetas,&ferr); // TODO overlap

			if(ferr != 0) {
				printf("Error: the %d. ev could not be determined\n",ferr);
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

#ifdef LIKWID_MARKER
			char regionName[] = "lanczosStep";
#pragma omp parallel
			likwid_markerStartRegion(regionName);
#endif
			lanczosStep(lcrp,me,vnew,vold,&alpha,&beta,kernel,iteration);
#ifdef LIKWID_MARKER
#pragma omp parallel
			likwid_markerStopRegion(regionName);
#endif
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
				printf("it: %6.2f ms, e: %6.2f ",(tend-tstart)*1e3,
						REAL(falphas[0]));
				fflush(stdout);
			}

		}
		if (me==0) {
			printf("| total: %.2f ms\n",tacc*1e3);
		}

	}


	freeHostVector( r0 );
	freeVector( vold );
	freeVector( vnew );
	freeVector( evec );
	freeLcrpType( lcrp );
	freeCRMatrix( cr );

	SpMVM_finish();

	return EXIT_SUCCESS;

}
