#include <ghost.h>
#include <ghost_util.h>
#include "lanczos.h"
#include <omp.h>


#ifdef LIKWID
#include <likwid.h>
#endif
#include <mpi.h>
#include <limits.h>
#include <getopt.h>
#include <libgen.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef OPENCL
static cl_kernel axpyKernel;
static cl_kernel vecscalKernel;
static cl_kernel dotprodKernel;
#endif

#define KERNEL SPMVM_KERNEL_VECTORMODE


typedef struct {
	char matrixPath[PATH_MAX];
	char matrixName[PATH_MAX];
	int nIter;
	SPM_GPUFORMATS *matrixFormats;
} PROPS;

static int converged(ghost_mdat_t evmin)
{
	static ghost_mdat_t oldevmin = -1e9;

	int converged = ABS(evmin-oldevmin) < 1e-9;
	//printf("%f %f %d\n",evmin,oldevmin,converged);

	oldevmin = evmin;

	return converged;
}

static void usage()
{
}

static void getOptions(int argc,  char * const *argv, PROPS *p)
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
							p->matrixFormats->format[i] = SPM_GPUFORMAT_ELR;
							p->matrixFormats->T[i] = atoi(format+4);
						}
						if (!strncasecmp(format,"PJDS",4)) {
							p->matrixFormats->format[i] = SPM_GPUFORMAT_PJDS;
							p->matrixFormats->T[i] = atoi(format+5);
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


static void dotprod(ghost_vec_t *v1, ghost_vec_t *v2, ghost_mdat_t *res, int n)
{

#ifdef OPENCL
	size_t localSize = CL_getLocalSize(dotprodKernel);
	int resVecSize = (n/localSize)+1;
	int i;
	*res = 0.0;

	ghost_vec_t *tmp = ghost_newVector(resVecSize*sizeof(ghost_mdat_t));

	CL_safecall(clSetKernelArg(dotprodKernel,0,sizeof(cl_mem),&v1->CL_val_gpu));
	CL_safecall(clSetKernelArg(dotprodKernel,1,sizeof(cl_mem),&v2->CL_val_gpu));
	CL_safecall(clSetKernelArg(dotprodKernel,2,sizeof(cl_mem),
				&tmp->CL_val_gpu));
	CL_safecall(clSetKernelArg(dotprodKernel,3,sizeof(int),&n));
	CL_safecall(clSetKernelArg(dotprodKernel,4,sizeof(ghost_mdat_t)*localSize,NULL));

	CL_enqueueKernel(dotprodKernel);

	CL_downloadVector(tmp);

	for(i = 0; i < resVecSize; ++i) {
		*res += tmp->val[i];
	}
	ghost_freeVector(tmp);
#else
	int i;
	ghost_mdat_t sum = 0;
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

static void axpy(ghost_vec_t *v1, ghost_vec_t *v2, ghost_mdat_t s, int n)
{

#ifdef OPENCL
	CL_safecall(clSetKernelArg(axpyKernel,0,sizeof(cl_mem),&v1->CL_val_gpu));
	CL_safecall(clSetKernelArg(axpyKernel,1,sizeof(cl_mem),&v2->CL_val_gpu));
	CL_safecall(clSetKernelArg(axpyKernel,2,sizeof(ghost_mdat_t),&s));
	CL_safecall(clSetKernelArg(axpyKernel,3,sizeof(int),&n));

	CL_enqueueKernel(axpyKernel);
#else
	int i;
#pragma omp parallel
	{
#ifdef LIKWID_MARKER_FINE
		likwid_markerStartRegion("axpy");
#endif


#pragma omp for private(i)
		for (i=0; i<n; i++)
			v1->val[i] = v1->val[i] + s*v2->val[i];

#ifdef LIKWID_MARKER_FINE
		likwid_markerStopRegion("axpy");
#endif

	}
#endif
}

static void vecscal(ghost_vec_t *vec, ghost_mdat_t s, int n)
{

#ifdef OPENCL
	CL_safecall(clSetKernelArg(vecscalKernel,0,sizeof(cl_mem),
				&vec->CL_val_gpu));
	CL_safecall(clSetKernelArg(vecscalKernel,1,sizeof(ghost_mdat_t),&s));
	CL_safecall(clSetKernelArg(vecscalKernel,2,sizeof(int),&n));

	CL_enqueueKernel(vecscalKernel);	
#else
	int i;
#pragma omp parallel
	{

#ifdef LIKWID_MARKER_FINE
		likwid_markerStartRegion("vecscal");
#endif

#pragma omp for private(i)
		for (i=0; i<n; i++)
			vec->val[i] = s*vec->val[i];

#ifdef LIKWID_MARKER_FINE
		likwid_markerStopRegion("vecscal");
#endif
	}

#endif
}


static void lanczosStep(LCRP_TYPE *lcrp, ghost_vec_t *vnew, ghost_vec_t *vold,
		ghost_mdat_t *alpha, ghost_mdat_t *beta, int me)
{
	vecscal(vnew,-*beta,lcrp->lnrows[me]);
	ghost_solve(vnew, lcrp, vold, KERNEL, 1);
	dotprod(vnew,vold,alpha,lcrp->lnrows[me]);
	MPI_Allreduce(MPI_IN_PLACE,alpha,1,MPI_MYDATATYPE,MPI_MYSUM,MPI_COMM_WORLD);
	axpy(vnew,vold,-(*alpha),lcrp->lnrows[me]);
	dotprod(vnew,vnew,beta,lcrp->lnrows[me]);
	MPI_Allreduce(MPI_IN_PLACE, beta,1,MPI_MYDATATYPE,MPI_MYSUM,MPI_COMM_WORLD);
	*beta=SQRT(*beta);
	vecscal(vnew,1./(*beta),lcrp->lnrows[me]);
}

static ghost_mdat_t rhsVal (int i)
{
	return i+1.0;
}

int main( int argc, char* argv[] )
{

	int me;


	ghost_vec_t *vold;
	ghost_vec_t *vnew;
	ghost_vec_t *evec;

	HOSTghost_vec_t *r0;

	int iteration;

	double start, end, tstart, tend, tacc, time_it_took;

	PROPS props;
	props.nIter = 1000;
	props.matrixFormats = (SPM_GPUFORMATS *)malloc(sizeof(SPM_GPUFORMATS));
#ifdef OPENCL
	props.matrixFormats->format[0] = SPM_GPUFORMAT_ELR; 
	props.matrixFormats->format[1] = SPM_GPUFORMAT_ELR;
	props.matrixFormats->format[2] = SPM_GPUFORMAT_ELR;
	props.matrixFormats->T[0] = 1;
	props.matrixFormats->T[1] = 1;
	props.matrixFormats->T[2] = 1;
#else
	props.matrixFormats = NULL;
#endif

	getOptions(argc,argv,&props);
	if (argc==optind) {
		usage();
		exit(EXIT_FAILURE);
	}
	// keep result vector on devic and eperform y <- y + A*x
	int options = SPMVM_OPTION_KEEPRESULT | SPMVM_OPTION_AXPY;

	me      = ghost_init(argc,argv,options);       // basic initialization
	LCRP_TYPE *lcrp = ghost_createCRS ( argv[optind],props.matrixFormats);



	r0 = ghost_createGlobalHostVector(lcrp->nrows,rhsVal);
	ghost_normalizeHostVector(r0);


#ifdef OPENCL

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
	cl_program program = CL_registerProgram("/home/hpc/unrz/unrza317/proj/SpMVM"
			"/libspmvm/examples/lanczos/lanczoskernels.cl",opt);

	int err;
	axpyKernel = clCreateKernel(program,"axpyKernel",&err);
	dotprodKernel = clCreateKernel(program,"dotprodKernel",&err);
	vecscalKernel = clCreateKernel(program,"vecscalKernel",&err);
#endif

	vnew = ghost_newVector(lcrp->lnrows[me]+lcrp->halo_elements); // = 0
	vold = ghost_newVector(lcrp->lnrows[me]+lcrp->halo_elements); // = r0
	evec = ghost_newVector(lcrp->lnrows[me]); // = r0

	vold = ghost_distributeVector(lcrp,r0);
	evec = ghost_distributeVector(lcrp,r0);

	ghost_printEnvInfo();
	ghost_printMatrixInfo(lcrp,strtok(basename(argv[optind]),"_."),options);


	//ghost_mdat_t *z = (ghost_mdat_t *)malloc(sizeof(ghost_mdat_t)*props.nIter*props.nIter,"z");
	ghost_mdat_t *alphas  = (ghost_mdat_t *)malloc(sizeof(ghost_mdat_t)*props.nIter);
	ghost_mdat_t *betas   = (ghost_mdat_t *)malloc(sizeof(ghost_mdat_t)*props.nIter);
	ghost_mdat_t *falphas = (ghost_mdat_t *)malloc(sizeof(ghost_mdat_t)*props.nIter);
	ghost_mdat_t *fbetas  = (ghost_mdat_t *)malloc(sizeof(ghost_mdat_t)*props.nIter);

	int ferr;

	ghost_mdat_t alpha=0., beta=0.;
	betas[0] = beta;
	int n;


	for( iteration = 0, n=1; 
			iteration < props.nIter && !converged(REAL(falphas[0])); 
			iteration++, n++ ) {

		if (me == 0) { 
			printf("\r");
			tstart = omp_get_wtime();
			start = omp_get_wtime();
		}
#ifdef LIKWID_MARKER
#pragma omp parallel
		likwid_markerStartRegion("Lanczos step");
#endif

		lanczosStep(lcrp,vnew,vold,&alpha,&beta,me);
#ifdef OPENCL
		CL_copyHostToDevice(vnew->CL_val_gpu,vnew->val,lcrp->lnrows[me]);
#endif

#ifdef LIKWID_MARKER
#pragma omp parallel 
		{
		likwid_markerStopRegion("Lanczos step");
		likwid_markerStartRegion("Housekeeping");
		}
#endif
		ghost_swapVectors(vnew,vold);


		alphas[iteration] = alpha;
		betas[iteration+1] = beta;
		memcpy(falphas,alphas,n*sizeof(ghost_mdat_t));
		memcpy(fbetas,betas,n*sizeof(ghost_mdat_t));

		if (me == 0) {
			end = omp_get_wtime();
			time_it_took = end-start;
			start = omp_get_wtime();
			printf("Lanczos: %6.2f ms, ",time_it_took*1e3);
		}
#ifdef LIKWID_MARKER
#pragma omp parallel
		{
		likwid_markerStopRegion("Housekeeping");
		likwid_markerStartRegion("Imtql");
		}
#endif

		imtql1_(&n,falphas,fbetas,&ferr); // TODO overlap
#ifdef LIKWID_MARKER
#pragma omp parallel
		likwid_markerStopRegion("Imtql");
#endif
		if(ferr != 0) printf("Error: the %d. ev could not be determined\n",ferr);

		if (me == 0) {
			end = omp_get_wtime();
			time_it_took = end-start;
			printf("imtql: %6.2f ms, ",time_it_took*1e3);
			tend = omp_get_wtime();
			tacc += tend-tstart;
			printf("e: %6.2f ",	REAL(falphas[0]));
			fflush(stdout);
		}

	}
	if (me==0) {
		printf("| total: %.2f ms, %d iterations\n",tacc*1e3,iteration);
	}



	ghost_freeHostVector( r0 );
	ghost_freeVector( vold );
	ghost_freeVector( vnew );
	ghost_freeVector( evec );
	ghost_freeLCRP( lcrp );

	ghost_finish();

	return EXIT_SUCCESS;

}
