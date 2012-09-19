#define _GNU_SOURCE

#include "spmvm.h"
#include "spmvm_util.h"
#include "matricks.h"
#include "mpihelper.h"
#include "kernel.h"
#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/param.h>
#include <sys/time.h>


#include <string.h>
#include <sched.h>
#include <omp.h>

#ifdef LIKWID
#include <likwid.h>
#endif

static Hybrid_kernel kernels[] = {
	{ &hybrid_kernel_0 },
	{ &hybrid_kernel_I },
	{ &hybrid_kernel_II },
	{ &hybrid_kernel_III },
};

static double wctime()
{
	struct timeval tp;
	double wctime;

	gettimeofday(&tp, NULL);
	wctime=(double) (tp.tv_sec + tp.tv_usec/1000000.0);

	return wctime; 
}

static int options;
#ifdef COMPLEX
typedef struct {
#ifdef DOUBLE
	double x;
	double y;
#endif
#ifdef SINGLE
	float x;
	float y;
#endif

} MPI_complex;

static void complAdd(MPI_complex *invec, MPI_complex *inoutvec, int *len)
{

	int i;
	MPI_complex c;

	for (i=0; i<*len; i++, invec++, inoutvec++){
		c.x = invec->x + inoutvec->x;
		c.y = invec->y + inoutvec->y;
		*inoutvec = c;
	}
}
#endif

int SpMVM_init(int argc, char **argv, int spmvmOptions)
{

	int me, req, prov;
	req = MPI_THREAD_MULTIPLE;
	MPI_safecall(MPI_Init_thread(&argc, &argv, req, &prov ));

	if (req != prov) {
		fprintf(stderr, "Required MPI threading level (%d) is not "
				"provided (%d)!\n",req,prov);
	}
	MPI_safecall(MPI_Comm_rank ( MPI_COMM_WORLD, &me ));

	setupSingleNodeComm();

#ifdef COMPLEX
#ifdef DOUBLE
	MPI_safecall(MPI_Type_contiguous(2,MPI_DOUBLE,&MPI_MYDATATYPE));
#endif
#ifdef SINGLE
	MPI_safecall(MPI_Type_contiguous(2,MPI_FLOAT,&MPI_MYDATATYPE));
#endif
	MPI_safecall(MPI_Type_commit(&MPI_MYDATATYPE));
	MPI_safecall(MPI_Op_create((MPI_User_function *)&complAdd,1,&MPI_MYSUM));
#endif

	if (spmvmOptions & SPMVM_OPTION_PIN || spmvmOptions & SPMVM_OPTION_PIN_SMT) {
		int nCores;
		int nPhysCores = getNumberOfPhysicalCores();
		if (spmvmOptions & SPMVM_OPTION_PIN)
			nCores = nPhysCores;
		else
			nCores = getNumberOfHwThreads();

		int offset = nPhysCores/getNumberOfRanksOnNode();
		omp_set_num_threads(nCores/getNumberOfRanksOnNode());
#pragma omp parallel
		{
			int error;
			int coreNumber;
	
			if (spmvmOptions & SPMVM_OPTION_PIN)
				coreNumber = omp_get_thread_num()+(offset*(getLocalRank()));
			else
				coreNumber = omp_get_thread_num()/2+(offset*(getLocalRank()))+(omp_get_thread_num()%2)*nPhysCores;

			//printf("p: %d, l: %d, t: %d, c: %d, o: %d\n",me,getLocalRank(),omp_get_thread_num(),coreNumber,offset);
			cpu_set_t cpu_set;
			CPU_ZERO(&cpu_set);
			CPU_SET(coreNumber, &cpu_set);

			error = sched_setaffinity((pid_t)0, sizeof(cpu_set_t), &cpu_set);

			if (error != 0) {
				printf("pinning thread to core %d failed (%d): %s\n", 
						coreNumber, error, strerror(error));
			}
		}
	}

	// TODO check options for plausability

#ifdef LIKWID_MARKER
	likwid_markerInit();
#endif


#ifdef OPENCL
	CL_init();
#endif

	options = spmvmOptions;
	if (options & SPMVM_OPTION_NO_SPLIT_KERNELS)
		options |= SPMVM_OPTION_NO_TASKMODE_KERNEL;

	return me;
}

void SpMVM_finish()
{

#ifdef LIKWID_MARKER
	likwid_markerClose();
#endif


#ifdef OPENCL
	CL_finish(options);
#endif

	MPI_Finalize();

}

VECTOR_TYPE *SpMVM_createVector(LCRP_TYPE *lcrp, int type, data_t (*fp)(int))
{
	VECTOR_TYPE* vec;
	size_t size_val;
	int i;
	int nRows;
	int me;

	MPI_safecall(MPI_Comm_rank ( MPI_COMM_WORLD, &me ));

	switch (type) {
		case VECTOR_TYPE_LHS:
			nRows = lcrp->lnRows[me];
			break;
		case VECTOR_TYPE_RHS:
		case VECTOR_TYPE_BOTH:
			nRows = lcrp->lnRows[me]+lcrp->halo_elements;
			break;
		default:
			SpMVM_abort("No valid type for vector (has to be one of VECTOR_TYPE_LHS/_RHS/_BOTH");
	}

	size_val = (size_t)( nRows * sizeof(data_t) );
	vec = (VECTOR_TYPE*) allocateMemory( sizeof( VECTOR_TYPE ), "vec");

	vec->val = (data_t*) allocateMemory( size_val, "vec->val");
	vec->nRows = nRows;

	if (fp) {
#pragma omp parallel for schedule(static)
		for (i=0; i<lcrp->lnRows[me]; i++) 
			vec->val[i] = fp(lcrp->lfRow[me]+i);
#pragma omp parallel for schedule(static)
		for (i=lcrp->lnRows[me]; i<nRows; i++) 
			vec->val[i] = fp(lcrp->lfRow[me]+i);
	}else {
#ifdef COMPLEX
#pragma omp parallel for schedule(static)
		for (i=0; i<lcrp->lnRows[me]; i++) vec->val[i] = 0.+I*0.;
#pragma omp parallel for schedule(static)
		for (i=lcrp->lnRows[me]; i<nRows; i++) vec->val[i] = 0.+I*0.;
#else
#pragma omp parallel for schedule(static)
		for (i=0; i<lcrp->lnRows[me]; i++) vec->val[i] = 0.;
#pragma omp parallel for schedule(static)
		for (i=lcrp->lnRows[me]; i<nRows; i++) vec->val[i] = 0.;
#endif
	}

#ifdef OPENCL
	int flag;
	switch (type) {
		case VECTOR_TYPE_LHS:
			if (options & SPMVM_OPTION_AXPY)
				flag = CL_MEM_READ_WRITE;
			else
				flag = CL_MEM_WRITE_ONLY;
			break;
		case VECTOR_TYPE_RHS:
			flag = CL_MEM_READ_ONLY;
			break;
		case VECTOR_TYPE_BOTH:
			flag = CL_MEM_READ_WRITE;
	}
	vec->CL_val_gpu = CL_allocDeviceMemoryMapped( size_val,vec->val,flag );
	CL_uploadVector(vec);
#endif

	return vec;
}

LCRP_TYPE * SpMVM_createCRS (char *matrixPath, void *deviceFormats)
{
	int me;
	CR_TYPE *cr;
	MM_TYPE *mm;
	LCRP_TYPE *lcrp;

	MPI_safecall(MPI_Comm_rank ( MPI_COMM_WORLD, &me ));
	cr = (CR_TYPE*) allocateMemory( sizeof( CR_TYPE ), "cr" );

	if (me == 0) { 
		// root process reads row pointers (parallel IO) oder entire matrix
		if (!isMMfile(matrixPath)){
			if (options & SPMVM_OPTION_SERIAL_IO)
				readCRbinFile(cr, matrixPath);
			else
				readCRrowsBinFile(cr, matrixPath);
		} else{
			mm = readMMFile( matrixPath);
			cr = convertMMToCRMatrix( mm );
			freeMMMatrix(mm);
		}
	}
	if (options & SPMVM_OPTION_SERIAL_IO)
		lcrp = setup_communication(cr, options);
	else
		lcrp = setup_communication_parallel(cr, matrixPath, options);

	if (deviceFormats == NULL) {
#ifdef OPENCL
		SpMVM_abort("Device matrix formats have to be passed to SPMVM_distributeCRS");
#endif
	}
#ifdef OPENCL
	SPM_GPUFORMATS *formats = (SPM_GPUFORMATS *)deviceFormats;
	CL_uploadCRS ( lcrp, formats, options);
#endif

	//	if (me==0)
	//	SpMVM_freeCRS(cr); FIXME

	return lcrp;

}

double SpMVM_solve(VECTOR_TYPE *res, LCRP_TYPE *lcrp, VECTOR_TYPE *invec, int kernel, int nIter)
{
	int it;
	int me;
	double time;
	FuncPrototype kernelFunc;
	MPI_safecall(MPI_Comm_rank ( MPI_COMM_WORLD, &me ));

	if ((kernel & SPMVM_KERNELS_SPLIT) && (options & SPMVM_OPTION_NO_SPLIT_KERNELS)) {
		IF_DEBUG(1) {
			if (me==0)
				fprintf(stderr,"Skipping the %s kernel because split kernels have not been configured.\n",SpMVM_kernelName(kernel));
		}
		return 0.; // kernel not selected
	}
	if ((kernel & SPMVM_KERNELS_COMBINED) && (options & SPMVM_OPTION_NO_COMBINED_KERNELS)) {
		IF_DEBUG(1) {
			if (me==0)
				fprintf(stderr,"Skipping the %s kernel because combined kernels have not been configured.\n",SpMVM_kernelName(kernel));
		}
		return 0.; // kernel not selected
	}
	if ((kernel & SPMVM_KERNEL_NOMPI)  && lcrp->nodes>1) {
		IF_DEBUG(1) {
			if (me==0)
				fprintf(stderr,"Skipping the %s kernel because there are multiple MPI processes.\n",SpMVM_kernelName(kernel));
		}
		return 0.; // non-MPI kernel
	} 
	if ((kernel & SPMVM_KERNEL_TASKMODE) &&  lcrp->threads==1) {
		IF_DEBUG(1) {
			if (me==0)
				fprintf(stderr,"Skipping the %s kernel because there is only one thread.\n",SpMVM_kernelName(kernel));
		}
		return 0.; // not enough threads
	}

	switch (kernel) {
		case SPMVM_KERNEL_NOMPI:
			kernelFunc = kernels[0].kernel;
			break;
		case SPMVM_KERNEL_VECTORMODE:
			kernelFunc = kernels[1].kernel;
			break;
		case SPMVM_KERNEL_GOODFAITH:
			kernelFunc = kernels[2].kernel;
			break;
		case SPMVM_KERNEL_TASKMODE:
			kernelFunc = kernels[3].kernel;
			break;
	}


	MPI_Barrier(MPI_COMM_WORLD);
	time = wctime();

	for( it = 0; it < nIter; it++ ) {
		kernelFunc(res, lcrp, invec, options);
		MPI_Barrier(MPI_COMM_WORLD);
	}
	time = wctime()-time;

	return time;
}
