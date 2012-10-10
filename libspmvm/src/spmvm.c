#define _GNU_SOURCE

#include "spmvm.h"
#include "spmvm_util.h"
#include "matricks.h"

#ifdef MPI
#include <mpi.h>
#include "mpihelper.h"
#endif

#include "kernel.h"
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

static double wctime()
{
	struct timeval tp;
	double wctime;

	gettimeofday(&tp, NULL);
	wctime=(double) (tp.tv_sec + tp.tv_usec/1000000.0);

	return wctime; 
}

static int options;

#if defined(COMPLEX) && defined(MPI)
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
	int me;

#ifdef MPI
	int req, prov, init;

	req = MPI_THREAD_MULTIPLE; // TODO not if not all kernels configured

	MPI_safecall(MPI_Initialized(&init));
	if (!init) {
		MPI_safecall(MPI_Init_thread(&argc, &argv, req, &prov ));

		if (req != prov) {
			DEBUG_LOG(0,"Warning! Required MPI threading level (%d) is not "
					"provided (%d)!",req,prov);
		}
	}
	me = SpMVM_getRank();;

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

			if (spmvmOptions & SPMVM_OPTION_PIN_SMT)
				coreNumber = omp_get_thread_num()/2+(offset*(getLocalRank()))+(omp_get_thread_num()%2)*nPhysCores;
			else
				coreNumber = omp_get_thread_num()+(offset*(getLocalRank()));

			DEBUG_LOG(1,"Pinning thread %d to core %d",omp_get_thread_num(),coreNumber);
			cpu_set_t cpu_set;
			CPU_ZERO(&cpu_set);
			CPU_SET(coreNumber, &cpu_set);

			error = sched_setaffinity((pid_t)0, sizeof(cpu_set_t), &cpu_set);

			if (error != 0) {
				DEBUG_LOG(0,"Pinning thread to core %d failed (%d): %s", 
						coreNumber, error, strerror(error));
			}
		}
	}
#else
	UNUSED(argc);
	UNUSED(argv);
	me = 0;
	spmvmOptions |= SPMVM_OPTION_SERIAL_IO; // important for createMatrix()

#endif

	// TODO check options for plausability

#ifdef LIKWID_MARKER
	likwid_markerInit();
#endif


#ifdef OPENCL
	CL_init();
#endif

	options = spmvmOptions;

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

#ifdef MPI
	MPI_Finalize();
#endif

}

void *SpMVM_createVector(MATRIX_TYPE *matrix, int type, data_t (*fp)(int))
{

	data_t *val;
	int nRows;
	size_t size_val;


	if (matrix->format & SPM_FORMATS_GLOB)
	{
		size_val = (size_t)matrix->nRows*sizeof(data_t);
		val = (data_t*) allocateMemory( size_val, "vec->val");
		nRows = matrix->nRows;

		unsigned int i;
		if (fp) {
#pragma omp parallel for schedule(static)
			for (i=0; i<matrix->nRows; i++) 
				val[i] = fp(i);
		}else {
#ifdef COMPLEX
#pragma omp parallel for schedule(static)
			for (i=0; i<matrix->nRows; i++) val[i] = 0.+I*0.;
#else
#pragma omp parallel for schedule(static)
			for (i=0; i<matrix->nRows; i++) val[i] = 0.;
#endif
		}




	} 
	else 
	{
		LCRP_TYPE *lcrp = matrix->matrix;
		int i;
		int me = SpMVM_getRank();

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

		val = (data_t*) allocateMemory( size_val, "vec->val");
		nRows = nRows;

		DEBUG_LOG(1,"NUMA-aware allocation of vector with %d+%d rows",lcrp->lnRows[me],lcrp->halo_elements);

		if (fp) {
#pragma omp parallel for schedule(static)
			for (i=0; i<lcrp->lnRows[me]; i++) 
				val[i] = fp(lcrp->lfRow[me]+i);
#pragma omp parallel for schedule(static)
			for (i=lcrp->lnRows[me]; i<nRows; i++) 
				val[i] = fp(lcrp->lfRow[me]+i);
		}else {
#ifdef COMPLEX
#pragma omp parallel for schedule(static)
			for (i=0; i<lcrp->lnRows[me]; i++) val[i] = 0.+I*0.;
#pragma omp parallel for schedule(static)
			for (i=lcrp->lnRows[me]; i<nRows; i++) val[i] = 0.+I*0.;
#else
#pragma omp parallel for schedule(static)
			for (i=0; i<lcrp->lnRows[me]; i++) val[i] = 0.;
#pragma omp parallel for schedule(static)
			for (i=lcrp->lnRows[me]; i<nRows; i++) val[i] = 0.;
#endif
		}
	}


	if (type & VECTOR_TYPE_HOSTONLY) {
		HOSTVECTOR_TYPE* vec;
		vec = (HOSTVECTOR_TYPE*) allocateMemory( sizeof( VECTOR_TYPE ), "vec");
		vec->val = val;
		vec->nRows = nRows; 

		DEBUG_LOG(1,"Vector created successfully");

		return vec;
	} else {
		VECTOR_TYPE* vec;
		vec = (VECTOR_TYPE*) allocateMemory( sizeof( VECTOR_TYPE ), "vec");
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
			default:
				SpMVM_abort("No valid type for vector (has to be one of VECTOR_TYPE_LHS/_RHS/_BOTH");
		}
		vec->CL_val_gpu = CL_allocDeviceMemoryMapped( size_val,val,flag );
		CL_uploadVector(vec);
#endif
		vec->val = val;
		vec->nRows = nRows; 

		DEBUG_LOG(1,"Vector created successfully");

		return vec;
	}

}

MATRIX_TYPE *SpMVM_createMatrix(char *matrixPath, int format, void *deviceFormats) 
{
	MATRIX_TYPE *mat;
	CR_TYPE *cr;

	mat = (MATRIX_TYPE *)allocateMemory(sizeof(MATRIX_TYPE),"matrix");
	cr = (CR_TYPE*) allocateMemory( sizeof( CR_TYPE ), "cr" );

	if (SpMVM_getRank() == 0) 
	{ // root process reads row pointers (parallel IO) oder entire matrix
		if (!isMMfile(matrixPath)){
			if (options & SPMVM_OPTION_SERIAL_IO)
				readCRbinFile(cr, matrixPath);
			else
				readCRrowsBinFile(cr, matrixPath);
		} else{
			MM_TYPE *mm = readMMFile( matrixPath);
			cr = convertMMToCRMatrix( mm );
			freeMMMatrix(mm);
		}
	}

	if (format & SPM_FORMATS_DIST)
	{ // distributed matrix
#ifndef MPI
		ABORT("Creating a distributed matrix without MPI is not possible");
#endif
		LCRP_TYPE *lcrp;

		if (options & SPMVM_OPTION_SERIAL_IO)    // TODO w/o MPI always serial I/O
			lcrp = setup_communication(cr, options);
		else
			lcrp = setup_communication_parallel(cr, matrixPath, options);
		mat->matrix = lcrp;

#ifdef OPENCL
		if (deviceFormats == NULL) {
			ABORT("Device matrix formats have to be passed to SPMVM_distributeCRS!");
		}
		SPM_GPUFORMATS *formats = (SPM_GPUFORMATS *)deviceFormats;
		mat->devMatrix = CL_uploadCRS ( lcrp, formats, options);
#else
		UNUSED(deviceFormats);
#endif
	} else 
	{ // global matrix
		switch (format) {
			case SPM_FORMAT_GLOB_CRS:
				mat->matrix = cr;
				break;
			case SPM_FORMAT_GLOB_MICVEC:
				mat->matrix = CRStoMICVEC(cr);
				break;
			default:
				ABORT("No valid matrix format specified!");
		}

	}

	mat->format = format;
	mat->nNonz = cr->nEnts;
	mat->nRows = cr->nRows;
	mat->nCols = cr->nCols;

	DEBUG_LOG(1,"Matrix created successfully");

	return mat;
}


/*LCRP_TYPE * SpMVM_createCRS (char *matrixPath, void *deviceFormats)
{
	CR_TYPE *cr;
	MM_TYPE *mm;
	LCRP_TYPE *lcrp;

	cr = (CR_TYPE*) allocateMemory( sizeof( CR_TYPE ), "cr" );

	if (SpMVM_getRank() == 0) { 
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

#ifdef MPI
	if (options & SPMVM_OPTION_SERIAL_IO)    // TODO w/o MPI always serial I/O
		lcrp = setup_communication(cr, options);
	else
		lcrp = setup_communication_parallel(cr, matrixPath, options);
#else
	lcrp = SpMVM_CRtoLCRP(cr);
#endif


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

}*/

double SpMVM_solve(VECTOR_TYPE *res, MATRIX_TYPE *mat, VECTOR_TYPE *invec, 
		int kernel, int nIter)
{
	int it;
	double time = 0;
	char *name = SpMVM_kernelName(kernel);
	SpMVM_kernelFunc kernelFunc = NULL;

#ifndef MPI
	if (!(kernel & SPMVM_KERNEL_NOMPI)) {
		DEBUG_LOG(1,"Skipping the %s kernel because the library is built without MPI.",name);
		return 0.; // kernel not selected
	}
#endif

	if ((kernel & SPMVM_KERNELS_SPLIT) && 
			(options & SPMVM_OPTION_NO_SPLIT_KERNELS)) {
		DEBUG_LOG(1,"Skipping the %s kernel because split kernels have not been configured.",name);
		return 0.; // kernel not selected
	}
	if ((kernel & SPMVM_KERNELS_COMBINED) && 
			(options & SPMVM_OPTION_NO_COMBINED_KERNELS)) {
		DEBUG_LOG(1,"Skipping the %s kernel because combined kernels have not been configured.",name);
		return 0.; // kernel not selected
	}
	if ((kernel & SPMVM_KERNEL_NOMPI)  && getNumberOfNodes() > 1) {
		DEBUG_LOG(1,"Skipping the %s kernel because there are multiple MPI processes.",name);
		return 0.; // non-MPI kernel
	} 
	if ((kernel & SPMVM_KERNEL_TASKMODE) && getNumberOfThreads() == 1) {
		DEBUG_LOG(1,"Skipping the %s kernel because there is only one thread.",name);
		return 0.; // not enough threads
	}

	switch (mat->format) {
		case SPM_FORMAT_DIST_CRS:
			switch (kernel) {
				case SPMVM_KERNEL_NOMPI:
					kernelFunc = (SpMVM_kernelFunc)&hybrid_kernel_0;
					break;
#ifdef MPI
				case SPMVM_KERNEL_VECTORMODE:
					kernelFunc = (SpMVM_kernelFunc)&hybrid_kernel_I;
					break;
				case SPMVM_KERNEL_GOODFAITH:
					kernelFunc = (SpMVM_kernelFunc)&hybrid_kernel_II;
					break;
				case SPMVM_KERNEL_TASKMODE:
					kernelFunc = (SpMVM_kernelFunc)&hybrid_kernel_III;
					break;
#endif
				default:
					ABORT("Non-valid kernel specified!");
			}
			break;
		case SPM_FORMAT_GLOB_CRS:
			switch (kernel) {
				case SPMVM_KERNEL_NOMPI:
					kernelFunc = (SpMVM_kernelFunc)&kern_glob_CRS_0;
					break;
				default:
					DEBUG_LOG(1,"Skipping the %s kernel because the matrix is not distributed.",name);
					return 0.;
			}
			break;
		case SPM_FORMAT_GLOB_MICVEC:
			switch (kernel) {
				case SPMVM_KERNEL_NOMPI:
					kernelFunc = (SpMVM_kernelFunc)&mic_kernel_0;
					break;
				default:
					DEBUG_LOG(1,"Skipping the %s kernel because there is no MICVEC version.",name);
			}
			break;
	}


#ifdef MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	double oldtime=1e9;

	for( it = 0; it < nIter; it++ ) {
		time = wctime();
		kernelFunc(res, mat->matrix, invec, options);

#ifdef MPI
		MPI_Barrier(MPI_COMM_WORLD);
#endif
		time = wctime()-time;
		time = time<oldtime?time:oldtime;
		oldtime=time;
	}

	return time;
}


/*MATRIX_TYPE *SpMVM_createGlobalMatrix (char *matrixPath, int format) 
  {
  MATRIX_TYPE *mat;
  CR_TYPE *cr;
  MM_TYPE *mm;

  mat = (MATRIX_TYPE *)allocateMemory(sizeof(MATRIX_TYPE),"matrix");
  cr = (CR_TYPE*) allocateMemory( sizeof( CR_TYPE ), "cr" );

  if (SpMVM_getRank() == 0) { 
  if (!isMMfile(matrixPath)){
  readCRbinFile(cr, matrixPath);
  } else{
  mm = readMMFile( matrixPath);
  cr = convertMMToCRMatrix( mm );
  freeMMMatrix(mm);
  }
  }

  mat->format = format;
  mat->nNonz = cr->nEnts;
  mat->nRows = cr->nRows;
  mat->nCols = cr->nCols;

  switch (format) {
  case SPM_FORMAT_MICVEC:
  mat->matrix = CRStoMICVEC(cr);
  break;
  case SPM_FORMAT_CRS:
  mat->matrix = cr;
  break;
  default:
  ABORT("No valid matrix format specified!");
  }



  return mat;

  }*/
