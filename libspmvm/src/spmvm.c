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

static int options;

static double wctime()
{
	struct timeval tp;

	gettimeofday(&tp, NULL);
	
	return (double) (tp.tv_sec + tp.tv_usec/1000000.0);
}


#if defined(COMPLEX) && defined(MPI)
typedef struct 
{
#ifdef DOUBLE
	double x;
	double y;
#endif
#ifdef SINGLE
	float x;
	float y;
#endif

} 
MPI_complex;

static void MPI_complAdd(MPI_complex *invec, MPI_complex *inoutvec, int *len)
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
	MPI_safecall(MPI_Op_create((MPI_User_function *)&MPI_complAdd,1,&MPI_MYSUM));
#endif

#else // ifdef MPI
	UNUSED(argc);
	UNUSED(argv);
	me = 0;
	spmvmOptions |= SPMVM_OPTION_SERIAL_IO; // important for createMatrix()

#endif // ifdef MPI

	if (spmvmOptions & SPMVM_OPTION_PIN || spmvmOptions & SPMVM_OPTION_PIN_SMT) {
		int nCores;
		int nPhysCores = SpMVM_getNumberOfPhysicalCores();
		if (spmvmOptions & SPMVM_OPTION_PIN)
			nCores = nPhysCores;
		else
			nCores = SpMVM_getNumberOfHwThreads();

		int offset = nPhysCores/SpMVM_getNumberOfRanksOnNode();
		omp_set_num_threads(nCores/SpMVM_getNumberOfRanksOnNode());
#pragma omp parallel
		{
			int error;
			int coreNumber;

			if (spmvmOptions & SPMVM_OPTION_PIN_SMT)
				coreNumber = omp_get_thread_num()/2+(offset*(SpMVM_getLocalRank()))+(omp_get_thread_num()%2)*nPhysCores;
			else
				coreNumber = omp_get_thread_num()+(offset*(SpMVM_getLocalRank()));

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

void *SpMVM_createVector(MATRIX_TYPE *matrix, int type, mat_data_t (*fp)(int))
{

	mat_data_t *val;
	int nRows;
	size_t size_val;


	if (matrix->format & SPM_FORMATS_GLOB)
	{
		size_val = (size_t)matrix->nRows*sizeof(mat_data_t);
		val = (mat_data_t*) allocateMemory( size_val, "vec->val");
		nRows = matrix->nRows;

		DEBUG_LOG(1,"NUMA-aware allocation of vector with %d rows",nRows);

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
				ABORT("No valid type for vector (has to be one of VECTOR_TYPE_LHS/_RHS/_BOTH");
		}

		size_val = (size_t)( nRows * sizeof(mat_data_t) );

		val = (mat_data_t*) allocateMemory( size_val, "vec->val");
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

#ifdef SBJDS_PERMCOLS
	if (matrix->format == SPM_FORMAT_GLOB_SBJDS)
		SpMVM_permuteVector(val,matrix->fullRowPerm,nRows);
#endif

	if (type & VECTOR_TYPE_HOSTONLY) {
		HOSTVECTOR_TYPE* vec;
		vec = (HOSTVECTOR_TYPE*) allocateMemory( sizeof( VECTOR_TYPE ), "vec");
		vec->val = val;
		vec->nRows = nRows; 

		DEBUG_LOG(1,"Host-only vector created successfully");

		return vec;
	} else {
		VECTOR_TYPE* vec;
		vec = (VECTOR_TYPE*) allocateMemory( sizeof( VECTOR_TYPE ), "vec");
		vec->val = val;
		vec->nRows = nRows; 
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
				ABORT("No valid type for vector (has to be one of VECTOR_TYPE_LHS/_RHS/_BOTH");
		}
		vec->CL_val_gpu = CL_allocDeviceMemoryMapped( size_val,vec->val,flag );
		CL_uploadVector(vec);
#endif

		DEBUG_LOG(1,"Vector created successfully");

		return vec;
	}

}

MATRIX_TYPE *SpMVM_createMatrix(char *matrixPath, int format, void *deviceFormats) 
{
	MATRIX_TYPE *mat;
	CR_TYPE *cr;

	mat = (MATRIX_TYPE *)allocateMemory(sizeof(MATRIX_TYPE),"matrix");
	mat->fullRowPerm = NULL;
	mat->fullInvRowPerm = NULL;
	mat->splitRowPerm = NULL;
	mat->splitInvRowPerm = NULL;

	cr = (CR_TYPE*) allocateMemory( sizeof( CR_TYPE ), "cr" );

	if (format & SPM_FORMATS_GLOB) {
		DEBUG_LOG(1,"Forcing serial I/O as the matrix format is a global one");
		options |= SPMVM_OPTION_SERIAL_IO;
	}

	if (SpMVM_getRank() == 0) 
	{ // root process reads row pointers (parallel IO) or entire matrix
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

#ifdef MPI
	// scatter matrix properties
	MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));
	MPI_safecall(MPI_Bcast(&(cr->nEnts),1,MPI_UNSIGNED,0,MPI_COMM_WORLD));
	MPI_safecall(MPI_Bcast(&(cr->nRows),1,MPI_UNSIGNED,0,MPI_COMM_WORLD));
	MPI_safecall(MPI_Bcast(&(cr->nCols),1,MPI_UNSIGNED,0,MPI_COMM_WORLD));
#endif


	if (format & SPM_FORMATS_DIST)
	{ // distributed matrix
#ifndef MPI
		UNUSED(deviceFormats);
		ABORT("Creating a distributed matrix without MPI is not possible");
#else
		if (format & SPM_FORMAT_DIST_CRS) {
			LCRP_TYPE *lcrp;
			if (options & SPMVM_OPTION_SERIAL_IO) 
				lcrp = setup_communication(cr, options);
			else
				lcrp = setup_communication_parallel(cr, matrixPath, options);
			mat->matrix = lcrp;
		} else {
			ABORT("Invalid format for distributed matrix");
		}

#endif
	} else 
	{ // global matrix
		if (format & SPM_FORMAT_GLOB_CRS) {
			mat->matrix = cr;
		} else if (format & SPM_FORMAT_GLOB_BJDS) {
			mat->matrix = CRStoBJDS(cr);
		} else if (format &  SPM_FORMAT_GLOB_SBJDS) {
			mat->matrix = CRStoSBJDS(cr,&(mat->fullRowPerm),&(mat->fullInvRowPerm));
		} else {
			ABORT("Invalid format for global matrix!");
		}

	}

#ifdef OPENCL
	if (!(format & SPM_FORMAT_HOSTONLY))
	{
		DEBUG_LOG(1,"Skipping device matrix creation because the matrix ist host-only.");
	} else if (deviceFormats == NULL) 
	{
		ABORT("Device matrix formats have to be passed to SPMVM_distributeCRS!");
	} else 
	{
		CL_uploadCRS ( mat, (SPM_GPUFORMATS *)deviceFormats, options);
	}
#else
	UNUSED(deviceFormats);
#endif


	mat->format = format;
	mat->nNonz = cr->nEnts;
	mat->nRows = cr->nRows;
	mat->nCols = cr->nCols;

	DEBUG_LOG(1,"%ux%u matrix (%u nonzeros) created successfully",mat->nCols,mat->nRows,mat->nNonz);

	return mat;
}


double SpMVM_solve(VECTOR_TYPE *res, MATRIX_TYPE *mat, VECTOR_TYPE *invec, 
		int kernel, int nIter)
{
	int it;
	double time = 0;
	double oldtime=1e9;
	SpMVM_kernelFunc kernelFunc = SpMVM_selectKernelFunc(options,kernel,mat);

	if (!kernelFunc)
		return -1.0;

#ifdef MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif

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

