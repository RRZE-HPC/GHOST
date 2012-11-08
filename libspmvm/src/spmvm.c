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
static SpMVM_kernelFunc * kernels;

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

void *SpMVM_createVector(SETUP_TYPE *setup, int type, mat_data_t (*fp)(int))
{

	mat_data_t *val;
	int nRows;
	size_t size_val;
	MATRIX_TYPE *matrix = setup->fullMatrix;


	if (setup->flags & SPM_GLOBAL)
	{
		size_val = (size_t)matrix->nRows*sizeof(mat_data_t);
		val = (mat_data_t*) allocateMemory( size_val, "vec->val");
		nRows = matrix->nRows;

		if (matrix->trait.flags & SPM_PERMUTECOLUMNS)
			SpMVM_permuteVector(val,matrix->rowPerm,nRows);

		DEBUG_LOG(1,"NUMA-aware allocation of vector with %d rows",nRows);

		mat_idx_t i;
		if (fp) {
#pragma omp parallel for schedule(runtime)
			for (i=0; i<matrix->nRows; i++) 
				val[i] = fp(i);
		}else {
#ifdef COMPLEX
#pragma omp parallel for schedule(runtime)
			for (i=0; i<matrix->nRows; i++) val[i] = 0.+I*0.;
#else
#pragma omp parallel for schedule(runtime)
			for (i=0; i<matrix->nRows; i++) val[i] = 0.;
#endif
		}

	} 
	else 
	{
		LCRP_TYPE *lcrp = setup->communicator;
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
#pragma omp parallel for schedule(runtime)
			for (i=0; i<lcrp->lnRows[me]; i++) 
				val[i] = fp(lcrp->lfRow[me]+i);
#pragma omp parallel for schedule(runtime)
			for (i=lcrp->lnRows[me]; i<nRows; i++) 
				val[i] = fp(lcrp->lfRow[me]+i);
		}else {
#ifdef COMPLEX
#pragma omp parallel for schedule(runtime)
			for (i=0; i<lcrp->lnRows[me]; i++) val[i] = 0.+I*0.;
#pragma omp parallel for schedule(runtime)
			for (i=lcrp->lnRows[me]; i<nRows; i++) val[i] = 0.+I*0.;
#else
#pragma omp parallel for schedule(runtime)
			for (i=0; i<lcrp->lnRows[me]; i++) val[i] = 0.;
#pragma omp parallel for schedule(runtime)
			for (i=lcrp->lnRows[me]; i<nRows; i++) val[i] = 0.;
#endif
		}
	}


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

SETUP_TYPE *SpMVM_createSetup(char *matrixPath, mat_trait_t *traits, int nTraits, setup_flags_t setup_flags, void *deviceFormats) 
{
	UNUSED(nTraits);
	DEBUG_LOG(1,"Creating setup");
	SETUP_TYPE *setup;
	CR_TYPE *cr;

	setup = (SETUP_TYPE *)allocateMemory(sizeof(SETUP_TYPE),"setup");
	setup->flags = setup_flags;

	if (setup_flags & SPM_GLOBAL) {
		DEBUG_LOG(1,"Forcing serial I/O as the matrix format is a global one");
		options |= SPMVM_OPTION_SERIAL_IO;
	}

	if (SpMVM_getRank() == 0) 
	{ // root process reads row pointers (parallel IO) or entire matrix
		if (!isMMfile(matrixPath)){
			if (options & SPMVM_OPTION_SERIAL_IO)
				cr = readCRbinFile(matrixPath,0,traits[0].format & SPM_FORMAT_CRSCD);
			else
				cr = readCRbinFile(matrixPath,1,traits[0].format & SPM_FORMAT_CRSCD);
		} else{
			MM_TYPE *mm = readMMFile( matrixPath);
			cr = convertMMToCRMatrix( mm );
			freeMMMatrix(mm);
		}
	} else // TODO scatter in read function 
	{
		cr = (CR_TYPE *)allocateMemory(sizeof(CR_TYPE),"cr");
	}

#ifdef MPI
	// scatter matrix properties
	MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));
	MPI_safecall(MPI_Bcast(&(cr->nEnts),1,MPI_UNSIGNED,0,MPI_COMM_WORLD));
	MPI_safecall(MPI_Bcast(&(cr->nRows),1,MPI_UNSIGNED,0,MPI_COMM_WORLD));
	MPI_safecall(MPI_Bcast(&(cr->nCols),1,MPI_UNSIGNED,0,MPI_COMM_WORLD));
	MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));
#endif

	if (setup_flags & SPM_DISTRIBUTED)
	{ // distributed matrix
#ifndef MPI
		UNUSED(deviceFormats);
		ABORT("Creating a distributed matrix without MPI is not possible");
#else
		//		if (traits[0].format == SPM_FORMAT_CRS) {
		if (options & SPMVM_OPTION_SERIAL_IO) 
			SpMVM_createDistributedSetupSerial(setup, cr, options);
		else
			SpMVM_createDistributedSetup(setup, cr, matrixPath, options);
		//		} else {
		//			ABORT("Invalid format for distributed matrix");
		//		}
		setup->fullMatrix->trait = traits[0];	
		setup->localMatrix->trait = traits[1];	
		setup->remoteMatrix->trait = traits[2];	

#endif
	} else 
	{ // global matrix
		// TODO in create function
		setup->fullMatrix = (MATRIX_TYPE *)allocateMemory(sizeof(MATRIX_TYPE),"full matrix");	
		DEBUG_LOG(1,"Creating global %s matrix",SpMVM_matrixFormatName(traits[0]));
		mat_format_t format = traits[0].format;
		mat_flags_t flags = traits[0].flags;
		if (format == SPM_FORMAT_CRS  || format == SPM_FORMAT_CRSCD) {
			setup->fullMatrix->data = cr;
		} else if (format == SPM_FORMAT_BJDS) {
			setup->fullMatrix->data = CRStoBJDS(cr);
		} else if (format ==  SPM_FORMAT_SBJDS) {
			setup->fullMatrix->data = CRStoSBJDS(cr,&(setup->fullMatrix->rowPerm),&(setup->fullMatrix->invRowPerm),flags);
		} else if (format ==  SPM_FORMAT_TBJDS) {
			setup->fullMatrix->data = CRStoTBJDS(cr,1);
		} else if (format ==  SPM_FORMAT_STBJDS) {
			setup->fullMatrix->data = CRStoSTBJDS(cr,1,
					*(unsigned int *)(traits[0].aux),
					&(setup->fullMatrix->rowPerm),&(setup->fullMatrix->invRowPerm),flags);
		} else if (format ==  SPM_FORMAT_TCBJDS) {
			setup->fullMatrix->data = CRStoTBJDS(cr,0);
		} else {
			ABORT("Invalid format for global matrix!");
		}

		setup->fullMatrix->trait = traits[0];	
		setup->fullMatrix->nRows = cr->nRows;
		setup->fullMatrix->nCols = cr->nCols;
		setup->fullMatrix->nNonz = cr->nEnts;

	}
	setup->nNz = cr->nEnts;
	setup->nRows = cr->nRows;
	setup->nCols = cr->nCols;

#ifdef OPENCL
	if (!(flags & SPM_HOSTONLY))
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


	//DEBUG_LOG(1,"%ux%u matrix (%u nonzeros) created successfully",mat->nCols,mat->nRows,mat->nNonz);

	kernels = SpMVM_setupKernels(setup);

	DEBUG_LOG(1,"Setup created successfully");
	return setup;
}


double SpMVM_solve(VECTOR_TYPE *res, SETUP_TYPE *setup, VECTOR_TYPE *invec, 
		int kernel, int nIter)
{
	int it;
	double time = 0;
	double oldtime=1e9;
	void * arg;

	SpMVM_kernelFunc kernelFunc = NULL;
	if (setup->flags & SPM_GLOBAL) {
		kernelFunc = kernels[setup->fullMatrix->trait.format];
		arg = setup->fullMatrix->data;
	} else {
		switch (kernel) {
			case SPMVM_KERNEL_VECTORMODE:
				kernelFunc = (SpMVM_kernelFunc)&hybrid_kernel_I;
				arg = setup;
				break;
			case SPMVM_KERNEL_GOODFAITH:
				kernelFunc = (SpMVM_kernelFunc)&hybrid_kernel_II;
				arg = setup;
				break;
			case SPMVM_KERNEL_TASKMODE:
				kernelFunc = (SpMVM_kernelFunc)&hybrid_kernel_III;
				arg = setup;
				break;
		}
	}



	if (!kernelFunc)
		return -1.0;

#ifdef MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif

	for( it = 0; it < nIter; it++ ) {
		time = wctime();
		kernelFunc(res, arg, invec, options); // TODO

#ifdef MPI
		MPI_Barrier(MPI_COMM_WORLD);
#endif
		time = wctime()-time;
		time = time<oldtime?time:oldtime;
		oldtime=time;
	}

	return time;
}

