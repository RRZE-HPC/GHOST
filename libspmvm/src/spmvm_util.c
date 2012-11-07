#define _GNU_SOURCE
#include "spmvm_util.h"
#include "spmvm.h"
#include "referencesolvers.h"
#include "matricks.h"
#include "kernel.h"
#include <sys/param.h>
#include <libgen.h>
#include <unistd.h>
#ifdef MPI
#include <mpihelper.h>
#endif

#ifdef OPENCL
#include "cl_matricks.h"
#endif

#ifdef LIKWID
#include <likwid.h>
#endif

#include <sched.h>
#include <errno.h>
#include <omp.h>
#include <string.h>




void SpMVM_printMatrixInfo(MATRIX_TYPE *matrix, char *matrixName, int options)
{

	int me;
	size_t ws;


	me = SpMVM_getRank();

#ifdef OPENCL	
	size_t fullMemSize, localMemSize, remoteMemSize, 
		   totalFullMemSize = 0, totalLocalMemSize = 0, totalRemoteMemSize = 0;

	if (!(options & SPMVM_OPTION_NO_COMBINED_KERNELS)) { // combined computation
		fullMemSize = getBytesize(matrix->devMatrix->fullMatrix, matrix->devMatrix->fullFormat)/
			(1024*1024);
		MPI_safecall(MPI_Reduce(&fullMemSize, &totalFullMemSize,1,MPI_LONG,MPI_SUM,0,
					MPI_COMM_WORLD));
	} 
	if (!(options & SPMVM_OPTION_NO_SPLIT_KERNELS)) { // split computation
		localMemSize = getBytesize(matrix->devMatrix->localMatrix,matrix->devMatrix->localFormat)/
			(1024*1024);
		remoteMemSize = getBytesize(matrix->devMatrix->remoteMatrix,matrix->devMatrix->remoteFormat)/
			(1024*1024);
		MPI_safecall(MPI_Reduce(&localMemSize, &totalLocalMemSize,1,MPI_LONG,MPI_SUM,0,
					MPI_COMM_WORLD));
		MPI_safecall(MPI_Reduce(&remoteMemSize, &totalRemoteMemSize,1,MPI_LONG,MPI_SUM,0,
					MPI_COMM_WORLD));
	}
#endif	

	if(me==0){
		int pin = (options & SPMVM_OPTION_PIN || options & SPMVM_OPTION_PIN_SMT)?
			1:0;
		char *pinStrategy = options & SPMVM_OPTION_PIN?"phys. cores":"virt. cores";
		ws = ((matrix->nRows+1)*sizeof(int) + 
				matrix->nNonz*(sizeof(mat_data_t)+sizeof(int)))/(1024*1024);
		printf("-----------------------------------------------\n");
		printf("-------        Matrix information       -------\n");
		printf("-----------------------------------------------\n");
		printf("Investigated matrix              : %12s\n", matrixName); 
		printf("Dimension of matrix              : %12.0f\n", (double)matrix->nRows); 
		printf("Non-zero elements                : %12.0f\n", (double)matrix->nNonz); 
		printf("Average elements per row         : %12.3f\n", (double)matrix->nNonz/
				(double)matrix->nRows);
		printf("CRS matrix                   [MB]: %12lu\n", ws);
		if (!(SpMVM_matrixTraitExtractFormat(matrix->trait) & SPM_FORMAT_CRS)) {
			printf("Host matrix %*s%s%s [MB]: %12u\n", 
					15-(int)strlen(SpMVM_matrixFormatName(matrix->trait)),"(",
					SpMVM_matrixFormatName(matrix->trait),")",
					SpMVM_matrixSize(matrix)/(1024*1024));
			if (SpMVM_matrixTraitExtractFormat(matrix->trait) & SPM_FORMAT_STBJDS)
				printf("S-TBJDS matrix nu                : %12f\n",((BJDS_TYPE *)(matrix->matrix))->nu);
		}
#ifdef OPENCL	
		if (!(options & SPMVM_OPTION_NO_COMBINED_KERNELS)) { // combined computation
			printf("Dev. matrix (combin.%4s-%2d) [MB]: %12lu\n", SPM_FORMAT_NAMES[matrix->devMatrix->fullFormat],matrix->devMatrix->fullT,totalFullMemSize);
		}	
		if (!(options & SPMVM_OPTION_NO_SPLIT_KERNELS)) { // split computation
			printf("Dev. matrix (local  %4s-%2d) [MB]: %12lu\n", SPM_FORMAT_NAMES[matrix->devMatrix->localFormat],matrix->devMatrix->localT,totalLocalMemSize); 
			printf("Dev. matrix (remote %4s-%2d) [MB]: %12lu\n", SPM_FORMAT_NAMES[matrix->devMatrix->remoteFormat],matrix->devMatrix->remoteT,totalRemoteMemSize);
			printf("Dev. matrix (local & remote) [MB]: %12lu\n", totalLocalMemSize+
					totalRemoteMemSize); 
		}
#endif
		printf("-----------------------------------------------\n\n");
		printf("-----------------------------------------------\n");
		printf("-------        Setup information        -------\n");
		printf("-----------------------------------------------\n");
		printf("Equation                         : %12s\n", options&SPMVM_OPTION_AXPY?"y <- y+A*x":"y <- A*x"); 
		printf("Work distribution scheme         : %12s\n", SpMVM_workdistName(options));
		printf("Automatic pinning                : %12s\n", pin?"enabled":"disabled");
		if (pin)
			printf("Pinning threads to               : %12s\n", pinStrategy);
		printf("-----------------------------------------------\n\n");
		fflush(stdout);
	}
}

void SpMVM_printEnvInfo() 
{

	int me = SpMVM_getRank();

	int nproc;
	int nnodes;
#ifdef MPI
	nnodes = SpMVM_getNumberOfNodes();
	MPI_safecall(MPI_Comm_size ( MPI_COMM_WORLD, &nproc ));
#else
	nnodes = 1;
	nproc = 1;
#endif

#ifdef OPENCL
	CL_DEVICE_INFO * devInfo = CL_getDeviceInfo();
#endif

	if (me==0) {
		int nthreads;
		int nphyscores = SpMVM_getNumberOfPhysicalCores();
		int ncores = SpMVM_getNumberOfHwThreads();

		omp_sched_t omp_sched;
		int omp_sched_mod;
		char omp_sched_str[32];
		omp_get_schedule(&omp_sched,&omp_sched_mod);
		switch (omp_sched) {
			case omp_sched_static:
				sprintf(omp_sched_str,"static,%d",omp_sched_mod);
				break;
			case omp_sched_dynamic:
				sprintf(omp_sched_str,"dynamic,%d",omp_sched_mod);
				break;
			case omp_sched_guided:
				sprintf(omp_sched_str,"guided,%d",omp_sched_mod);
				break;
			case omp_sched_auto:
				sprintf(omp_sched_str,"auto,%d",omp_sched_mod);
				break;
			default:
				sprintf(omp_sched_str,"unknown");
				break;
		}


#pragma omp parallel
#pragma omp master
		nthreads = omp_get_num_threads();

		printf("-----------------------------------------------\n");
		printf("-------       System information        -------\n");
		printf("-----------------------------------------------\n");
		printf("Nodes                            : %12d\n", nnodes); 
		//printf("MPI processes                    : %12d\n", nproc); 
		printf("MPI processes     per node       : %12d\n", nproc/nnodes); 
		printf("Available cores   per node       : %12d\n", nphyscores); 
		printf("Available threads per node       : %12d\n", ncores); 
		printf("OpenMP threads    per node       : %12d\n", nproc/nnodes*nthreads);
		printf("OpenMP threads    per process    : %12d\n", nthreads);
		printf("OpenMP scheduling                : %12s\n", omp_sched_str);
#ifdef OPENCL
		printf("OpenCL devices                   :\n");
		int i;
		for (i=0; i<devInfo->nDistinctDevices; i++) {
			printf("                            %3d x %13s\n",devInfo->nDevices[i],devInfo->names[i]);
		}
#endif
		printf("-----------------------------------------------\n\n");

		printf("-----------------------------------------------\n");
		printf("-------      LibSpMVM information       -------\n");
		printf("-----------------------------------------------\n");
		printf("Build date                       : %12s\n", __DATE__); 
		printf("Build time                       : %12s\n", __TIME__); 
		printf("Data type                        : %12s\n", DATATYPE_NAMES[DATATYPE_DESIRED]);
#ifdef MIC
		printf("MIC kernels                      :      enabled\n");
#else
		printf("MIC kernels                      :     disabled\n");
#endif
#ifdef AVX
		printf("AVX kernels                      :      enabled\n");
#else
		printf("AVX kernels                      :     disabled\n");
#endif
#ifdef SSE
		printf("SSE kernels                      :      enabled\n");
#else
		printf("SSE kernels                      :     disabled\n");
#endif
#ifdef MPI
		printf("MPI support                      :      enabled\n");
#else
		printf("MPI support                      :     disabled\n");
#endif
#ifdef OPENCL
		printf("OpenCL support                   :      enabled\n");
#else
		printf("OpenCL support                   :     disabled\n");
#endif
#ifdef LIKWID
		printf("Likwid support                   :      enabled\n");
#ifdef LIKWID_MARKER_FINE
		printf("Likwid Marker API (high res)     :      enabled\n");
#else
#ifdef LIKWID_MARKER
		printf("Likwid Marker API                :      enabled\n");
#endif
#endif
#else
		printf("Likwid support                   :     disabled\n");
#endif
		printf("-----------------------------------------------\n\n");
		fflush(stdout);

	}
#ifdef OPENCL
	destroyCLdeviceInfo(devInfo);
#endif


}

HOSTVECTOR_TYPE * SpMVM_createGlobalHostVector(int nRows, mat_data_t (*fp)(int))
{

	int me = SpMVM_getRank();

	if (me==0) {
		return SpMVM_newHostVector( nRows,fp );
	} else {
		return SpMVM_newHostVector(0,NULL);
	}
}

void SpMVM_referenceSolver(CR_TYPE *cr, mat_data_t *rhs, mat_data_t *lhs, int nIter, int spmvmOptions) 
{

	int iteration;
	int i;

	for( i = 0; i < cr->rowOffset[cr->nRows]; ++i) {
		cr->col[i] += 1;
	}	

	if (spmvmOptions & SPMVM_OPTION_AXPY) {

		for (iteration=0; iteration<nIter; iteration++) {
#ifdef DOUBLE
#ifdef COMPLEX
			fortrancrsaxpyc_(&(cr->nRows), &(cr->nEnts), lhs, rhs, cr->val ,
					cr->col, cr->rowOffset);
#else
			fortrancrsaxpy_(&(cr->nRows), &(cr->nEnts), lhs, rhs, cr->val ,
					cr->col, cr->rowOffset);
#endif
#endif
#ifdef SINGLE
#ifdef COMPLEX
			fortrancrsaxpycf_(&(cr->nRows), &(cr->nEnts), lhs, rhs, cr->val,
					cr->col, cr->rowOffset);
#else
			fortrancrsaxpyf_(&(cr->nRows), &(cr->nEnts), lhs, rhs, cr->val,
					cr->col, cr->rowOffset);
#endif
#endif
		}
	} else {
#ifdef DOUBLE
#ifdef COMPLEX
		fortrancrsc_(&(cr->nRows), &(cr->nEnts), lhs, rhs, cr->val, cr->col,
				cr->rowOffset);
#else
		fortrancrs_(&(cr->nRows), &(cr->nEnts), lhs, rhs, cr->val, cr->col,
				cr->rowOffset);
#endif
#endif
#ifdef SINGLE
#ifdef COMPLEX
		fortrancrscf_(&(cr->nRows), &(cr->nEnts), lhs, rhs, cr->val, cr->col,
				cr->rowOffset);
#else
		fortrancrsf_(&(cr->nRows), &(cr->nEnts), lhs, rhs, cr->val , cr->col,
				cr->rowOffset);
#endif
#endif
	}
	for( i = 0; i < cr->rowOffset[cr->nRows]; ++i) {
		cr->col[i] -= 1;
	}
}

void SpMVM_zeroVector(VECTOR_TYPE *vec) 
{
	int i;
	for (i=0; i<vec->nRows; i++) {
#ifdef COMPLEX
		vec->val[i] = 0;
#else
		vec->val[i] = 0+I*0;
#endif
	}

#ifdef OPENCL
	CL_uploadVector(vec);
#endif


}

HOSTVECTOR_TYPE* SpMVM_newHostVector( const int nRows, mat_data_t (*fp)(int)) 
{
	HOSTVECTOR_TYPE* vec;
	size_t size_val;
	int i;

	size_val = (size_t)( nRows * sizeof(mat_data_t) );
	vec = (HOSTVECTOR_TYPE*) allocateMemory( sizeof( VECTOR_TYPE ), "vec");


	vec->val = (mat_data_t*) allocateMemory( size_val, "vec->val");
	vec->nRows = nRows;

	if (fp) {
#pragma omp parallel for schedule(runtime)
		for (i=0; i<nRows; i++) 
			vec->val[i] = fp(i);

	}else {
#ifdef COMPLEX
#pragma omp parallel for schedule(runtime)
		for (i=0; i<nRows; i++) vec->val[i] = 0.+I*0.;
#else
#pragma omp parallel for schedule(runtime)
		for (i=0; i<nRows; i++) vec->val[i] = 0.;
#endif
	}


	return vec;
}

VECTOR_TYPE* SpMVM_newVector( const int nRows ) 
{
	VECTOR_TYPE* vec;
	size_t size_val;
	int i;

	size_val = (size_t)( nRows * sizeof(mat_data_t) );
	vec = (VECTOR_TYPE*) allocateMemory( sizeof( VECTOR_TYPE ), "vec");


	vec->val = (mat_data_t*) allocateMemory( size_val, "vec->val");
	vec->nRows = nRows;

#pragma omp parallel for schedule(runtime) 
	for( i = 0; i < nRows; i++ ) 
		vec->val[i] = 0.0;

#ifdef OPENCL
#ifdef CL_IMAGE
	vec->CL_val_gpu = CL_allocDeviceMemoryCached( size_val,vec->val );
#else
	vec->CL_val_gpu = CL_allocDeviceMemoryMapped( size_val,vec->val,CL_MEM_READ_WRITE );
#endif
	//vec->CL_val_gpu = CL_allocDeviceMemory( size_val );
	//printf("before: %p\n",vec->val);
	//vec->val = CL_mapBuffer(vec->CL_val_gpu,size_val);
	//printf("after: %p\n",vec->val);
	//CL_uploadVector(vec);
#endif

	return vec;
}

VECTOR_TYPE * SpMVM_distributeVector(LCRP_TYPE *lcrp, HOSTVECTOR_TYPE *vec)
{
	int me = SpMVM_getRank();

	int pseudo_ldim = lcrp->lnRows[me]+lcrp->halo_elements ;


	VECTOR_TYPE *nodeVec = SpMVM_newVector( pseudo_ldim ); 

#ifdef MPI
	MPI_safecall(MPI_Scatterv ( vec->val, lcrp->lnRows, lcrp->lfRow, MPI_MYDATATYPE,
				nodeVec->val, lcrp->lnRows[me], MPI_MYDATATYPE, 0, MPI_COMM_WORLD ));
#else
	int i;
	for (i=0; i<vec->nRows; i++) nodeVec->val[i] = vec->val[i];
#endif
#ifdef OPENCL
	CL_uploadVector(nodeVec);
#endif

	return nodeVec;
}

void SpMVM_collectVectors(MATRIX_TYPE *matrix, VECTOR_TYPE *vec, 
		HOSTVECTOR_TYPE *totalVec, int kernel) {

	if (SpMVM_matrixTraitExtractFlags(matrix->trait) & SPM_DISTRIBUTED)
	{
		if (SpMVM_matrixTraitExtractFormat(matrix->trait) != SPM_FORMAT_CRS)
			DEBUG_LOG(0,"Cannot handle other matrices than CRS in the MPI case!");

		LCRP_TYPE *lcrp = (LCRP_TYPE *)(matrix->matrix);
		int me = SpMVM_getRank();
		if ( 0x1<<kernel & SPMVM_KERNELS_COMBINED)  {
			SpMVM_permuteVector(vec->val,matrix->fullInvRowPerm,lcrp->lnRows[me]);
		} else if ( 0x1<<kernel & SPMVM_KERNELS_SPLIT ) {
			SpMVM_permuteVector(vec->val,matrix->splitInvRowPerm,lcrp->lnRows[me]);
		}
		MPI_safecall(MPI_Gatherv(vec->val,lcrp->lnRows[me],MPI_MYDATATYPE,totalVec->val,
					lcrp->lnRows,lcrp->lfRow,MPI_MYDATATYPE,0,MPI_COMM_WORLD));
	} else
	{
		int i;
		UNUSED(kernel);
		SpMVM_permuteVector(vec->val,matrix->fullInvRowPerm,matrix->nRows);
		for (i=0; i<totalVec->nRows; i++) 
			totalVec->val[i] = vec->val[i];
	}
}

void SpMVM_swapVectors(VECTOR_TYPE *v1, VECTOR_TYPE *v2) 
{
	mat_data_t *dtmp;

	dtmp = v1->val;
	v1->val = v2->val;
	v2->val = dtmp;
#ifdef OPENCL
	cl_mem tmp;
	tmp = v1->CL_val_gpu;
	v1->CL_val_gpu = v2->CL_val_gpu;
	v2->CL_val_gpu = tmp;
#endif

}

void SpMVM_normalizeVector( VECTOR_TYPE *vec)
{
	int i;
	mat_data_t sum = 0;

	for (i=0; i<vec->nRows; i++)	
		sum += vec->val[i]*vec->val[i];

	mat_data_t f = (mat_data_t)1/SQRT(ABS(sum));

	for (i=0; i<vec->nRows; i++)	
		vec->val[i] *= f;

#ifdef OPENCL
	CL_uploadVector(vec);
#endif
}

void SpMVM_normalizeHostVector( HOSTVECTOR_TYPE *vec)
{
	int i;
	mat_data_t sum = 0;

	for (i=0; i<vec->nRows; i++)	
		sum += vec->val[i]*vec->val[i];

	mat_data_t f = (mat_data_t)1/SQRT(ABS(sum));

	for (i=0; i<vec->nRows; i++)	
		vec->val[i] *= f;
}

void SpMVM_freeHostVector( HOSTVECTOR_TYPE* const vec ) 
{
	if( vec ) {
		freeMemory( (size_t)(vec->nRows*sizeof(mat_data_t)), "vec->val",  vec->val );
		free( vec );
	}
}

void SpMVM_freeVector( VECTOR_TYPE* const vec ) 
{
	if( vec ) {
		freeMemory( (size_t)(vec->nRows*sizeof(mat_data_t)), "vec->val",  vec->val );
#ifdef OPENCL
		CL_freeDeviceMemory( vec->CL_val_gpu );
#endif
		free( vec );
	}
}

void SpMVM_freeCRS( CR_TYPE* const cr ) 
{

	if (cr) {
		if (cr->rowOffset)
			free(cr->rowOffset);
		if (cr->col)
			free(cr->col);
		if (cr->val)
			free(cr->val);
		free(cr);
	}
	//TODO
	/*	size_t size_rowOffset, size_col, size_val;

		if( cr ) {

		size_rowOffset  = (size_t)( (cr->nRows+1) * sizeof( int ) );
		size_col        = (size_t)( cr->nEnts     * sizeof( int ) );
		size_val        = (size_t)( cr->nEnts     * sizeof( mat_data_t) );

		freeMemory( size_rowOffset,  "cr->rowOffset", cr->rowOffset );
		freeMemory( size_col,        "cr->col",       cr->col );
		freeMemory( size_val,        "cr->val",       cr->val );
		freeMemory( sizeof(CR_TYPE), "cr",            cr );

		}*/
}

void SpMVM_freeLCRP( LCRP_TYPE* const lcrp ) 
{
	if( lcrp ) {
		free( lcrp->lnEnts );
		free( lcrp->lnRows );
		free( lcrp->lfEnt );
		free( lcrp->lfRow );
		free( lcrp->wishes );
		free( lcrp->wishlist_mem );
		free( lcrp->wishlist );
		free( lcrp->dues );
		free( lcrp->duelist_mem );
		free( lcrp->duelist );
		free( lcrp->due_displ );
		free( lcrp->wish_displ );
		free( lcrp->hput_pos );
		free( lcrp->val );
		free( lcrp->col );
		free( lcrp->lrow_ptr );
		free( lcrp->lrow_ptr_l );
		free( lcrp->lrow_ptr_r );
		free( lcrp->lcol );
		free( lcrp->rcol );
		free( lcrp->lval );
		free( lcrp->rval );
		/*free( lcrp->fullRowPerm );
		  free( lcrp->fullInvRowPerm );
		  free( lcrp->splitRowPerm );
		  free( lcrp->splitInvRowPerm );
#ifdef OPENCL
CL_freeMatrix( lcrp->fullMatrix, lcrp->fullFormat );
CL_freeMatrix( lcrp->localMatrix, lcrp->localFormat );
CL_freeMatrix( lcrp->remoteMatrix, lcrp->remoteFormat );
#endif*/
		free( lcrp );
	}
}

void SpMVM_permuteVector( mat_data_t* vec, int* perm, int len) 
{
	/* permutes values in vector so that i-th entry is mapped to position perm[i] */
	int i;
	mat_data_t* tmp;

	if (perm == NULL) {
		DEBUG_LOG(1,"Permutation vector is NULL, returning.");
		return;
	} else {
		DEBUG_LOG(1,"Permuting vector");
	}


	tmp = (mat_data_t*)allocateMemory(sizeof(mat_data_t)*len, "permute tmp");

	for(i = 0; i < len; ++i) {
		if( perm[i] >= len ) {
			ABORT("Permutation index out of bounds: %d > %d\n",perm[i],len);
		}
		tmp[perm[i]] = vec[i];
	}
	for(i=0; i < len; ++i) {
		vec[i] = tmp[i];
	}

	free(tmp);
}

SpMVM_kernelFunc SpMVM_selectKernelFunc(int options, int kernel, MATRIX_TYPE *mat) 
{
	char *name = SpMVM_kernelName(kernel);
	SpMVM_kernelFunc kernelFunc = NULL;

#ifndef MPI
	if (!(kernel & SPMVM_KERNEL_NOMPI)) {
		DEBUG_LOG(1,"Skipping the %s kernel because the library is built without MPI.",name);
		return NULL; // kernel not selected
	}
#endif

	if ((kernel & SPMVM_KERNELS_SPLIT) && 
			(options & SPMVM_OPTION_NO_SPLIT_KERNELS)) {
		DEBUG_LOG(1,"Skipping the %s kernel because split kernels have not been configured.",name);
		return NULL; // kernel not selected
	}
	if ((kernel & SPMVM_KERNELS_COMBINED) && 
			(options & SPMVM_OPTION_NO_COMBINED_KERNELS)) {
		DEBUG_LOG(1,"Skipping the %s kernel because combined kernels have not been configured.",name);
		return NULL; // kernel not selected
	}
	if ((kernel & SPMVM_KERNEL_NOMPI)  && SpMVM_getNumberOfNodes() > 1) {
		DEBUG_LOG(1,"Skipping the %s kernel because there are multiple MPI processes.",name);
		return NULL; // non-MPI kernel
	} 
	if ((kernel & SPMVM_KERNEL_TASKMODE) && SpMVM_getNumberOfThreads() == 1) {
		DEBUG_LOG(1,"Skipping the %s kernel because there is only one thread.",name);
		return NULL; // not enough threads
	}

	// TODO format muss nicht bitweise sein weil eh nur eins geht
	switch (SpMVM_matrixTraitExtractFormat(mat->trait)) {
		case SPM_FORMAT_CRS:
			switch (kernel) {
				case SPMVM_KERNEL_NOMPI:
					if (SpMVM_matrixTraitExtractFlags(mat->trait) & SPM_DISTRIBUTED)
						kernelFunc = (SpMVM_kernelFunc)&hybrid_kernel_0;
					else
						kernelFunc = (SpMVM_kernelFunc)&kern_glob_CRS_0;
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
					DEBUG_LOG(1,"Non-valid kernel specified!");
					return NULL;
			}
			break;
		case SPM_FORMAT_CRSCD:
			switch (kernel) {
				case SPMVM_KERNEL_NOMPI:
					kernelFunc = (SpMVM_kernelFunc)&kern_glob_CRS_CD_0;
					break;
				default:
					DEBUG_LOG(1,"Skipping the %s kernel because the matrix is not distributed.",name);
					return NULL;
			}
			break;
		case SPM_FORMAT_SBJDS:
		case SPM_FORMAT_BJDS:
			switch (kernel) {
#ifdef MIC
				case SPMVM_KERNEL_NOMPI:
					//kernelFunc = (SpMVM_kernelFunc)&mic_kernel_0_intr;
					kernelFunc = (SpMVM_kernelFunc)&mic_kernel_0_intr_16;
					break;
#endif
#ifdef AVX
				case SPMVM_KERNEL_NOMPI:
					kernelFunc = (SpMVM_kernelFunc)&avx_kernel_0_intr;
					//kernelFunc = (SpMVM_kernelFunc)&avx_kernel_0_intr_rem;
					break;
#endif
#ifdef SSE
				case SPMVM_KERNEL_NOMPI:
					kernelFunc = (SpMVM_kernelFunc)&sse_kernel_0_intr;
					break;
#endif
				default:
					DEBUG_LOG(1,"Skipping the %s kernel because there is no BJDS version.",name);
					return NULL;
			}
			break;
		case SPM_FORMAT_STBJDS:
		case SPM_FORMAT_TBJDS:
			switch (kernel) {
#ifdef MIC
				case SPMVM_KERNEL_NOMPI:
					kernelFunc = (SpMVM_kernelFunc)&mic_kernel_0_intr_16_rem;
					break;
#endif
#ifdef SSE
				case SPMVM_KERNEL_NOMPI:
					kernelFunc = (SpMVM_kernelFunc)&sse_kernel_0_intr_rem;
					break;
#endif
#ifdef AVX
				case SPMVM_KERNEL_NOMPI:
					kernelFunc = (SpMVM_kernelFunc)&avx_kernel_0_intr_rem;
					break;
#endif
				default:
					DEBUG_LOG(1,"Skipping the %s kernel because there is no BJDS version.",name);
					return NULL;
			}
			break;
		case SPM_FORMAT_TCBJDS:
			switch (kernel) {
#ifdef AVX
				case SPMVM_KERNEL_NOMPI:
					kernelFunc = (SpMVM_kernelFunc)&avx_kernel_0_intr_rem_if;
					break;
#endif
				default:
					DEBUG_LOG(1,"Skipping the %s kernel because there is no BJDS version.",name);
					return NULL;
			}
			break;

		default:
			DEBUG_LOG(1,"Non-valid matrix format specified (%hu)!",SpMVM_matrixTraitExtractFormat(mat->trait));
			return NULL;
	}


	return kernelFunc;


}

char * SpMVM_kernelName(int kernel) 
{

	switch (kernel) {
		case SPMVM_KERNEL_NOMPI:
			return "non-MPI";
			break;
		case SPMVM_KERNEL_VECTORMODE:
			return "vector mode";
			break;
		case SPMVM_KERNEL_GOODFAITH:
			return "g/f hybrid";
			break;
		case SPMVM_KERNEL_TASKMODE:
			return "task mode";
			break;
		default:
			return "invalid";
			break;
	}
}

char * SpMVM_workdistName(int options)
{
	if (options & SPMVM_OPTION_WORKDIST_NZE)
		return "equal nze";
	else if (options & SPMVM_OPTION_WORKDIST_LNZE)
		return "equal lnze";
	else
		return "equal rows";
}

char * SpMVM_matrixFormatName(mat_trait_t trait) 
{
	char * name = (char *) allocateMemory(16*sizeof(char),"name");

	switch (SpMVM_matrixTraitExtractFormat(trait)) {
		case SPM_FORMAT_CRS:
			sprintf(name,"CRS");
			break;
		case SPM_FORMAT_CRSCD:
			sprintf(name,"CRS-CD");
			break;
		case SPM_FORMAT_BJDS:
			sprintf(name,"BJDS-%d",BJDS_LEN);
			break;
		case SPM_FORMAT_TBJDS:
			sprintf(name,"TBJDS-%d",BJDS_LEN);
			break;
		case SPM_FORMAT_SBJDS:
			if (SpMVM_matrixTraitExtractFlags(trait) & SPM_PERMUTECOLUMNS)
				sprintf(name,"SBJDS-PC-%d",BJDS_LEN);
			else
				sprintf(name,"SBJDS-%d",BJDS_LEN);
			break;
		case SPM_FORMAT_STBJDS:

			if (SpMVM_matrixTraitExtractFlags(trait) & SPM_PERMUTECOLUMNS)
				sprintf(name,"S%u-TBJDS-PC-%d",SpMVM_matrixTraitExtractAux(trait),BJDS_LEN);
			else
				sprintf(name,"S%u-TBJDS-%d",SpMVM_matrixTraitExtractAux(trait),BJDS_LEN);
			break;
		default:
			name = "invalid";
			break;
	}
	return name;
}

unsigned int SpMVM_matrixSize(MATRIX_TYPE *matrix) 
{
	unsigned int size = 0;

	switch (SpMVM_matrixTraitExtractFormat(matrix->trait)) {
		case SPM_FORMAT_STBJDS:
		case SPM_FORMAT_TBJDS:
			{
				BJDS_TYPE * mv= (BJDS_TYPE *)matrix->matrix;
				size = mv->nEnts*(sizeof(mat_data_t) + sizeof(int));
				size += mv->nRowsPadded/BJDS_LEN*sizeof(int); // chunkStart
				size += mv->nRowsPadded/BJDS_LEN*sizeof(int); // chunkMin
				size += mv->nRowsPadded*sizeof(int); // rowLen
				break;
			}
		case SPM_FORMAT_SBJDS:
		case SPM_FORMAT_BJDS:
			{
				BJDS_TYPE * mv= (BJDS_TYPE *)matrix->matrix;
				size = mv->nEnts*(sizeof(mat_data_t) + sizeof(int));
				size += mv->nRowsPadded/BJDS_LEN*sizeof(int); // chunkStart
				break;
			}
		case SPM_FORMAT_CRSCD:
			{
				CR_TYPE *crs = (CR_TYPE *)matrix->matrix;
				size = crs->nEnts*(sizeof(int)*sizeof(mat_data_t))+(crs->nRows+1)*sizeof(int);
				size += crs->nConstDiags*sizeof(CONST_DIAG);
				break;
			}

		default:
			return 0;
	}

	return size;
}

mat_trait_t SpMVM_stringToMatrixTrait(char *str)
{
	mat_format_t format = 0;
	mat_flags_t flags = 0;
	mat_aux_t aux = 0;

	if (!strncasecmp(str,"CRS",3)) {
		format = SPM_FORMAT_CRS;
	} else if (!strncasecmp(str,"BJDS",4)) {
		format = SPM_FORMAT_BJDS;
	} else if (!strncasecmp(str,"TBJDS",5)) {
		format = SPM_FORMAT_TBJDS;
	} else if (!strncasecmp(str,"SBJDS",5)) {
		format = SPM_FORMAT_SBJDS;
	} else if (!strncasecmp(str,"STBJDS",6)) {
		format = SPM_FORMAT_STBJDS;
		aux = (mat_aux_t)atoi(str+7);
	} else {
		DEBUG_LOG(0,"Warning! Falling back to CRS format...");
		format = SPM_FORMAT_CRS;
	}

#ifndef MPI
	flags |= SPM_GLOBAL;
#else
	flags |= SPM_DISTRIBUTED;
#endif

	mat_trait_t trait = SpMVM_createMatrixTrait(format,flags,aux);
	return trait;

}

mat_trait_t SpMVM_createMatrixTrait(mat_format_t format, mat_flags_t flags, mat_aux_t aux)
{
	return (mat_trait_t)((mat_trait_t)aux << 32 | flags << 16 | format);
}

mat_aux_t SpMVM_matrixTraitExtractAux(mat_trait_t trait)
{
	return (mat_aux_t)(trait >> 32);
}

mat_flags_t SpMVM_matrixTraitExtractFlags(mat_trait_t trait)
{
	return (mat_flags_t)(trait >> 16);
}

mat_format_t SpMVM_matrixTraitExtractFormat(mat_trait_t trait)
{
	return (mat_format_t)(trait);
}

void SpMVM_matrixTraitAddFlag(mat_trait_t * trait, mat_flags_t flag)
{
	*trait = SpMVM_createMatrixTrait(SpMVM_matrixTraitExtractFormat(*trait),
			(mat_flags_t)(SpMVM_matrixTraitExtractFlags(*trait)|flag),
			SpMVM_matrixTraitExtractAux(*trait));
}



int SpMVM_getRank() 
{
#ifdef MPI
	int rank;
	MPI_safecall(MPI_Comm_rank ( MPI_COMM_WORLD, &rank ));
	return rank;
#else
	return 0;
#endif
}

int SpMVM_getLocalRank() 
{
#ifdef MPI
	int rank;
	MPI_safecall(MPI_Comm_rank ( getSingleNodeComm(), &rank));

	return rank;
#else
	return 0;
#endif
}

int SpMVM_getNumberOfRanksOnNode()
{
#ifdef MPI
	int size;
	MPI_safecall(MPI_Comm_size ( getSingleNodeComm(), &size));

	return size;
#else
	return 1;
#endif

}
int SpMVM_getNumberOfPhysicalCores()
{
	FILE *fp;
	char nCoresS[4];
	int nCores;

	fp = popen("cat /sys/devices/system/cpu/cpu*/topology/thread_siblings_list | sort -u | wc -l","r");
	if (!fp) {
		printf("Failed to get number of physical cores\n");
	}

	fgets(nCoresS,sizeof(nCoresS)-1,fp);
	nCores = atoi(nCoresS);

	pclose(fp);

	return nCores;

}

int SpMVM_getNumberOfHwThreads()
{
	return sysconf(_SC_NPROCESSORS_ONLN);
}

int SpMVM_getNumberOfThreads() 
{
	int nthreads;
#pragma omp parallel
	nthreads = omp_get_num_threads();

	return nthreads;
}

int SpMVM_getNumberOfNodes() 
{
#ifndef MPI
	return 1;
#else
	static int stringcmp(const void *x, const void *y)
	{
		return (strcmp((char *)x, (char *)y));
	}

	int nameLen,me,size,i,distinctNames = 1;
	char name[MPI_MAX_PROCESSOR_NAME];
	char *names = NULL;

	MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD,&me));
	MPI_safecall(MPI_Comm_size(MPI_COMM_WORLD,&size));
	MPI_safecall(MPI_Get_processor_name(name,&nameLen));


	if (me==0) {
		names = (char *)allocateMemory(size*MPI_MAX_PROCESSOR_NAME*sizeof(char),
				"names");
	}


	MPI_safecall(MPI_Gather(name,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,names,
				MPI_MAX_PROCESSOR_NAME,MPI_CHAR,0,MPI_COMM_WORLD));

	if (me==0) {
		qsort(names,size,MPI_MAX_PROCESSOR_NAME*sizeof(char),stringcmp);
		for (i=1; i<size; i++) {
			if (strcmp(names+(i-1)*MPI_MAX_PROCESSOR_NAME,names+
						i*MPI_MAX_PROCESSOR_NAME)) {
				distinctNames++;
			}
		}
		free(names);
	}

	MPI_safecall(MPI_Bcast(&distinctNames,1,MPI_INT,0,MPI_COMM_WORLD));

	return distinctNames;
#endif
}


