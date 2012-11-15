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
#include <stdarg.h>

//#define PRETTYPRINT

#define PRINTWIDTH 80
#define LABELWIDTH 40

#ifdef PRETTYPRINT
#define PRINTSEP "┊"
#else
#define PRINTSEP ":"
#endif

#define VALUEWIDTH (PRINTWIDTH-LABELWIDTH-(int)strlen(PRINTSEP))


void SpMVM_printHeader(const char *label)
{
	if(SpMVM_getRank() == 0){
		const int spacing = 4;
		int len = strlen(label);
		int nDash = (PRINTWIDTH-2*spacing-len)/2;
		int rem = (PRINTWIDTH-2*spacing-len)%2;
		int i;
#ifdef PRETTYPRINT
		printf("┌");
		for (i=0; i<PRINTWIDTH-2; i++) printf("─");
		printf("┐");
		printf("\n");
		printf("├");
		for (i=0; i<nDash-1; i++) printf("─");
		for (i=0; i<spacing; i++) printf(" ");
		printf("%s",label);
		for (i=0; i<spacing+rem; i++) printf(" ");
		for (i=0; i<nDash-1; i++) printf("─");
		printf("┤");
		printf("\n");
		printf("├");
		for (i=0; i<LABELWIDTH; i++) printf("─");
		printf("┬");
		for (i=0; i<VALUEWIDTH; i++) printf("─");
		printf("┤");
		printf("\n");
#else
		for (i=0; i<PRINTWIDTH; i++) printf("-");
		printf("\n");
		for (i=0; i<nDash; i++) printf("-");
		for (i=0; i<spacing; i++) printf(" ");
		printf("%s",label);
		for (i=0; i<spacing+rem; i++) printf(" ");
		for (i=0; i<nDash; i++) printf("-");
		printf("\n");
		for (i=0; i<PRINTWIDTH; i++) printf("-");
		printf("\n");
#endif
	}
}

void SpMVM_printFooter() 
{
	if (SpMVM_getRank() == 0) {
		int i;
#ifdef PRETTYPRINT
		printf("└");
		for (i=0; i<LABELWIDTH; i++) printf("─");
		printf("┴");
		for (i=0; i<VALUEWIDTH; i++) printf("─");
		printf("┘");
#else
		for (i=0; i<PRINTWIDTH; i++) printf("-");
#endif
		printf("\n\n");
	}
}

void SpMVM_printLine(const char *label, const char *unit, const char *fmt, ...)
{
	va_list args;
	va_start(args,fmt);
	char dummy[1024];
	vsnprintf(dummy,1024,fmt,args);
	va_end(args);

#ifdef PRETTYPRINT
	printf("│");
#endif
	if (unit) {
		int unitLen = strlen(unit);
		printf("%-*s (%s)%s%*s",LABELWIDTH-unitLen-3,label,unit,PRINTSEP,VALUEWIDTH,dummy);
	} else {
		printf("%-*s%s%*s",LABELWIDTH,label,PRINTSEP,VALUEWIDTH,dummy);
	}
#ifdef PRETTYPRINT
	printf("│");
#endif
	printf("\n");
}

void SpMVM_printSetupInfo(ghost_setup_t *setup, int options)
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
		ws = ((setup->nrows+1)*sizeof(mat_idx_t) + 
				setup->nnz*(sizeof(mat_data_t)+sizeof(mat_idx_t)))/(1024*1024);

		char *matrixLocation = (char *)allocateMemory(64,"matrixLocation");
		if (setup->flags & SETUP_HOSTANDDEVICE)
			matrixLocation = "Host and Device";
		else if (setup->flags & SETUP_DEVICEONLY)
			matrixLocation = "Device only";
		else
			matrixLocation = "Host only";

		char *matrixPlacement = (char *)allocateMemory(64,"matrixPlacement");
		if (setup->flags & SETUP_DISTRIBUTED)
			matrixLocation = "Distributed";
		else if (setup->flags & SETUP_GLOBAL)
			matrixLocation = "Global";


		SpMVM_printHeader("Matrix information");
		SpMVM_printLine("Matrix name",NULL,"%s",setup->matrixName);
		SpMVM_printLine("Dimension",NULL,"%"PRmatIDX,setup->nrows);
		SpMVM_printLine("Nonzeros",NULL,"%"PRmatNNZ,setup->nnz);
		SpMVM_printLine("Avg. nonzeros per row",NULL,"%.3f",(double)setup->nnz/setup->nrows);
		SpMVM_printLine("Matrix location",NULL,"%s",matrixLocation);
		SpMVM_printLine("Matrix placement",NULL,"%s",matrixPlacement);
		SpMVM_printLine("Global CRS size","MB","%lu",ws);

		SpMVM_printLine("Full   host matrix format",NULL,"%s",SpMVM_matrixFormatName(setup->fullMatrix->trait));
		if (setup->flags & SETUP_DISTRIBUTED)
		{
			SpMVM_printLine("Local  host matrix format",NULL,"%s",SpMVM_matrixFormatName(setup->localMatrix->trait));
			SpMVM_printLine("Remote host matrix format",NULL,"%s",SpMVM_matrixFormatName(setup->remoteMatrix->trait));
		}
		SpMVM_printLine("Full   host matrix size (rank 0)","MB","%u",SpMVM_matrixSize(setup->fullMatrix)/(1024*1024));
		if (setup->flags & SETUP_DISTRIBUTED)
		{
			SpMVM_printLine("Local  host matrix size (rank 0)","MB","%u",SpMVM_matrixSize(setup->localMatrix)/(1024*1024));
			SpMVM_printLine("Remote host matrix size (rank 0)","MB","%u",SpMVM_matrixSize(setup->remoteMatrix)/(1024*1024));
		}

		if (setup->flags & SETUP_GLOBAL)
		{ //additional information depending on format
			switch (setup->fullMatrix->trait.format) {
				case SPM_FORMAT_STBJDS:
					SpMVM_printLine("Row length oscillation nu",NULL,"%f",((BJDS_TYPE *)(setup->fullMatrix->data))->nu);
				case SPM_FORMAT_SBJDS:
					SpMVM_printLine("Sort block size",NULL,"%u",*(unsigned int *)(setup->fullMatrix->trait.aux));
					SpMVM_printLine("Permuted columns",NULL,"%s",setup->fullMatrix->trait.flags&SPM_PERMUTECOLIDX?"yes":"no");
				case SPM_FORMAT_BJDS:
				case SPM_FORMAT_TBJDS:
					SpMVM_printLine("Block size",NULL,"%d",BJDS_LEN);
					break;
			}
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
		SpMVM_printFooter();

		SpMVM_printHeader("Setup information");
		SpMVM_printLine("Equation",NULL,"%s",options&SPMVM_OPTION_AXPY?"y <- y+A*x":"y <- A*x");
		SpMVM_printLine("Work distribution scheme",NULL,"%s",SpMVM_workdistName(options));
		SpMVM_printLine("Automatic pinning",NULL,"%s",pin?"enabled":"disabled");
		if (pin)
			SpMVM_printLine("Pinning threads to ",NULL,"%s",pinStrategy);
		SpMVM_printFooter();
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

		SpMVM_printHeader("System information");
		SpMVM_printLine("Nodes",NULL,"%d",nnodes);
		SpMVM_printLine("MPI processes per node",NULL,"%d",nproc/nnodes);
		SpMVM_printLine("Avail. threads (phys/HW) per node",NULL,"%d/%d",nphyscores,ncores);
		SpMVM_printLine("OpenMP threads per node",NULL,"%d",nproc/nnodes*nthreads);
		SpMVM_printLine("OpenMP threads per process",NULL,"%d",nthreads);
		SpMVM_printLine("OpenMP scheduling",NULL,"%s",omp_sched_str);
#ifdef OPENCL
		// TODO
		printf("OpenCL devices                   :\n");
		int i;
		for (i=0; i<devInfo->nDistinctDevices; i++) {
			printf("                            %3d x %13s\n",devInfo->nDevices[i],devInfo->names[i]);
		}
#endif
		SpMVM_printFooter();

		SpMVM_printHeader("LibSpMVM information");
		SpMVM_printLine("Version",NULL,"%s",LIBSPMVM_VERSION);
		SpMVM_printLine("Build date",NULL,"%s",__DATE__);
		SpMVM_printLine("Build time",NULL,"%s",__TIME__);
		SpMVM_printLine("Data type",NULL,"%s",DATATYPE_NAMES[DATATYPE_DESIRED]);
#ifdef MIC
		SpMVM_printLine("MIC kernels",NULL,"enabled");
#else
		SpMVM_printLine("MIC kernels",NULL,"disabled");
#endif
#ifdef AVX
		SpMVM_printLine("AVX kernels",NULL,"enabled");
#else
		SpMVM_printLine("AVX kernels",NULL,"disabled");
#endif
#ifdef SSE
		SpMVM_printLine("SSE kernels",NULL,"enabled");
#else
		SpMVM_printLine("SSE kernels",NULL,"disabled");
#endif
#ifdef MPI
		SpMVM_printLine("MPI support",NULL,"enabled");
#else
		SpMVM_printLine("MPI support",NULL,"disabled");
#endif
#ifdef OPENCL
		SpMVM_printLine("OpenCL support",NULL,"enabled");
#else
		SpMVM_printLine("OpenCL support",NULL,"disabled");
#endif
#ifdef LIKWID
		SpMVM_printLine("Likwid support",NULL,"enabled");
		printf("Likwid support                   :      enabled\n");
		/*#ifdef LIKWID_MARKER_FINE
		  printf("Likwid Marker API (high res)     :      enabled\n");
#else
#ifdef LIKWID_MARKER
printf("Likwid Marker API                :      enabled\n");
#endif
#endif*/
#else
		SpMVM_printLine("Likwid support",NULL,"disabled");
#endif
		SpMVM_printFooter();

	}
#ifdef OPENCL
	destroyCLdeviceInfo(devInfo);
#endif


}

VECTOR_TYPE *SpMVM_referenceSolver(char *matrixPath, ghost_setup_t *distSetup, mat_data_t (*rhsVal)(int), int nIter, int spmvmOptions)
{

	DEBUG_LOG(1,"Computing reference solution");
	int me = SpMVM_getRank();
	//VECTOR_TYPE *lhs = SpMVM_createVector(distSetup,VECTOR_TYPE_LHS|VECTOR_TYPE_HOSTONLY,NULL);
	VECTOR_TYPE *globLHS; 

	if (me==0) {
	mat_trait_t trait = {.format = SPM_FORMAT_CRS, .flags = SPM_DEFAULT, .aux = NULL};
	ghost_setup_t *setup = SpMVM_createSetup(matrixPath, &trait, 1, SETUP_GLOBAL|SETUP_HOSTONLY, NULL);
		globLHS = SpMVM_createVector(setup,VECTOR_TYPE_LHS|VECTOR_TYPE_HOSTONLY,NULL); 
		VECTOR_TYPE *globRHS = SpMVM_createVector(setup,VECTOR_TYPE_RHS|VECTOR_TYPE_HOSTONLY,rhsVal);

		CR_TYPE *cr = (CR_TYPE *)(setup->fullMatrix->data);
		int iter;

		for (iter=0; iter<nIter; iter++)
			SpMVM_referenceKernel(globLHS->val, cr->col, cr->rpt, cr->val, globRHS->val, cr->nrows, spmvmOptions);
	} else {
		globLHS = SpMVM_newVector(0,VECTOR_TYPE_LHS|VECTOR_TYPE_HOSTONLY);
	}
	MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));

	DEBUG_LOG(1,"Scattering result of reference solution");

	VECTOR_TYPE *lhs = SpMVM_distributeVector(distSetup->communicator,globLHS);
		
	DEBUG_LOG(1,"Reference solution has been computed and scattered successfully");
	return lhs;

}

void SpMVM_zeroVector(VECTOR_TYPE *vec) 
{
	int i;
	for (i=0; i<vec->nrows; i++) {
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

VECTOR_TYPE* SpMVM_newVector( const int nrows, vec_flags_t flags ) 
{
	VECTOR_TYPE* vec;
	size_t size_val;
	int i;

	size_val = (size_t)( nrows * sizeof(mat_data_t) );
	vec = (VECTOR_TYPE*) allocateMemory( sizeof( VECTOR_TYPE ), "vec");


	vec->val = (mat_data_t*) allocateMemory( size_val, "vec->val");
	vec->nrows = nrows;
	vec->flags = flags;

#pragma omp parallel for schedule(runtime) 
	for( i = 0; i < nrows; i++ ) 
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

VECTOR_TYPE * SpMVM_distributeVector(ghost_comm_t *lcrp, VECTOR_TYPE *vec)
{
	DEBUG_LOG(1,"Distributing vector");
	int me = SpMVM_getRank();

	int nrows;

	MPI_safecall(MPI_Bcast(&(vec->flags),1,vec_flags_t_MPI,0,MPI_COMM_WORLD));

	if (vec->flags & VECTOR_TYPE_LHS)
		nrows = lcrp->lnrows[me];
	else if ((vec->flags & VECTOR_TYPE_RHS) || (vec->flags & VECTOR_TYPE_BOTH))
		nrows = lcrp->lnrows[me]+lcrp->halo_elements;
	else
		ABORT("No valid type for vector (has to be one of VECTOR_TYPE_LHS/_RHS/_BOTH");


	DEBUG_LOG(2,"Creating local vector with %d rows",nrows);
	VECTOR_TYPE *nodeVec = SpMVM_newVector( nrows, vec->flags ); 

	DEBUG_LOG(2,"Scattering global vector to local vectors");
#ifdef MPI
	MPI_safecall(MPI_Scatterv ( vec->val, (int *)lcrp->lnrows, (int *)lcrp->lfRow, MPI_MYDATATYPE,
				nodeVec->val, (int)lcrp->lnrows[me], MPI_MYDATATYPE, 0, MPI_COMM_WORLD ));
#else
	int i;
	for (i=0; i<vec->nrows; i++) nodeVec->val[i] = vec->val[i];
#endif
#ifdef OPENCL // TODO depending on flag
	CL_uploadVector(nodeVec);
#endif

	DEBUG_LOG(1,"Vector distributed successfully");
	return nodeVec;
}

void SpMVM_collectVectors(ghost_setup_t *setup, VECTOR_TYPE *vec, 
		VECTOR_TYPE *totalVec, int kernel) {

#ifdef MPI
	// TODO
	//if (matrix->trait.format != SPM_FORMAT_CRS)
	//	DEBUG_LOG(0,"Cannot handle other matrices than CRS in the MPI case!");

	int me = SpMVM_getRank();
	if ( 0x1<<kernel & SPMVM_KERNELS_COMBINED)  {
		SpMVM_permuteVector(vec->val,setup->fullMatrix->invRowPerm,setup->communicator->lnrows[me]);
	} else if ( 0x1<<kernel & SPMVM_KERNELS_SPLIT ) {
		// one of those must return immediately
		SpMVM_permuteVector(vec->val,setup->localMatrix->invRowPerm,setup->communicator->lnrows[me]);
		SpMVM_permuteVector(vec->val,setup->remoteMatrix->invRowPerm,setup->communicator->lnrows[me]);
	}
	MPI_safecall(MPI_Gatherv(vec->val,(int)setup->communicator->lnrows[me],MPI_MYDATATYPE,totalVec->val,
				(int *)setup->communicator->lnrows,(int *)setup->communicator->lfRow,MPI_MYDATATYPE,0,MPI_COMM_WORLD));
#else
	int i;
	UNUSED(kernel);
	SpMVM_permuteVector(vec->val,setup->fullMatrix->invRowPerm,setup->fullMatrix->nrows);
	for (i=0; i<totalVec->nrows; i++) 
		totalVec->val[i] = vec->val[i];
#endif
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

	for (i=0; i<vec->nrows; i++)	
		sum += vec->val[i]*vec->val[i];

	mat_data_t f = (mat_data_t)1/SQRT(ABS(sum));

	for (i=0; i<vec->nrows; i++)	
		vec->val[i] *= f;

#ifdef OPENCL
	CL_uploadVector(vec);
#endif
}

void SpMVM_freeVector( VECTOR_TYPE* const vec ) 
{
	if( vec ) {
		freeMemory( (size_t)(vec->nrows*sizeof(mat_data_t)), "vec->val",  vec->val );
#ifdef OPENCL
		CL_freeDeviceMemory( vec->CL_val_gpu );
#endif
		free( vec );
	}
}

unsigned int SpMVM_getRowLen( ghost_mat_t *matrix, mat_idx_t i)
{
	switch(matrix->trait.format) {
		case SPM_FORMAT_CRS:
		case SPM_FORMAT_CRSCD:
			{
				CR_TYPE *cr = (CR_TYPE *)(matrix->data);
				return cr->rpt[i+1]-cr->rpt[i];
			}
		case SPM_FORMAT_BJDS:
		case SPM_FORMAT_SBJDS:
		case SPM_FORMAT_STBJDS:
		case SPM_FORMAT_TBJDS:
			{
				BJDS_TYPE *bjds = (BJDS_TYPE *)(matrix->data);
				return bjds->rowLen[i];
			}
		default:
			return 0;
	}
}

void SpMVM_freeLCRP( ghost_comm_t* const lcrp ) 
{
	if( lcrp ) {
		/*		free( lcrp->lnEnts );
				free( lcrp->lnrows );
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
				free( lcrp->fullRowPerm );
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

void SpMVM_permuteVector( mat_data_t* vec, mat_idx_t* perm, mat_idx_t len) 
{
	/* permutes values in vector so that i-th entry is mapped to position perm[i] */
	mat_idx_t i;
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
			ABORT("Permutation index out of bounds: %"PRmatIDX" > %"PRmatIDX,perm[i],len);
		}
		tmp[perm[i]] = vec[i];
	}
	for(i=0; i < len; ++i) {
		vec[i] = tmp[i];
	}

	free(tmp);
}

SpMVM_kernelFunc SpMVM_selectKernelFunc(int options, int kernel, ghost_setup_t *setup) 
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


	UNUSED(setup);
	// TODO format muss nicht bitweise sein weil eh nur eins geht

	/*	switch (mat->trait.format) {
		case SPM_FORMAT_CRS:
		switch (kernel) {
		case SPMVM_KERNEL_NOMPI:
		if (mat->trait.flags & SPM_DISTRIBUTED)
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
kernelFunc = (SpMVM_kernelFunc)&mic_kernel_0_intr_16;
break;
#endif
#ifdef AVX
case SPMVM_KERNEL_NOMPI:
kernelFunc = (SpMVM_kernelFunc)&avx_kernel_0_intr;
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
DEBUG_LOG(1,"Non-valid matrix format specified (%hu)!",mat->trait.format);
return NULL;
}*/


return kernelFunc;


}

SpMVM_kernelFunc * SpMVM_setupKernels(ghost_setup_t *setup)
{
	int i;
	SpMVM_kernelFunc *kf = (SpMVM_kernelFunc *)allocateMemory(SPM_NUMFORMATS*sizeof(SpMVM_kernelFunc),"kf");

	for (i=0; i<SPM_NUMFORMATS; i++)
		kf[i] = NULL;

	UNUSED(setup);

	kf[SPM_FORMAT_CRS] = (SpMVM_kernelFunc)&kern_glob_CRS_0;
	kf[SPM_FORMAT_CRSCD] = (SpMVM_kernelFunc)&kern_glob_CRS_CD_0;
#ifdef MIC
	kf[SPM_FORMAT_SBJDS] = (SpMVM_kernelFunc)&mic_kernel_0_intr_16; 
	kf[SPM_FORMAT_BJDS] = (SpMVM_kernelFunc)&mic_kernel_0_intr_16; 
	kf[SPM_FORMAT_STBJDS] = (SpMVM_kernelFunc)&mic_kernel_0_intr_16_rem; 
	kf[SPM_FORMAT_TBJDS] = (SpMVM_kernelFunc)&mic_kernel_0_intr_16_rem;
#endif
#ifdef SSE
	kf[SPM_FORMAT_SBJDS] = (SpMVM_kernelFunc)&sse_kernel_0_intr; 
	kf[SPM_FORMAT_BJDS] = (SpMVM_kernelFunc)&sse_kernel_0_intr; 
	kf[SPM_FORMAT_STBJDS] = (SpMVM_kernelFunc)&sse_kernel_0_intr_rem; 
	kf[SPM_FORMAT_TBJDS] = (SpMVM_kernelFunc)&sse_kernel_0_intr_rem;
#endif
#ifdef AVX
	kf[SPM_FORMAT_SBJDS] = (SpMVM_kernelFunc)&avx_kernel_0_intr; 
	kf[SPM_FORMAT_BJDS] = (SpMVM_kernelFunc)&avx_kernel_0_intr; 
	kf[SPM_FORMAT_STBJDS] = (SpMVM_kernelFunc)&avx_kernel_0_intr_rem;
	if (setup->fullMatrix->trait.flags & SPM_FLAG_COLMAJOR)
		kf[SPM_FORMAT_TBJDS] = (SpMVM_kernelFunc)&avx_kernel_0_intr_rem_if;
	else 
		kf[SPM_FORMAT_TBJDS] = (SpMVM_kernelFunc)&avx_kernel_0_intr_rem;
#endif

	return kf;
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

// TODO matrix als argument
char * SpMVM_matrixFormatName(mat_trait_t trait) 
{
	char * name = (char *) allocateMemory(16*sizeof(char),"name");

	switch (trait.format) {
		case SPM_FORMAT_CRS:
			sprintf(name,"CRS");
			break;
		case SPM_FORMAT_CRSCD:
			sprintf(name,"CRS-CD");
			break;
		case SPM_FORMAT_SBJDS:
		case SPM_FORMAT_BJDS:
			sprintf(name,"BJDS");
			break;
		case SPM_FORMAT_STBJDS:
		case SPM_FORMAT_TBJDS:
			sprintf(name,"TBJDS");
			break;
		default:
			name = "invalid";
			break;
	}
	return name;
}

unsigned int SpMVM_matrixSize(ghost_mat_t *matrix) 
{
	unsigned int size = 0;

	switch (matrix->trait.format) {
		case SPM_FORMAT_STBJDS:
		case SPM_FORMAT_TBJDS:
			{
				BJDS_TYPE * mv= (BJDS_TYPE *)matrix->data;
				size = mv->nEnts*(sizeof(mat_data_t) + sizeof(int));
				size += mv->nrowsPadded/BJDS_LEN*sizeof(int); // chunkStart
				size += mv->nrowsPadded/BJDS_LEN*sizeof(int); // chunkMin
				size += mv->nrowsPadded*sizeof(int); // rowLen
				break;
			}
		case SPM_FORMAT_SBJDS:
		case SPM_FORMAT_BJDS:
			{
				BJDS_TYPE * mv= (BJDS_TYPE *)matrix->data;
				size = mv->nEnts*(sizeof(mat_data_t) + sizeof(int));
				size += mv->nrowsPadded/BJDS_LEN*sizeof(int); // chunkStart
				break;
			}
		case SPM_FORMAT_CRS:
			{
				CR_TYPE *crs = (CR_TYPE *)matrix->data;
				size = crs->nEnts*(sizeof(mat_idx_t)+sizeof(mat_data_t)) + (crs->nrows+1)*sizeof(mat_idx_t);
				break;
			}
		case SPM_FORMAT_CRSCD:
			{
				CR_TYPE *crs = (CR_TYPE *)matrix->data;
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
	mat_trait_t trait = {.format = 0, .flags = 0, .aux = NULL};

	if (!strncasecmp(str,"CRS",3)) {
		trait.format = SPM_FORMAT_CRS;
	} else if (!strncasecmp(str,"BJDS",4)) {
		trait.format = SPM_FORMAT_BJDS;
	} else if (!strncasecmp(str,"TBJDS",5)) {
		trait.format = SPM_FORMAT_TBJDS;
	} else if (!strncasecmp(str,"SBJDS",5)) {
		trait.format = SPM_FORMAT_SBJDS;
	} else if (!strncasecmp(str,"STBJDS",6)) {
		trait.format = SPM_FORMAT_STBJDS;
		trait.aux = (unsigned int *) allocateMemory(sizeof(unsigned int),"aux");
		*(unsigned int *)trait.aux = (unsigned int)atoi(str+7);
	} else {
		DEBUG_LOG(0,"Warning! Falling back to CRS format...");
		trait.format = SPM_FORMAT_CRS;
	}

#ifndef MPI
	trait.flags |= SETUP_GLOBAL;
#else
	trait.flags |= SETUP_DISTRIBUTED;
#endif

	return trait;
}

mat_trait_t SpMVM_createMatrixTrait(mat_format_t format, mat_flags_t flags, void * aux)
{
	mat_trait_t trait = {.format = format, .flags = flags, .aux = aux};
	return trait;
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

unsigned int SpMVM_getNumberOfNodes() 
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

unsigned int SpMVM_getNumberOfProcesses() 
{
#ifndef MPI
	return 1;
#else

	int nnodes;

	MPI_safecall(MPI_Comm_size(MPI_COMM_WORLD, &nnodes));

	return (unsigned int)nnodes;
#endif
}
