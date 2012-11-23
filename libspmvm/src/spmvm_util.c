#define _GNU_SOURCE
#include "ghost_util.h"
#include "ghost.h"
#include "ghost_vec.h"
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

static int allocatedMem;

void SpMVM_printHeader(const char *fmt, ...)
{
	if(SpMVM_getRank() == 0){
		va_list args;
		va_start(args,fmt);
		char label[1024];
		vsnprintf(label,1024,fmt,args);
		va_end(args);
		
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
	if (SpMVM_getRank() == 0) {
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
}

void SpMVM_printSetupInfo(ghost_setup_t *setup, int options)
{

	int me;
	size_t ws;


	me = SpMVM_getRank();

/*#ifdef OPENCL	
	size_t fullMemSize, localMemSize, remoteMemSize, 
		   totalFullMemSize = 0, totalLocalMemSize = 0, totalRemoteMemSize = 0;

	if (!(options & GHOST_OPTION_NO_COMBINED_KERNELS)) { // combined computation
		fullMemSize = getBytesize(matrix->devMatrix->fullMatrix, matrix->devMatrix->fullFormat)/
			(1024*1024);
		MPI_safecall(MPI_Reduce(&fullMemSize, &totalFullMemSize,1,MPI_LONG,MPI_SUM,0,
					MPI_COMM_WORLD));
	} 
	if (!(options & GHOST_OPTION_NO_SPLIT_KERNELS)) { // split computation
		localMemSize = getBytesize(matrix->devMatrix->localMatrix,matrix->devMatrix->localFormat)/
			(1024*1024);
		remoteMemSize = getBytesize(matrix->devMatrix->remoteMatrix,matrix->devMatrix->remoteFormat)/
			(1024*1024);
		MPI_safecall(MPI_Reduce(&localMemSize, &totalLocalMemSize,1,MPI_LONG,MPI_SUM,0,
					MPI_COMM_WORLD));
		MPI_safecall(MPI_Reduce(&remoteMemSize, &totalRemoteMemSize,1,MPI_LONG,MPI_SUM,0,
					MPI_COMM_WORLD));
	}
#endif	*/

	if(me==0){
		int pin = (options & GHOST_OPTION_PIN || options & GHOST_OPTION_PIN_SMT)?
			1:0;
		char *pinStrategy = options & GHOST_OPTION_PIN?"phys. cores":"virt. cores";
		ws = ((setup->nrows+1)*sizeof(mat_idx_t) + 
				setup->nnz*(sizeof(mat_data_t)+sizeof(mat_idx_t)))/(1024*1024);

		char *matrixLocation = (char *)allocateMemory(64,"matrixLocation");
	/*	if (setup->flags & GHOST_SETUP_HOST && setup->flags & GHOST_SETUP_DEVICE)
			matrixLocation = "Host and Device";
		else if (setup->flags & GHOST_SETUP_DEVICE)
			matrixLocation = "Device only";
		else
			matrixLocation = "Host only";*/

		char *matrixPlacement = (char *)allocateMemory(64,"matrixPlacement");
		if (setup->flags & GHOST_SETUP_DISTRIBUTED)
			matrixLocation = "Distributed";
		else if (setup->flags & GHOST_SETUP_GLOBAL)
			matrixLocation = "Global";


		SpMVM_printHeader("Matrix information");
		SpMVM_printLine("Matrix name",NULL,"%s",setup->matrixName);
		SpMVM_printLine("Dimension",NULL,"%"PRmatIDX,setup->nrows);
		SpMVM_printLine("Nonzeros",NULL,"%"PRmatNNZ,setup->nnz);
		SpMVM_printLine("Avg. nonzeros per row",NULL,"%.3f",(double)setup->nnz/setup->nrows);
		SpMVM_printLine("Matrix location",NULL,"%s",matrixLocation);
		SpMVM_printLine("Matrix placement",NULL,"%s",matrixPlacement);
		SpMVM_printLine("Global CRS size","MB","%lu",ws);

		SpMVM_printLine("Full   host matrix format",NULL,"%s",setup->fullMatrix->formatName());
		if (setup->flags & GHOST_SETUP_DISTRIBUTED)
		{
			SpMVM_printLine("Local  host matrix format",NULL,"%s",setup->localMatrix->formatName());
			SpMVM_printLine("Remote host matrix format",NULL,"%s",setup->remoteMatrix->formatName());
		}
		SpMVM_printLine("Full   host matrix size (rank 0)","MB","%u",setup->fullMatrix->byteSize()/(1024*1024));
		if (setup->flags & GHOST_SETUP_DISTRIBUTED)
		{
			SpMVM_printLine("Local  host matrix size (rank 0)","MB","%u",setup->localMatrix->byteSize()/(1024*1024));
			SpMVM_printLine("Remote host matrix size (rank 0)","MB","%u",setup->remoteMatrix->byteSize()/(1024*1024));
		}
		
		if (setup->flags & GHOST_SETUP_GLOBAL)
		{ //additional information depending on format
			setup->fullMatrix->printInfo();
		}
/*#ifdef OPENCL	
		if (!(options & GHOST_OPTION_NO_COMBINED_KERNELS)) { // combined computation
			printf("Dev. matrix (combin.%4s-%2d) [MB]: %12lu\n", GHOST_SPMFORMAT_NAMES[matrix->devMatrix->fullFormat],matrix->devMatrix->fullT,totalFullMemSize);
		}	
		if (!(options & GHOST_OPTION_NO_SPLIT_KERNELS)) { // split computation
			printf("Dev. matrix (local  %4s-%2d) [MB]: %12lu\n", GHOST_SPMFORMAT_NAMES[matrix->devMatrix->localFormat],matrix->devMatrix->localT,totalLocalMemSize); 
			printf("Dev. matrix (remote %4s-%2d) [MB]: %12lu\n", GHOST_SPMFORMAT_NAMES[matrix->devMatrix->remoteFormat],matrix->devMatrix->remoteT,totalRemoteMemSize);
			printf("Dev. matrix (local & remote) [MB]: %12lu\n", totalLocalMemSize+
					totalRemoteMemSize); 
		}
#endif*/
		SpMVM_printFooter();

		SpMVM_printHeader("Setup information");
		SpMVM_printLine("Equation",NULL,"%s",options&GHOST_OPTION_AXPY?"y <- y+A*x":"y <- A*x");
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
/*		// TODO
		printf("OpenCL devices                   :\n");
		int i;
		for (i=0; i<devInfo->nDistinctDevices; i++) {
			printf("                            %3d x %13s\n",devInfo->nDevices[i],devInfo->names[i]);
		}*/
#endif
		SpMVM_printFooter();

		SpMVM_printHeader("%s information", GHOST_NAME);
		SpMVM_printLine("Version",NULL,"%s",GHOST_VERSION);
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

ghost_vec_t *SpMVM_referenceSolver(char *matrixPath, ghost_setup_t *distSetup, mat_data_t (*rhsVal)(int), int nIter, int spmvmOptions)
{

	DEBUG_LOG(1,"Computing reference solution");
	int me = SpMVM_getRank();
	//ghost_vec_t *lhs = SpMVM_createVector(distSetup,ghost_vec_t_LHS|ghost_vec_t_HOSTONLY,NULL);
	ghost_vec_t *globLHS; 

	if (me==0) {
		mat_trait_t trait = {.format = "CRS", .flags = GHOST_SPM_HOST, .aux = NULL};
		ghost_setup_t *setup = SpMVM_createSetup(matrixPath, &trait, 1, GHOST_SETUP_GLOBAL, NULL);
		globLHS = SpMVM_createVector(setup,ghost_vec_t_LHS|ghost_vec_t_HOSTONLY,NULL); 
		ghost_vec_t *globRHS = SpMVM_createVector(setup,ghost_vec_t_RHS|ghost_vec_t_HOSTONLY,rhsVal);

		CR_TYPE *cr = (CR_TYPE *)(setup->fullMatrix->data);
		printf("ref: %p %p\n",cr,cr->clmat);
		int iter;

		for (iter=0; iter<nIter; iter++)
			SpMVM_referenceKernel(globLHS->val, cr->col, cr->rpt, cr->val, globRHS->val, cr->nrows, spmvmOptions);
	} else {
		globLHS = SpMVM_newVector(0,ghost_vec_t_LHS|ghost_vec_t_HOSTONLY);
	}
	DEBUG_LOG(1,"Scattering result of reference solution");

	ghost_vec_t *lhs = SpMVM_distributeVector(distSetup->communicator,globLHS);

	DEBUG_LOG(1,"Reference solution has been computed and scattered successfully");
	return lhs;

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


char * SpMVM_modeName(int mode) 
{

	switch (mode) {
		case GHOST_MODE_NOMPI:
			return "non-MPI";
			break;
		case GHOST_MODE_VECTORMODE:
			return "vector mode";
			break;
		case GHOST_MODE_GOODFAITH:
			return "g/f hybrid";
			break;
		case GHOST_MODE_TASKMODE:
			return "task mode";
			break;
		default:
			return "invalid";
			break;
	}
}

char * SpMVM_workdistName(int options)
{
	if (options & GHOST_OPTION_WORKDIST_NZE)
		return "equal nze";
	else if (options & GHOST_OPTION_WORKDIST_LNZE)
		return "equal lnze";
	else
		return "equal rows";
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

void* allocateMemory( const size_t size, const char* desc ) 
{

	/* allocate size bytes of posix-aligned memory;
	 * check for success and increase global counter */

	size_t boundary = 1024;
	int ierr;

	void* mem;

	DEBUG_LOG(2,"Allocating %8.2f MB of memory for %-18s  -- %6.3f", 
			size/(1024.0*1024.0), desc, (1.0*allocatedMem)/(1024.0*1024.0));

	if (  (ierr = posix_memalign(  (void**) &mem, boundary, size)) != 0 ) {
		ABORT("Error while allocating using posix_memalign: %s",strerror(ierr));
	}

	if( ! mem ) {
		ABORT("Error in memory allocation of %lu bytes for %s",size,desc);
	}

	allocatedMem += size;
	return mem;
}

void freeMemory( size_t size, const char* desc, void* this_array ) 
{

	DEBUG_LOG(2,"Freeing %8.2f MB of memory for %s", size/(1024.*1024.), desc);

	allocatedMem -= size;
	free (this_array);

}
