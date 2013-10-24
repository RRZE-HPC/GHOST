#define _GNU_SOURCE
#include "ghost_util.h"
#include "ghost.h"
#include "ghost_vec.h"
#include "ghost_mat.h"
#include <sys/param.h>
#include <libgen.h>
#include <unistd.h>
#include <byteswap.h>

#ifdef LIKWID
#include <likwid.h>
#endif

#include <sched.h>
#include <errno.h>
#include <omp.h>
#include <string.h>
#include <stdarg.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/syscall.h>
#include <dirent.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <fcntl.h>


//#define PRETTYPRINT

#define PRINTWIDTH 80
#define LABELWIDTH 40

#ifdef PRETTYPRINT
#define PRINTSEP "┊"
#else
#define PRINTSEP ":"
#endif

#define VALUEWIDTH (PRINTWIDTH-LABELWIDTH-(int)strlen(PRINTSEP))

#define GHOST_MAX_NTASKS 1024
//static int allocatedMem;

//ghost_threadstate_t *threadpool = NULL;
//static ghost_task_t *tasklist[GHOST_MAX_NTASKS];
//static int nTasks = 0;

extern char ** environ;

double ghost_wctime()
{
	/*	struct timespec ts;
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&ts);
		return (double)(ts.tv_sec + ts.tv_nsec/1.e9);*/
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return (double) (tp.tv_sec + tp.tv_usec/1000000.0);
}
/*static double ghost_timediff(struct timespec start, struct timespec end)
  {
  struct timespec tmp;
  if (end.tv_nsec-start.tv_nsec < 0) {
  tmp.tv_sec = end.tv_sec-start.tv_sec-1;
  tmp.tv_nsec = 1e9+end.tv_nsec-start.tv_nsec;
  } else {
  tmp.tv_sec = end.tv_sec-start.tv_sec;
  tmp.tv_nsec = end.tv_nsec-start.tv_nsec;

  }

  printf("%ld:%ld\n",tmp.tv_sec,tmp.tv_nsec);
  return tmp.tv_sec + tmp.tv_nsec/1.e9;



  }*/

void ghost_printHeader(const char *fmt, ...)
{
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

void ghost_printFooter() 
{
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

void ghost_printLine(const char *label, const char *unit, const char *fmt, ...)
{
		va_list args;
		va_start(args,fmt);
		char dummy[1025];
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


void ghost_printMatrixInfo(ghost_mat_t *mat)
{
	ghost_midx_t nrows = ghost_getMatNrows(mat);
	ghost_midx_t nnz = ghost_getMatNnz(mat);
	
	int myrank;

   	if (mat->context->communicator != NULL) {
		myrank = ghost_getRank(mat->context->mpicomm); 
	} else {
		myrank = 0;
	}
	if (myrank == 0) {

	char *matrixLocation;
	if (mat->traits->flags & GHOST_SPM_DEVICE)
		matrixLocation = "Device";
	else if (mat->traits->flags & GHOST_SPM_HOST)
		matrixLocation = "Host";
	else
		matrixLocation = "Default";


	ghost_printHeader(mat->name);
	ghost_printLine("Data type",NULL,"%s",ghost_datatypeName(mat->traits->datatype));
	ghost_printLine("Matrix location",NULL,"%s",matrixLocation);
	ghost_printLine("Number of rows",NULL,"%"PRmatIDX,nrows);
	ghost_printLine("Number of nonzeros",NULL,"%"PRmatNNZ,nnz);
	ghost_printLine("Avg. nonzeros per row",NULL,"%.3f",(double)nrows/nnz);

	ghost_printLine("Full   matrix format",NULL,"%s",mat->formatName(mat));
	if (mat->context->flags & GHOST_CONTEXT_DISTRIBUTED)
	{
		ghost_printLine("Local  matrix format",NULL,"%s",mat->localPart->formatName(mat->localPart));
		ghost_printLine("Remote matrix format",NULL,"%s",mat->remotePart->formatName(mat->remotePart));
		ghost_printLine("Local  matrix symmetry",NULL,"%s",ghost_symmetryName(mat->localPart->symmetry));
	} else {
		ghost_printLine("Full   matrix symmetry",NULL,"%s",ghost_symmetryName(mat->symmetry));
	}

	ghost_printLine("Full   matrix size (rank 0)","MB","%u",mat->byteSize(mat)/(1024*1024));
	if (mat->context->flags & GHOST_CONTEXT_DISTRIBUTED)
	{
		ghost_printLine("Local  matrix size (rank 0)","MB","%u",mat->localPart->byteSize(mat->localPart)/(1024*1024));
		ghost_printLine("Remote matrix size (rank 0)","MB","%u",mat->remotePart->byteSize(mat->remotePart)/(1024*1024));
	}

	mat->printInfo(mat);
	ghost_printFooter();

	}

}

void ghost_printContextInfo(ghost_context_t *context)
{
	int nranks;
	int myrank;

   	if (context->communicator != NULL) {
		nranks = ghost_getNumberOfRanks(context->mpicomm);
		myrank = ghost_getRank(context->mpicomm); 
	} else {
		nranks = 1;
		myrank = 0;
	}

	if (myrank == 0) {
		char *contextType;
		if (context->flags & GHOST_CONTEXT_DISTRIBUTED)
			contextType = "Distributed";
		else if (context->flags & GHOST_CONTEXT_GLOBAL)
			contextType = "Global";


		ghost_printHeader("Context");
		ghost_printLine("MPI processes",NULL,"%d",nranks);
		ghost_printLine("Number of rows",NULL,"%"PRmatIDX,context->gnrows);
		ghost_printLine("Type",NULL,"%s",contextType);
		ghost_printLine("Work distribution scheme",NULL,"%s",ghost_workdistName(context->flags));
		ghost_printFooter();
	}

}

static char *env(char *key)
{
	int i=0;
	while (environ[i]) {
		if (!strncasecmp(key,environ[i],strlen(key)))
		{
			return environ[i]+strlen(key)+1;
		}
		i++;
	}
	return "undefined";

}

void ghost_printSysInfo()
{
	int nproc = ghost_getNumberOfRanks(MPI_COMM_WORLD);
	int nnodes = ghost_getNumberOfNodes();

#ifdef GHOST_HAVE_CUDA
	ghost_acc_info_t * CUdevInfo = CU_getDeviceInfo();
#endif
#ifdef GHOST_HAVE_OPENCL
	ghost_acc_info_t * CLdevInfo = CL_getDeviceInfo();
#endif
	if (ghost_getRank(MPI_COMM_WORLD) == 0) {

		int nthreads;
		int nphyscores = ghost_getNumberOfPhysicalCores();
		int ncores = ghost_getNumberOfHwThreads();

#ifdef GHOST_HAVE_OPENMP
		char omp_sched_str[32];
		omp_sched_t omp_sched;
		int omp_sched_mod;
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
#else
		char omp_sched_str[] = "N/A";
#endif
#pragma omp parallel
#pragma omp master
		nthreads = ghost_ompGetNumThreads();

		ghost_printHeader("System");
		ghost_printLine("Overall nodes",NULL,"%d",nnodes);
		ghost_printLine("Overall MPI processes",NULL,"%d",nproc);
		ghost_printLine("MPI processes per node",NULL,"%d",nproc/nnodes);
		ghost_printLine("Avail. threads (phys/HW) per node",NULL,"%d/%d",nphyscores,ncores);
		ghost_printLine("OpenMP threads per node",NULL,"%d",nproc/nnodes*nthreads);
		ghost_printLine("OpenMP threads per process",NULL,"%d",nthreads);
		ghost_printLine("OpenMP scheduling",NULL,"%s",omp_sched_str);
		ghost_printLine("KMP_BLOCKTIME",NULL,"%s",env("KMP_BLOCKTIME"));
#ifdef GHOST_HAVE_OPENCL
		ghost_printLine("OpenCL version",NULL,"%s",CL_getVersion());
		ghost_printLine("OpenCL devices",NULL,"%dx %s",CLdevInfo->nDevices[0],CLdevInfo->names[0]);
		int i;
		for (i=1; i<CLdevInfo->nDistinctDevices; i++) {
			ghost_printLine("",NULL,"%dx %s",CLdevInfo->nDevices[i],CLdevInfo->names[i]);
		}
#endif
#ifdef GHOST_HAVE_CUDA
		ghost_printLine("CUDA version",NULL,"%s",CU_getVersion());
		ghost_printLine("CUDA devices",NULL,NULL);
		int j;
		for (j=0; j<CUdevInfo->nDistinctDevices; j++) {
			if (strcasecmp(CUdevInfo->names[j],"None")) {
				ghost_printLine("",NULL,"%dx %s",CUdevInfo->nDevices[j],CUdevInfo->names[j]);
			}
		}
#endif
		ghost_printFooter();
	}
#ifdef GHOST_HAVE_OPENCL
	destroyCLdeviceInfo(CLdevInfo);
#endif

}

void ghost_printGhostInfo() 
{

	if (ghost_getRank(MPI_COMM_WORLD)==0) {
		/*	int nDataformats;
			char *availDataformats = NULL;
			char *avDF = NULL;
			size_t avDFlen = 0;
			ghost_getAvailableDataFormats(&availDataformats,&nDataformats);
			int i;
			for (i=0; i<nDataformats; i++) {
			char *curFormat = availDataformats+i*GHOST_DATAFORMAT_NAME_MAX;
			avDFlen += strlen(curFormat)+1;
			avDF = realloc(avDF,avDFlen);
			strncpy(avDF+avDFlen-strlen(curFormat)-1,curFormat,strlen(curFormat));
			strncpy(avDF+avDFlen-1,",",1);
			}
			avDF[avDFlen-1] = '\0'; // skip trailing comma */


		ghost_printHeader("%s", GHOST_NAME);
		ghost_printLine("Version",NULL,"%s",GHOST_VERSION);
		//	ghost_printLine("Available sparse matrix formats",NULL,"%s",avDF);
		ghost_printLine("Build date",NULL,"%s",__DATE__);
		ghost_printLine("Build time",NULL,"%s",__TIME__);
		//		ghost_printLine("Matrix data type",NULL,"%s",ghost_datatypeName(GHOST_MY_MDATATYPE));
		//		ghost_printLine("Vector data type",NULL,"%s",ghost_datatypeName(GHOST_MY_VDATATYPE));
#ifdef MIC
		ghost_printLine("MIC kernels",NULL,"enabled");
#else
		ghost_printLine("MIC kernels",NULL,"disabled");
#endif
#ifdef AVX
		ghost_printLine("AVX kernels",NULL,"enabled");
#else
		ghost_printLine("AVX kernels",NULL,"disabled");
#endif
#ifdef SSE
		ghost_printLine("SSE kernels",NULL,"enabled");
#else
		ghost_printLine("SSE kernels",NULL,"disabled");
#endif
#ifdef GHOST_HAVE_OPENMP
		ghost_printLine("OpenMP support",NULL,"enabled");
#else
		ghost_printLine("OpenMP support",NULL,"disabled");
#endif
#ifdef GHOST_HAVE_MPI
		ghost_printLine("MPI support",NULL,"enabled");
#else
		ghost_printLine("MPI support",NULL,"disabled");
#endif
#ifdef GHOST_HAVE_OPENCL
		ghost_printLine("OpenCL support",NULL,"enabled");
#else
		ghost_printLine("OpenCL support",NULL,"disabled");
#endif
#ifdef GHOST_HAVE_CUDA
		ghost_printLine("CUDA support",NULL,"enabled");
#else
		ghost_printLine("CUDA support",NULL,"disabled");
#endif
#ifdef LIKWID
		ghost_printLine("Likwid support",NULL,"enabled");
		printf("Likwid support                   :      enabled\n");
		/*#ifdef LIKWID_MARKER_FINE
		  printf("Likwid Marker API (high res)     :      enabled\n");
#else
#ifdef LIKWID_MARKER
printf("Likwid Marker API                :      enabled\n");
#endif
#endif*/
#else
		ghost_printLine("Likwid support",NULL,"disabled");
#endif
		ghost_printFooter();

		//	free(avDF);
		//	free(availDataformats);

	}


}

void ghost_referenceSolver(ghost_vec_t *nodeLHS, char *matrixPath, int datatype, ghost_vec_t *rhs, int nIter, int spmvmOptions)
{

	DEBUG_LOG(1,"Computing reference solution");
	int me;
   	if (nodeLHS->context->communicator != NULL) {
		me = ghost_getRank(nodeLHS->context->mpicomm); 
	} else {
		me = 0;
	}
	char *zero = (char *)ghost_malloc(ghost_sizeofDataType(datatype));
	memset(zero,0,ghost_sizeofDataType(datatype));
	//ghost_vec_t *res = ghost_createVector(distContext,GHOST_VEC_LHS|GHOST_VEC_HOST,NULL);
	ghost_vec_t *globLHS; 
	ghost_mtraits_t trait = {.format = GHOST_SPM_FORMAT_CRS, .flags = GHOST_SPM_HOST, .aux = NULL, .datatype = datatype};
	ghost_context_t *context;

	ghost_matfile_header_t fileheader;
	ghost_readMatFileHeader(matrixPath,&fileheader);

	context = ghost_createContext(fileheader.nrows,fileheader.ncols,GHOST_CONTEXT_GLOBAL,matrixPath,MPI_COMM_WORLD,1.0);
	ghost_mat_t *mat = ghost_createMatrix(context, &trait, 1);
	mat->fromFile(mat,matrixPath);
	ghost_vtraits_t rtraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_RHS|GHOST_VEC_HOST, .datatype = rhs->traits->datatype);
	ghost_vec_t *globRHS = ghost_createVector(context, &rtraits);
	globRHS->fromScalar(globRHS,zero);


	DEBUG_LOG(2,"Collection RHS vector for reference solver");
	rhs->collect(rhs,globRHS);

	if (me==0) {
		DEBUG_LOG(1,"Computing actual reference solution with one process");


		ghost_vtraits_t ltraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_LHS|GHOST_VEC_HOST, .datatype = rhs->traits->datatype);

		globLHS = ghost_createVector(context, &ltraits); 
		globLHS->fromScalar(globLHS,&zero);

		//CR_TYPE *cr = (CR_TYPE *)(context->fullMatrix->data);
		int iter;

		if (mat->symmetry == GHOST_BINCRS_SYMM_GENERAL) {
			for (iter=0; iter<nIter; iter++) {
				mat->kernel(mat,globLHS,globRHS,spmvmOptions);
			}
		} else if (mat->symmetry == GHOST_BINCRS_SYMM_SYMMETRIC) {
			WARNING_LOG("Computing the refernce solution for a symmetric matrix is not implemented!");
			for (iter=0; iter<nIter; iter++) {
			}
		}

		globRHS->destroy(globRHS);
		ghost_freeContext(context);
	} else {
		ghost_vtraits_t ltraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_LHS|GHOST_VEC_HOST|GHOST_VEC_DUMMY, .datatype = rhs->traits->datatype);
		globLHS = ghost_createVector(context, &ltraits);
	}
	DEBUG_LOG(1,"Scattering result of reference solution");

	nodeLHS->fromScalar(nodeLHS,&zero);
	globLHS->distribute(globLHS, nodeLHS);

	globLHS->destroy(globLHS);
	mat->destroy(mat);
	

	free(zero);
	DEBUG_LOG(1,"Reference solution has been computed and scattered successfully");
}
/*
// FIXME
void ghost_referenceKernel_symm(ghost_vdat_t *res, ghost_mnnz_t *col, ghost_midx_t *rpt, ghost_mdat_t *val, ghost_vdat_t *rhs, ghost_midx_t nrows, int spmvmOptions)
{
ghost_midx_t i, j;
ghost_vdat_t hlp1;

#pragma omp	parallel for schedule(runtime) private (hlp1, j)
for (i=0; i<nrows; i++){
hlp1 = 0.0;
for (j=rpt[i]; j<rpt[i+1]; j++){
hlp1 = hlp1 + (ghost_vdat_t)val[j] * rhs[col[j]];

if (i!=col[j]) {	
if (spmvmOptions & GHOST_SPMVM_AXPY) { 
#pragma omp atomic
res[col[j]] += (ghost_vdat_t)val[j] * rhs[i];
} else {
#pragma omp atomic
res[col[j]] += (ghost_vdat_t)val[i] * rhs[i];  // FIXME non-axpy case doesnt work
}
}

}
if (spmvmOptions & GHOST_SPMVM_AXPY) {
res[i] += hlp1;
} else {
res[i] = hlp1;
}
}
}

void ghost_referenceKernel(ghost_vdat_t *res, ghost_mnnz_t *col, ghost_midx_t *rpt, ghost_mdat_t *val, ghost_vdat_t *rhs, ghost_midx_t nrows, int spmvmOptions)
{
ghost_midx_t i, j;
ghost_vdat_t hlp1;

#pragma omp	parallel for schedule(runtime) private (hlp1, j)
for (i=0; i<nrows; i++){
hlp1 = 0.0;
for (j=rpt[i]; j<rpt[i+1]; j++){
hlp1 = hlp1 + (ghost_vdat_t)val[j] * rhs[col[j]]; // TODO do not multiply with zero if different datatypes 
}
if (spmvmOptions & GHOST_SPMVM_AXPY) 
res[i] += hlp1;
else
res[i] = hlp1;
}
}*/	

void ghost_freeCommunicator( ghost_comm_t* const comm ) 
{
	if(comm) {
		free(comm->lnEnts);
		free(comm->lnrows);
		free(comm->lfEnt);
		free(comm->lfRow);
		free(comm->wishes);
		free(comm->dues);
		free(comm->wishlist[0]);
		free(comm->duelist[0]);
		free(comm->wishlist);
		free(comm->duelist);
		free(comm->due_displ);
		free(comm->wish_displ);
		free(comm->hput_pos);
		free(comm);
	}
}

char * ghost_modeName(int spmvmOptions) 
{
	if (spmvmOptions & GHOST_SPMVM_MODE_NOMPI)
		return "non-MPI";
	if (spmvmOptions & GHOST_SPMVM_MODE_VECTORMODE)
		return "vector mode";
	if (spmvmOptions & GHOST_SPMVM_MODE_GOODFAITH)
		return "g/f hybrid";
	if (spmvmOptions & GHOST_SPMVM_MODE_TASKMODE)
		return "task mode";
	return "invalid";

}

int ghost_symmetryValid(int symmetry)
{
	if ((symmetry & GHOST_BINCRS_SYMM_GENERAL) &&
			(symmetry & ~GHOST_BINCRS_SYMM_GENERAL))
		return 0;

	if ((symmetry & GHOST_BINCRS_SYMM_SYMMETRIC) &&
			(symmetry & ~GHOST_BINCRS_SYMM_SYMMETRIC))
		return 0;

	return 1;
}

char * ghost_symmetryName(int symmetry)
{
	if (symmetry & GHOST_BINCRS_SYMM_GENERAL)
		return "General";

	if (symmetry & GHOST_BINCRS_SYMM_SYMMETRIC)
		return "Symmetric";

	if (symmetry & GHOST_BINCRS_SYMM_SKEW_SYMMETRIC) {
		if (symmetry & GHOST_BINCRS_SYMM_HERMITIAN)
			return "Skew-hermitian";
		else
			return "Skew-symmetric";
	} else {
		if (symmetry & GHOST_BINCRS_SYMM_HERMITIAN)
			return "Hermitian";
	}

	return "Invalid";
}

int ghost_datatypeValid(int datatype)
{
	if ((datatype & GHOST_BINCRS_DT_FLOAT) &&
			(datatype & GHOST_BINCRS_DT_DOUBLE))
		return 0;

	if (!(datatype & GHOST_BINCRS_DT_FLOAT) &&
			!(datatype & GHOST_BINCRS_DT_DOUBLE))
		return 0;

	if ((datatype & GHOST_BINCRS_DT_REAL) &&
			(datatype & GHOST_BINCRS_DT_COMPLEX))
		return 0;

	if (!(datatype & GHOST_BINCRS_DT_REAL) &&
			!(datatype & GHOST_BINCRS_DT_COMPLEX))
		return 0;

	return 1;
}

char * ghost_datatypeName(int datatype)
{
	if (datatype & GHOST_BINCRS_DT_FLOAT) {
		if (datatype & GHOST_BINCRS_DT_REAL)
			return "float";
		else
			return "complex float";
	} else {
		if (datatype & GHOST_BINCRS_DT_REAL)
			return "double";
		else
			return "complex double";
	}
}

char * ghost_workdistName(int options)
{
	if (options & GHOST_CONTEXT_WORKDIST_NZE)
		return "Equal no. of nonzeros";
	//else if (options & GHOST_CONTEXT_WORKDIST_LNZE)
	//	return "Equal no. of local nonzeros";
	else
		return "Equal no. of rows";
}

int ghost_getRank(MPI_Comm comm) 
{
#ifdef GHOST_HAVE_MPI
	int rank;
	MPI_safecall(MPI_Comm_rank ( comm, &rank ));
	return rank;
#else
	UNUSED(comm);
	return 0;
#endif
}

/*int ghost_getRank() 
{
#ifdef GHOST_HAVE_MPI
	int rank;
	MPI_safecall(MPI_Comm_rank ( MPI_COMM_WORLD, &rank ));
	return rank;
#else
	return 0;
#endif
}*/

int ghost_getLocalRank() 
{
#ifdef GHOST_HAVE_MPI
	return ghost_getRank(getSingleNodeComm());
#else
	return 0;
#endif
}

int ghost_getNumberOfRanksOnNode()
{
#ifdef GHOST_HAVE_MPI
	int size;
	MPI_safecall(MPI_Comm_size ( getSingleNodeComm(), &size));

	return size;
#else
	return 1;
#endif

}
int ghost_getNumberOfPhysicalCores()
{
	return hwloc_get_nbobjs_by_type(topology,HWLOC_OBJ_CORE);	
}

int ghost_getNumberOfHwThreads()
{
	return hwloc_get_nbobjs_by_type(topology,HWLOC_OBJ_PU);	
}

int ghost_getNumberOfThreads() 
{
	int nthreads;
#pragma omp parallel
	nthreads = ghost_ompGetNumThreads();

	return nthreads;
}

int ghost_getNumberOfNumaNodes()
{
	int depth = hwloc_get_type_depth(topology,HWLOC_OBJ_NODE);
	return hwloc_get_nbobjs_by_depth(topology,depth);
}

static int stringcmp(const void *x, const void *y)
{
	return (strcmp((char *)x, (char *)y));
}

int ghost_getNumberOfNodes() 
{
#ifndef GHOST_HAVE_MPI
	UNUSED(stringcmp);
	return 1;
#else

	int nameLen,me,size,i,distinctNames = 1;
	char name[MPI_MAX_PROCESSOR_NAME] = "";
	char *names = NULL;

	MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD,&me));
	MPI_safecall(MPI_Comm_size(MPI_COMM_WORLD,&size));
	MPI_safecall(MPI_Get_processor_name(name,&nameLen));


	if (me==0) {
		names = ghost_malloc(size*MPI_MAX_PROCESSOR_NAME*sizeof(char));
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

/*int ghost_getNumberOfProcesses() 
{
#ifndef GHOST_HAVE_MPI
	return 1;
#else

	int nnodes;

	MPI_safecall(MPI_Comm_size(MPI_COMM_WORLD, &nnodes));

	return nnodes;
#endif
}*/

int ghost_getNumberOfRanks(MPI_Comm comm)
{
#ifdef GHOST_HAVE_MPI
	int nnodes;
	MPI_safecall(MPI_Comm_size(comm, &nnodes));
	return nnodes;
#else
	UNUSED(comm);
	return 1;
#endif

}

size_t ghost_sizeofDataType(int dt)
{
	size_t size;

	if (dt & GHOST_BINCRS_DT_FLOAT)
		size = sizeof(float);
	else
		size = sizeof(double);

	if (dt & GHOST_BINCRS_DT_COMPLEX)
		size *= 2;

	return size;
}

int ghost_pad(int nrows, int padding) 
{
	int nrowsPadded;

	if(  nrows % padding != 0) {
		nrowsPadded = nrows + padding - nrows % padding;
	} else {
		nrowsPadded = nrows;
	}
	return nrowsPadded;
}
/*
void* allocateMemory( const size_t size, const char* desc ) 
{


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
}*/

void *ghost_malloc(const size_t size)
{
	void *mem = NULL;

	if (size/(1024.*1024.*1024.) > 1.) {
		DEBUG_LOG(1,"Allocating big array of size %f GB",size/(1024.*1024.*1024.));
	}

	mem = malloc(size);

	if( ! mem ) {
		ABORT("Error in memory allocation of %lu bytes: %s",size,strerror(errno));
	}
	return mem;
}

void *ghost_malloc_align(const size_t size, const size_t align)
{
	void *mem = NULL;
	int ierr;

	if ((ierr = posix_memalign((void**) &mem, align, size)) != 0) {
		ABORT("Error while allocating using posix_memalign: %s",strerror(ierr));
	}

	return mem;
}
/*
void freeMemory( size_t size, const char* desc, void* this_array ) 
{

	DEBUG_LOG(2,"Freeing %8.2f MB of memory for %s", size/(1024.*1024.), desc);

	allocatedMem -= size;
	free (this_array);

}*/

double ghost_bench_spmvm(ghost_context_t *context, ghost_vec_t *res, ghost_mat_t *mat, ghost_vec_t *invec, 
		int *spmvmOptions, int nIter)
{
	DEBUG_LOG(1,"Benchmarking the SpMVM");
	int it;
	double time = 0;
	double ttime = 0;
	double oldtime=1e9;
	//struct timespec end,start;

	ghost_solver_t solver = NULL;

	ghost_pickSpMVMMode(context,spmvmOptions);
	solver = context->solvers[ghost_getSpmvmModeIdx(*spmvmOptions)];

	if (!solver) {
		DEBUG_LOG(1,"The solver for the specified is not available, skipping");
		return -1.0;
	}

#ifdef GHOST_HAVE_MPI
	MPI_safecall(MPI_Barrier(context->mpicomm));
#endif
#ifdef GHOST_HAVE_OPENCL
	CL_barrier();
#endif

//	ttime = ghost_wctime();
	for( it = 0; it < nIter; it++ ) {
		//clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start);
		time = ghost_wctime();
		solver(context,res,mat,invec,*spmvmOptions);

#ifdef GHOST_HAVE_OPENCL
		CL_barrier();
#endif
#ifdef GHOST_HAVE_CUDA
		CU_barrier();
#endif
#ifdef GHOST_HAVE_MPI
		MPI_safecall(MPI_Barrier(context->mpicomm));
#endif
		//clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end);
		//time = ghost_timediff(start,end);
		time = ghost_wctime()-time;
	//	printf("%f\n",time);
//		if (time < 0)
//			printf("dummy\n");
		time = time<oldtime?time:oldtime;
		oldtime=time;
	}
	//DEBUG_LOG(0,"Total time: %f sec",ghost_wctime()-ttime);
	solver(NULL,NULL,NULL,NULL,0); // clean up

	DEBUG_LOG(1,"Downloading result from device");
	res->download(res);

	if ( *spmvmOptions & GHOST_SPMVM_MODES_COMBINED)  {
		res->permute(res,res->context->invRowPerm);
	} else if ( *spmvmOptions & GHOST_SPMVM_MODES_SPLIT ) {
		// one of those must return immediately
		res->permute(res,res->context->invRowPerm);
		res->permute(res,res->context->invRowPerm);
	}

	return time;
}

void ghost_pickSpMVMMode(ghost_context_t * context, int *spmvmOptions)
{
	if (!(*spmvmOptions & GHOST_SPMVM_MODES_ALL)) { // no mode specified
#ifdef GHOST_HAVE_MPI
		if (context->flags & GHOST_CONTEXT_GLOBAL)
			*spmvmOptions |= GHOST_SPMVM_MODE_NOMPI;
		else
			*spmvmOptions |= GHOST_SPMVM_MODE_GOODFAITH;
#else
		UNUSED(context);
		*spmvmOptions |= GHOST_SPMVM_MODE_NOMPI;
#endif
		DEBUG_LOG(1,"No spMVM mode has been specified, picking a sensible default, namely %s",ghost_modeName(*spmvmOptions));

	}

}

int ghost_getSpmvmModeIdx(int spmvmOptions)
{
	if (spmvmOptions & GHOST_SPMVM_MODE_NOMPI)
		return GHOST_SPMVM_MODE_NOMPI_IDX;
	if (spmvmOptions & GHOST_SPMVM_MODE_VECTORMODE)
		return GHOST_SPMVM_MODE_VECTORMODE_IDX;
	if (spmvmOptions & GHOST_SPMVM_MODE_GOODFAITH)
		return GHOST_SPMVM_MODE_GOODFAITH_IDX;
	if (spmvmOptions & GHOST_SPMVM_MODE_TASKMODE)
		return GHOST_SPMVM_MODE_TASKMODE_IDX;

	return 0;
}
int ghost_dataTypeIdx(int datatype)
{
	if (datatype & GHOST_BINCRS_DT_FLOAT) {
		if (datatype & GHOST_BINCRS_DT_COMPLEX)
			return GHOST_DT_C_IDX;
		else
			return GHOST_DT_S_IDX;
	} else {
		if (datatype & GHOST_BINCRS_DT_COMPLEX)
			return GHOST_DT_Z_IDX;
		else
			return GHOST_DT_D_IDX;
	}
}


void ghost_getAvailableDataFormats(char **dataformats, int *nDataformats)
{
	/*	char pluginPath[PATH_MAX];
		DIR * pluginDir = opendir(PLUGINPATH);
		struct dirent * dirEntry;
		ghost_spmf_plugin_t myPlugin;

	 *nDataformats=0;


	 if (pluginDir) {
	 while (0 != (dirEntry = readdir(pluginDir))) {
	 if (dirEntry->d_name[0] == 'd') 
	 { // only use double variant ==> only count each format once
	 snprintf(pluginPath,PATH_MAX,"%s/%s",PLUGINPATH,dirEntry->d_name);
	 myPlugin.so = dlopen(pluginPath,RTLD_LAZY);
	 if (!myPlugin.so) {
	 continue;
	 }

	 myPlugin.formatID = (char *)dlsym(myPlugin.so,"formatID");
	 if (!myPlugin.formatID) {
	 dlclose(myPlugin.so);
	 continue;
	 }

	 (*nDataformats)++;
	 *dataformats = realloc(*dataformats,(*nDataformats)*GHOST_DATAFORMAT_NAME_MAX);
	 strncpy((*dataformats)+((*nDataformats)-1)*GHOST_DATAFORMAT_NAME_MAX,myPlugin.formatID,GHOST_DATAFORMAT_NAME_MAX);
	 dlclose(myPlugin.so);
	 }
	 }
	 closedir(pluginDir);
	 } else {
	 ABORT("The plugin directory does not exist");
	 }
	 */
	UNUSED(dataformats);
	UNUSED(nDataformats);
}

int ghost_archIsBigEndian()
{
	int test = 1;
	unsigned char *endiantest = (unsigned char *)&test;

	return (endiantest[0] == 0);
}

int ghost_getCoreNumbering()
{
	int i,j;
	int fd;
	char cpuFile[1024];
	char siblings[32];
	char sblPhysicalFirst[32];

	int physFirst = 0;
	int smtFirst = 0;
	char sblSmtFirst[32];

	int nSMT = ghost_getNumberOfHwThreads()/ghost_getNumberOfPhysicalCores();

	for (i=0; i<ghost_getNumberOfHwThreads(); i++)
	{
		sprintf(cpuFile,"/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list",i);
		fd = open(cpuFile,O_RDONLY);
		read(fd,siblings,31);
		close(fd);

		strtok(siblings,"\n");

		sblPhysicalFirst[0] = '\0';
		for (j=0; j<nSMT; j++) {
			sprintf(sblPhysicalFirst+strlen(sblPhysicalFirst),"%d,",
					i+ghost_getNumberOfPhysicalCores()*(j-i/ghost_getNumberOfPhysicalCores()));
		}
		sblPhysicalFirst[strlen(sblPhysicalFirst)-1] = '\0';

		sprintf(sblSmtFirst,"%d-%d",i-i%nSMT,i+(nSMT-i%nSMT)-1);

		if (!strcmp(siblings,sblPhysicalFirst))
			physFirst++;
		else if (!strcmp(siblings,sblSmtFirst))
			smtFirst++;


	}

	if (physFirst == ghost_getNumberOfHwThreads())
		return GHOST_CORENUMBERING_PHYSICAL_FIRST;
	else if (smtFirst == ghost_getNumberOfHwThreads())
		return GHOST_CORENUMBERING_SMT_FIRST;
	else
		return GHOST_CORENUMBERING_INVALID;

}

int ghost_getCore()
{
	cpu_set_t  cpu_set;
	CPU_ZERO(&cpu_set);
	sched_getaffinity(syscall(SYS_gettid),sizeof(cpu_set_t), &cpu_set);
	int processorId = -1;

	for (processorId=0;processorId<GHOST_MAX_THREADS;processorId++)
	{
		if (CPU_ISSET(processorId,&cpu_set))
		{
			break;
		}
	}
	return processorId;
}

char ghost_datatypePrefix(int dt)
{
	char p;

	if (dt & GHOST_BINCRS_DT_FLOAT) {
		if (dt & GHOST_BINCRS_DT_COMPLEX)
			p = 'c';
		else
			p = 's';
	} else {
		if (dt & GHOST_BINCRS_DT_COMPLEX)
			p = 'z';
		else
			p = 'd';
	}

	return p;
}


ghost_midx_t ghost_globalIndex(ghost_context_t *ctx, ghost_midx_t lidx)
{
	if (ctx->flags & GHOST_CONTEXT_DISTRIBUTED)
		return ctx->communicator->lfRow[ghost_getRank(ctx->mpicomm)] + lidx;

	return lidx;	
}

void ghost_readMatFileHeader(char *matrixPath, ghost_matfile_header_t *header)
{
	FILE* file;
	long filesize;
	int swapReq = 0;

	DEBUG_LOG(1,"Reading header from %s",matrixPath);

	if ((file = fopen(matrixPath, "rb"))==NULL){
		ABORT("Could not open binary CRS file %s",matrixPath);
	}

	fseek(file,0L,SEEK_END);
	filesize = ftell(file);
	fseek(file,0L,SEEK_SET);

	fread(&header->endianess, 4, 1, file);
	if (header->endianess == GHOST_BINCRS_LITTLE_ENDIAN && ghost_archIsBigEndian()) {
		DEBUG_LOG(1,"Need to convert from little to big endian.");
		swapReq = 1;
	} else if (header->endianess != GHOST_BINCRS_LITTLE_ENDIAN && !ghost_archIsBigEndian()) {
		DEBUG_LOG(1,"Need to convert from big to little endian.");
		swapReq = 1;
	} else {
		DEBUG_LOG(1,"OK, file and library have same endianess.");
	}

	fread(&header->version, 4, 1, file);
	if (swapReq) header->version = bswap_32(header->version);

	fread(&header->base, 4, 1, file);
	if (swapReq) header->base = bswap_32(header->base);

	fread(&header->symmetry, 4, 1, file);
	if (swapReq) header->symmetry = bswap_32(header->symmetry);

	fread(&header->datatype, 4, 1, file);
	if (swapReq) header->datatype = bswap_32(header->datatype);

	fread(&header->nrows, 8, 1, file);
	if (swapReq) header->nrows  = bswap_64(header->nrows);

	fread(&header->ncols, 8, 1, file);
	if (swapReq)  header->ncols  = bswap_64(header->ncols);

	fread(&header->nnz, 8, 1, file);
	if (swapReq)  header->nnz  = bswap_64(header->nnz);


	long rightFilesize = GHOST_BINCRS_SIZE_HEADER +
		(long)(header->nrows+1) * GHOST_BINCRS_SIZE_RPT_EL +
		(long)header->nnz * GHOST_BINCRS_SIZE_COL_EL +
		(long)header->nnz * ghost_sizeofDataType(header->datatype);

	if (filesize != rightFilesize)
		ABORT("File has invalid size! (is: %ld, should be: %ld)",filesize, rightFilesize);

	fclose(file);
}

inline void ghost_setCore(int coreNumber)
{

	DEBUG_LOG(2,"Pinning thread %d to core %d",ghost_ompGetThreadNum(),coreNumber);
	cpu_set_t cpu_set;
	CPU_ZERO(&cpu_set);
	CPU_SET(coreNumber, &cpu_set);

	int error = sched_setaffinity((pid_t)0, sizeof(cpu_set_t), &cpu_set);
	if (error != 0) {
		WARNING_LOG("Pinning thread to core %d failed (%d): %s", 
				coreNumber, error, strerror(error));
	}

}

inline void ghost_unsetCore()
{
	DEBUG_LOG(2,"Unpinning thread %d from core %d",ghost_ompGetThreadNum(),ghost_getCore());
	cpu_set_t cpu_set;
	CPU_ZERO(&cpu_set);
	int i;
	for (i=0; i<ghost_getNumberOfHwThreads(); i++) {
		CPU_SET(i,&cpu_set);
	}

	int error = sched_setaffinity((pid_t)0, sizeof(cpu_set_t), &cpu_set);
	if (error != 0) {
		WARNING_LOG("Unpinning thread from core %d failed (%d): %s", 
				ghost_getCore(), error, strerror(error));
	}

}

void ghost_pinThreads(int options, char *procList)
{
	if (procList != NULL) {
		char *list = strdup(procList);
		DEBUG_LOG(1,"Setting number of threads and pinning them to cores %s",list);

		const char delim[] = ",";
		char *coreStr;
		int *cores = NULL;
		int nCores = 0;

		coreStr = strtok(list,delim);
		while(coreStr != NULL) 
		{
			nCores++;
			cores = (int *)realloc(cores,nCores*sizeof(int));
			cores[nCores-1] = atoi(coreStr);
			coreStr = strtok(NULL,delim);
		}

		DEBUG_LOG(1,"Adjusting number of threads to %d",nCores);
		ghost_ompSetNumThreads(nCores);

#pragma omp parallel
		ghost_setCore(cores[ghost_ompGetThreadNum()]);

		free(cores);


	} else {
		DEBUG_LOG(1,"Trying to automatically pin threads");
		int numbering = ghost_getCoreNumbering();
		if (numbering == GHOST_CORENUMBERING_PHYSICAL_FIRST) {
			DEBUG_LOG(1,"The core numbering seems to be 'physical cores first'");
		} else {
			DEBUG_LOG(1,"The core numbering seems to be 'SMT threads first'");
		}

		int nCores;
		int nPhysCores = ghost_getNumberOfPhysicalCores();
		if (options & GHOST_PIN_PHYS)
			nCores = nPhysCores;
		else
			nCores = ghost_getNumberOfHwThreads();

		int offset = nPhysCores/ghost_getNumberOfRanksOnNode();
		int SMT = ghost_getNumberOfHwThreads()/ghost_getNumberOfPhysicalCores();
		ghost_ompSetNumThreads(nCores/ghost_getNumberOfRanksOnNode());

#pragma omp parallel
		{
			int coreNumber;

			if (options & GHOST_PIN_SMT) {
				coreNumber = ghost_ompGetThreadNum()/SMT+(offset*(ghost_getLocalRank()))+(ghost_ompGetThreadNum()%SMT)*nPhysCores;
			} else {
				if (numbering == GHOST_CORENUMBERING_PHYSICAL_FIRST)
					coreNumber = ghost_ompGetThreadNum()+(offset*(ghost_getLocalRank()));
				else
					coreNumber = ghost_ompGetThreadNum()*SMT+(offset*(ghost_getLocalRank()));
			}

			ghost_setCore(coreNumber);


		}

	}
#pragma omp parallel
	{
		DEBUG_LOG(1,"Thread %d is running on core %d",ghost_ompGetThreadNum(),ghost_getCore());

	}


}

ghost_mnnz_t ghost_getMatNrows(ghost_mat_t *mat)
{
	ghost_mnnz_t nrows;
	ghost_mnnz_t lnrows = mat->nrows(mat);

	if (mat->context->flags & GHOST_CONTEXT_GLOBAL) {
		nrows = lnrows;
	} else {
#ifdef GHOST_HAVE_MPI
		MPI_safecall(MPI_Allreduce(&lnrows,&nrows,1,ghost_mpi_dt_midx,MPI_SUM,mat->context->mpicomm));
#else
		ABORT("Trying to get the number of matrix rows in a distributed context without MPI");
#endif
	}

	return nrows;
}

ghost_mnnz_t ghost_getMatNnz(ghost_mat_t *mat)
{
	ghost_mnnz_t nnz;
	ghost_mnnz_t lnnz = mat->nnz(mat);

	if (mat->context->flags & GHOST_CONTEXT_GLOBAL) {
		nnz = lnnz;
	} else {
#ifdef GHOST_HAVE_MPI
		MPI_safecall(MPI_Allreduce(&lnnz,&nnz,1,ghost_mpi_dt_mnnz,MPI_SUM,mat->context->mpicomm));
#else
		ABORT("Trying to get the number of matrix nonzeros in a distributed context without MPI");
#endif
	}

	return nnz;
}

#if 0
static inline void *ghost_enterTask(void *arg)
{
	ghost_task_t *task = (ghost_task_t *)arg;

	omp_set_num_threads(task->nThreads);

#pragma omp parallel
	{
		threadpool[task->coreList[omp_get_thread_num()]].state = GHOST_THREAD_RUNNING;
		ghost_setCore(task->coreList[omp_get_thread_num()]);	
	}

	return task->func(task->arg);

}

inline void ghost_spawnTask(ghost_task_t *task) //void *(*func) (void *), void *arg, int nThreads, void *affinity, char *desc, int flags)
{

	DEBUG_LOG(2,"There are %d threads available",ghost_getNumberOfThreads());
	DEBUG_LOG(1,"Starting %s task %s which requires %d threads",task->flags&GHOST_TASK_ASYNC?"asynchronous":"synchronous",task->desc,task->nThreads);

	// TODO: if sync: use core 0 as well, if async: skip core 0

	int i;

	if (task->coreList == NULL) {
		DEBUG_LOG(1,"Auto-selecting cores for this task");
		task->coreList = (int *)ghost_malloc(sizeof(int)*task->nThreads);

		int c = 0;

		for (i=0; i<ghost_getNumberOfThreads() && c<task->nThreads; i++) {
			if ((task->flags & GHOST_TASK_EXCLUSIVE) && (threadpool[i].state == GHOST_THREAD_RUNNING)) {
				DEBUG_LOG(2,"Skipping core %d %d",i,threadpool[i].state);
				continue;
			}

			DEBUG_LOG(2,"Thread %d running on core %d",c,i);
			task->coreList[c++] = i;
		}
	}


	DEBUG_LOG(2,"Calling pthread_create");	

	pthread_create(&(task->tid),NULL,&ghost_enterTask,task);


	nTasks++;
	if (nTasks > GHOST_MAX_NTASKS) ABORT("Maximum number of tasks reached");
	tasklist[nTasks-1] = task;
	/*tasklist[nTasks-1].flags = flags;
	  tasklist[nTasks-1].tid = tid;
	  tasklist[nTasks-1].nThreads = args->nThreads;
	  tasklist[nTasks-1].coreList = args->coreList;

	 */

	if (task->flags & GHOST_TASK_SYNC)
		ghost_waitTask(tasklist[nTasks-1]);

	//omp_set_num_threads(ghost_getNumberOfThreads()-nThreads);

	// register task

	//return tasklist[nTasks-1];

}

void ghost_waitTask(ghost_task_t *task)
{
	int i;
	pthread_join(task->tid,NULL);

	for (i=0; i<task->nThreads; i++)
		threadpool[task->coreList[i]].state = GHOST_THREAD_HALTED;

	//omp_set_num_threads(ghost_getNumberOfThreads()+task->nThreads);
}
#endif

int ghost_flopsPerSpmvm(int m_t, int v_t)
{
	int flops = 2;

	if (m_t & GHOST_BINCRS_DT_COMPLEX) {
		if (v_t & GHOST_BINCRS_DT_COMPLEX) {
			flops = 8;
		}
	} else {
		if (v_t & GHOST_BINCRS_DT_COMPLEX) {
			flops = 4;
		}
	}

	return flops;
}

ghost_vtraits_t * ghost_cloneVtraits(ghost_vtraits_t *t1)
{
	ghost_vtraits_t *t2 = (ghost_vtraits_t *)ghost_malloc(sizeof(ghost_vtraits_t));
	memcpy(t2,t1,sizeof(ghost_vtraits_t));

	return t2;
}

void ghost_ompSetNumThreads(int nthreads)
{
#ifdef GHOST_HAVE_OPENMP
	omp_set_num_threads(nthreads);
#endif
}
	
int ghost_ompGetNumThreads()
{
#ifdef GHOST_HAVE_OPENMP
	return omp_get_num_threads();
#else 
	return 1;
#endif
}

int ghost_ompGetThreadNum()
{
#ifdef GHOST_HAVE_OPENMP
	return omp_get_thread_num();
#else
	return 0;
#endif
}
