#ifndef __GHOST_UTIL_H__
#define __GHOST_UTIL_H__

#include "ghost.h"

#ifdef MPI
#include <mpi.h>
#include "mpi/ghost_mpi_util.h"
#endif

#ifdef OPENCL
#include "ghost_cl_util.h"
#endif

#ifdef CUDA
#include "ghost_cu_util.h"
#endif
/******************************************************************************/
/****** Makros ****************************************************************/
/******************************************************************************/
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define IF_DEBUG(level) if( DEBUG >= level )
//#define DEBUG_IDT 0
//extern int DEBUG_IDT;

//#define DEBUG_INDENT DEBUG_IDT+=2
//#define DEBUG_OUTDENT DEBUG_IDT-=2

#ifdef MPI
#define DEBUG_LOG(level,msg, ...) {\
	if(DEBUG >= level) {\
		int __me;\
		MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD,&__me));\
		fprintf(stderr,"PE%d at %s:%d: "msg"\n",__me,__FILE__,__LINE__,##__VA_ARGS__);\
		fflush(stderr);\
	}\
}
#else
#define DEBUG_LOG(level,msg, ...) {\
	if(DEBUG >= level) {\
		fprintf(stderr,"%s:%d: ",__FILE__,__LINE__);\
		fprintf(stderr, msg, ##__VA_ARGS__);\
		fprintf(stderr, "\n");\
		fflush(stderr);\
	}\
}
#endif

#ifdef MPI
#define WARNING_LOG(msg, ...) {\
	int __me;\
	MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD,&__me));\
	fprintf(stderr,ANSI_COLOR_YELLOW "PE%d WARNING at %s:%d: "msg"\n"ANSI_COLOR_RESET ,__me,__FILE__,__LINE__,##__VA_ARGS__);\
	fflush(stderr);\
}
#else
#define WARNING_LOG(msg, ...) {\
	fprintf(stderr,ANSI_COLOR_YELLOW "WARNING at %s:%d: " ,__FILE__,__LINE__);\
	fprintf(stderr, msg, ##__VA_ARGS__);\
	fprintf(stderr, ANSI_COLOR_RESET"\n");\
	fflush(stderr);\
}
#endif

#ifdef MPI
#define ABORT(msg, ...) {\
	int __me;\
	MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD,&__me));\
	fprintf(stderr,ANSI_COLOR_MAGENTA "PE%d ABORTING at %s:%d: "msg"\n"ANSI_COLOR_RESET ,__me,__FILE__,__LINE__,##__VA_ARGS__);\
	fflush(stderr);\
	MPI_safecall(MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE));\
}
#else
#define ABORT(msg, ...) {\
	fprintf(stderr,ANSI_COLOR_MAGENTA "ABORTING at %s:%d: ",__FILE__,__LINE__);\
	fprintf(stderr,msg, ##__VA_ARGS__);\
	fprintf(stderr, ANSI_COLOR_RESET"\n");\
	fflush(stderr);\
	exit(EXIT_FAILURE);\
}
#endif

#define MPI_safecall(call) {\
	int mpierr = call ;\
	if( MPI_SUCCESS != mpierr ){\
		fprintf(stderr, ANSI_COLOR_RED "MPI error at %s:%d, %d\n" ANSI_COLOR_RESET,\
				__FILE__, __LINE__, mpierr);\
		fflush(stderr);\
	}\
}

#ifdef MPI
#define CL_safecall(call) {\
	cl_int clerr = call ;\
	if( CL_SUCCESS != clerr ){\
		int __me;\
		MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD,&__me));\
		fprintf(stderr, ANSI_COLOR_RED "PE%d: OpenCL error at %s:%d, %s\n" ANSI_COLOR_RESET,\
				__me, __FILE__, __LINE__, CL_errorString(clerr));\
		fflush(stderr);\
	}\
}

#define CL_checkerror(clerr) do{\
	if( CL_SUCCESS != clerr ){\
		int __me;\
		MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD,&__me));\
		fprintf(stdout, ANSI_COLOR_RED "PE%d: OpenCL error at %s:%d, %s\n" ANSI_COLOR_RESET,\
				__me, __FILE__, __LINE__, CL_errorString(clerr));\
		fflush(stdout);\
	}\
} while(0)
#else
#define CL_safecall(call) {\
	cl_int clerr = call ;\
	if( CL_SUCCESS != clerr ){\
		fprintf(stderr, ANSI_COLOR_RED "OpenCL error at %s:%d, %s\n" ANSI_COLOR_RESET,\
				__FILE__, __LINE__, CL_errorString(clerr));\
		fflush(stderr);\
	}\
}

#define CL_checkerror(clerr) do{\
	if( CL_SUCCESS != clerr ){\
		fprintf(stdout, ANSI_COLOR_RED "OpenCL error at %s:%d, %s\n" ANSI_COLOR_RESET,\
				__FILE__, __LINE__, CL_errorString(clerr));\
		fflush(stdout);\
	}\
} while(0)
#endif

#ifdef MPI
#define CU_safecall(call) {\
	cudaError_t __cuerr = call ;\
	if( cudaSuccess != __cuerr ){\
		int __me;\
		MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD,&__me));\
		fprintf(stdout, ANSI_COLOR_RED "PE%d: CUDA error at %s:%d, %s\n" ANSI_COLOR_RESET,\
				__me, __FILE__, __LINE__, cudaGetErrorString(__cuerr));\
		fflush(stdout);\
	}\
}

#define CU_checkerror() {\
	cudaError_t __cuerr = cudaGetLastError();\
	if( cudaSuccess != __cuerr ){\
		int __me;\
		MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD,&__me));\
		fprintf(stdout, ANSI_COLOR_RED "PE%d: CUDA error at %s:%d, %s\n" ANSI_COLOR_RESET,\
				__me, __FILE__, __LINE__, cudaGetErrorString(__cuerr));\
		fflush(stdout);\
	}\
}

#else

#define CU_safecall(call) {\
	cudaError_t __cuerr = call ;\
	if( cudaSuccess != __cuerr ){\
		fprintf(stdout, ANSI_COLOR_RED "CUDA error at %s:%d, %s\n" ANSI_COLOR_RESET,\
				__FILE__, __LINE__, cudaGetErrorString(__cuerr));\
		fflush(stdout);\
	}\
}

#define CU_checkerror() {\
	cudaError_t __cuerr = cudaGetLastError();\
	if( cudaSuccess != __cuerr ){\
		fprintf(stdout, ANSI_COLOR_RED "CUDA error at %s:%d, %s\n" ANSI_COLOR_RESET,\
				__FILE__, __LINE__, cudaGetErrorString(__cuerr));\
		fflush(stdout);\
	}\
}
#endif

#define UNUSED(x) (void)(x)
/******************************************************************************/

#ifdef MPI
MPI_Datatype GHOST_MPI_DT_C;
MPI_Op GHOST_MPI_OP_SUM_C;
MPI_Datatype GHOST_MPI_DT_Z;
MPI_Op GHOST_MPI_OP_SUM_Z;
#endif

void ghost_printHeader(const char *fmt, ...);
void ghost_printFooter(); 
void ghost_printLine(const char *label, const char *unit, const char *format, ...);
void ghost_printContextInfo(ghost_context_t *context);
void ghost_printOptionsInfo(int options);
void ghost_printSysInfo();
void ghost_printGhostInfo();


void ghost_solver_nompi(ghost_context_t *context, ghost_vec_t* res, ghost_mat_t* mat, ghost_vec_t* invec, int spmvmOptions);
ghost_vec_t *ghost_referenceSolver(char *matrixPath, int datatype, ghost_context_t *distContext, ghost_vec_t *rhs, int nIter, int spmvmOptions);
//void ghost_referenceKernel(ghost_vdat_t *res, ghost_mnnz_t *col, ghost_midx_t *rpt, ghost_mdat_t *val, ghost_vdat_t *rhs, ghost_midx_t nrows, int spmvmOptions);
//void ghost_referenceKernel_symm(ghost_vdat_t *res, ghost_mnnz_t *col, ghost_midx_t *rpt, ghost_mdat_t *val, ghost_vdat_t *rhs, ghost_midx_t nrows, int spmvmOptions);

char * ghost_workdistName(int ghostOptions);
char * ghost_modeName(int spmvmOptions);
char * ghost_datatypeName(int datatype);
char * ghost_symmetryName(int symmetry);

int ghost_pad(int nrows, int padding);

void* allocateMemory( const size_t size, const char* desc );
void freeMemory(size_t, const char*, void*);

void ghost_freeCommunicator( ghost_comm_t* const );
void ghost_freeSpmfPlugin(ghost_spmf_plugin_t *plugin);
int ghost_getNumberOfPhysicalCores();
int ghost_getRank();
int ghost_getLocalRank();
int ghost_getNumberOfRanksOnNode();
int ghost_getNumberOfHwThreads();
int ghost_getNumberOfThreads();
int ghost_getNumberOfNodes();
int ghost_getNumberOfProcesses();
size_t ghost_sizeofDataType(int dt);
int ghost_datatypeValid(int datatype);
int ghost_symmetryValid(int symmetry);
int ghost_archIsBigEndian();
int ghost_getCoreNumbering();
int ghost_getCore();
void ghost_pickSpMVMMode(ghost_context_t * context, int *spmvmOptions);
char ghost_datatypePrefix(int dt);
int ghost_dataTypeIdx(int datatype);
ghost_midx_t ghost_globalIndex(ghost_context_t *, ghost_midx_t);

int ghost_getSpmvmModeIdx(int spmvmOptions);
void ghost_getAvailableDataFormats(char **dataformats, int *nDataformats);

double ghost_bench_spmvm(ghost_context_t *context, ghost_vec_t *res, ghost_mat_t *mat, ghost_vec_t *invec, int *spmvmOptions, int nIter);
void ghost_readMatFileHeader(char *, ghost_matfile_header_t *);
#endif
