#ifndef _SPMVM_UTIL_H_
#define _SPMVM_UTIL_H_

#include "ghost.h"
#ifdef MPI
#include <mpi.h>
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

#ifdef MPI
#define DEBUG_LOG(level,msg, ...) {\
	if(DEBUG >= level) {\
		int __me;\
		MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD,&__me));\
		fprintf(stderr,"PE%d at %s:%d: ",__me,__FILE__,__LINE__);\
		fprintf(stderr,msg, ##__VA_ARGS__);\
		fprintf(stderr, "\n");\
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
	fprintf(stderr,ANSI_COLOR_YELLOW "PE%d WARNING at %s:%d: " ,__me,__FILE__,__LINE__);\
	fprintf(stderr,msg, ##__VA_ARGS__);\
	fprintf(stderr, ANSI_COLOR_RESET"\n");\
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
	fprintf(stderr,ANSI_COLOR_MAGENTA "PE%d ABORTING at %s:%d: " ,__me,__FILE__,__LINE__);\
	fprintf(stderr,msg, ##__VA_ARGS__);\
	fprintf(stderr, ANSI_COLOR_RESET"\n");\
	fflush(stderr);\
}
#else
#define ABORT(msg, ...) {\
	fprintf(stderr,ANSI_COLOR_MAGENTA "ABORTING at %s:%d: ",__FILE__,__LINE__);\
	fprintf(stderr,msg, ##__VA_ARGS__);\
	fprintf(stderr, ANSI_COLOR_RESET"\n");\
	fflush(stderr);\
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

#define UNUSED(x) (void)(x)
/******************************************************************************/

#ifdef OPENCL


void CL_init();
cl_program CL_registerProgram(const char *filename, const char *options);
void CL_bindMatrixToKernel(void *mat, int format, int T, int kernelIdx, int spmvmOptions);

//CL_uploadCRS (MATRIY_TYPE *lcrp, GHOST_SPM_GPUFORMATS *matrixFormats, int spmvmOptions);
void CL_uploadVector( ghost_vec_t *vec );
void CL_downloadVector( ghost_vec_t *vec );

cl_mem CL_allocDeviceMemory( size_t );
cl_mem CL_allocDeviceMemoryMapped( size_t bytesize, void *hostPtr, int flag );
cl_mem CL_allocDeviceMemoryCached( size_t bytesize, void *hostPtr );
void * CL_mapBuffer(cl_mem devmem, size_t bytesize);
void CL_copyDeviceToHost( void*, cl_mem, size_t );
cl_event CL_copyDeviceToHostNonBlocking( void* hostmem, cl_mem devmem,
	   	size_t bytesize );
void CL_copyHostToDevice( cl_mem, void*, size_t );
void CL_copyHostToDeviceOffset( cl_mem, void*, size_t, size_t);
void CL_freeDeviceMemory( cl_mem );
void freeHostMemory( void* );
void CL_finish();

void CL_SpMVM(cl_mem rhsVec, cl_mem resVec, int type); 
void CL_vecscal(cl_mem a, ghost_mdat_t s, int nrows);
void CL_axpy(cl_mem a, cl_mem b, ghost_mdat_t s, int nrows);
void CL_dotprod(cl_mem a, cl_mem b, ghost_mdat_t *out, int nrows);
//void CL_context_communication(ghost_comm_t* lcrp, GHOST_SPM_GPUFORMATS *matrixFormats, int);
//GPUghost_mat_t * CL_createMatrix(ghost_comm_t* lcrp, GHOST_SPM_GPUFORMATS *matrixFormats, int spmvmOptions);
void CL_enqueueKernel(cl_kernel kernel, cl_uint dim, size_t *gSize, size_t *lSize);
const char * CL_errorString(cl_int err);
 
size_t CL_getLocalSize(cl_kernel kernel);
CL_DEVICE_INFO * CL_getDeviceInfo();
void destroyCLdeviceInfo(CL_DEVICE_INFO * di);
void CL_barrier();
const char * CL_getVersion();
#endif


void ghost_printHeader(const char *fmt, ...);
void ghost_printFooter(); 
void ghost_printLine(const char *label, const char *unit, const char *format, ...);
void ghost_printContextInfo(ghost_context_t *context);
void ghost_printOptionsInfo(int options);
void ghost_printSysInfo();
void ghost_printGhostInfo();
ghost_vec_t *ghost_referenceSolver(char *matrixPath, ghost_context_t *distContext,  ghost_vdat_t (*fp)(int), int nIter, int spmvmOptions);
void ghost_referenceKernel(ghost_vdat_t *res, ghost_mnnz_t *col, ghost_midx_t *rpt, ghost_mdat_t *val, ghost_vdat_t *rhs, ghost_midx_t nrows, int spmvmOptions);
void ghost_referenceKernel_symm(ghost_vdat_t *res, ghost_mnnz_t *col, ghost_midx_t *rpt, ghost_mdat_t *val, ghost_vdat_t *rhs, ghost_midx_t nrows, int spmvmOptions);
char * ghost_workdistName(int options);
char * ghost_modeName(int mode);
char * ghost_datatypeName(int datatype);

void* allocateMemory( const size_t size, const char* desc );
void freeMemory(size_t, const char*, void*);

void ghost_freeCommunicator( ghost_comm_t* const );
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
char * ghost_symmetryName(int symmetry);
#endif
