#ifndef _SPMVM_UTIL_H_
#define _SPMVM_UTIL_H_

#include "ghost.h"
#ifdef MPI
#include <mpi.h>
#endif

/******************************************************************************/
/****** Makros ****************************************************************/
/******************************************************************************/
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
#define ABORT(msg, ...) {\
	int __me;\
	MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD,&__me));\
	fprintf(stderr,"PE%d ABORTING at %s:%d: ",__me,__FILE__,__LINE__);\
	fprintf(stderr,msg, ##__VA_ARGS__);\
	fprintf(stderr, "\n");\
	fflush(stderr);\
}
#else
#define ABORT(msg, ...) {\
	fprintf(stderr,"ABORTING at %s:%d: ",__FILE__,__LINE__);\
	fprintf(stderr,msg, ##__VA_ARGS__);\
	fprintf(stderr, "\n");\
	fflush(stderr);\
}
#endif

#define MPI_safecall(call) {\
  int mpierr = call ;\
  if( MPI_SUCCESS != mpierr ){\
    fprintf(stderr, "MPI error at %s:%d, %d\n",\
      __FILE__, __LINE__, mpierr);\
    fflush(stderr);\
  }\
  }
#define CL_safecall(call) {\
  cl_int clerr = call ;\
  if( CL_SUCCESS != clerr ){\
    fprintf(stderr, "OpenCL error at %s:%d, %s\n",\
      __FILE__, __LINE__, CL_errorString(clerr));\
    fflush(stderr);\
  }\
  }

#define CL_checkerror(clerr) do{\
  if( CL_SUCCESS != clerr ){\
    fprintf(stdout, "OpenCL error at %s:%d, %s\n",\
      __FILE__, __LINE__, CL_errorString(clerr));\
    fflush(stdout);\
  }\
  } while(0)

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
void CL_finish(int);

void CL_SpMVM(cl_mem rhsVec, cl_mem resVec, int type); 
void CL_vecscal(cl_mem a, ghost_mdat_t s, int nrows);
void CL_axpy(cl_mem a, cl_mem b, ghost_mdat_t s, int nrows);
void CL_dotprod(cl_mem a, cl_mem b, ghost_mdat_t *out, int nrows);
//void CL_setup_communication(ghost_comm_t* lcrp, GHOST_SPM_GPUFORMATS *matrixFormats, int);
//GPUghost_mat_t * CL_createMatrix(ghost_comm_t* lcrp, GHOST_SPM_GPUFORMATS *matrixFormats, int spmvmOptions);
void CL_enqueueKernel(cl_kernel kernel, cl_uint dim, size_t *gSize, size_t *lSize);
const char * CL_errorString(cl_int err);
 
size_t CL_getLocalSize(cl_kernel kernel);
CL_DEVICE_INFO * CL_getDeviceInfo();
void destroyCLdeviceInfo(CL_DEVICE_INFO * di);
void CL_barrier();
#endif


void ghost_printHeader(const char *fmt, ...);
void ghost_printFooter(); 
void ghost_printLine(const char *label, const char *unit, const char *format, ...);
void ghost_printSetupInfo(ghost_setup_t *setup, int options);
void              ghost_printEnvInfo();
ghost_vec_t *ghost_referenceSolver(char *matrixPath, ghost_setup_t *distSetup,  ghost_mdat_t (*fp)(int), int nIter, int spmvmOptions);
void ghost_referenceKernel(ghost_mdat_t *res, mat_nnz_t *col, mat_idx_t *rpt, ghost_mdat_t *val, ghost_mdat_t *rhs, mat_idx_t nrows, int spmvmOptions);
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
unsigned int ghost_getNumberOfNodes();
unsigned int ghost_getNumberOfProcesses();
#endif
