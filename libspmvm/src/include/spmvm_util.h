#ifndef _SPMVM_UTIL_H_
#define _SPMVM_UTIL_H_


#include "spmvm.h"

/******************************************************************************/
/****** Makros ****************************************************************/
/******************************************************************************/
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
    fprintf(stderr, "OpenCL error at %s:%d, %d\n",\
      __FILE__, __LINE__, clerr);\
    fflush(stderr);\
  }\
  }

#define CL_checkerror(err) do{\
  if( CL_SUCCESS != err ){\
    fprintf(stdout, "OpenCL error at %s:%d, %d\n",\
      __FILE__, __LINE__, err);\
    fflush(stdout);\
  }\
  } while(0)
/******************************************************************************/

#ifdef OPENCL


void CL_init();
cl_program CL_registerProgram(char *filename, const char *options);
void CL_bindMatrixToKernel(void *mat, int format, int T, int kernelIdx, int spmvmOptions);

void CL_uploadCRS (LCRP_TYPE *lcrp, SPM_GPUFORMATS *matrixFormats, int spmvmOptions);
void CL_uploadVector( VECTOR_TYPE *vec );
void CL_downloadVector( VECTOR_TYPE *vec );

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
void CL_vecscal(cl_mem a, data_t s, int nRows);
void CL_axpy(cl_mem a, cl_mem b, data_t s, int nRows);
void CL_dotprod(cl_mem a, cl_mem b, data_t *out, int nRows);
void CL_setup_communication(LCRP_TYPE* lcrp, SPM_GPUFORMATS *matrixFormats, int);
void CL_enqueueKernel(cl_kernel kernel);
 
size_t CL_getLocalSize(cl_kernel kernel);
CL_DEVICE_INFO * CL_getDeviceInfo();
void destroyCLdeviceInfo(CL_DEVICE_INFO * di);
#endif


void              SpMVM_printMatrixInfo(LCRP_TYPE *lcrp, char *matrixName, int options);
void              SpMVM_printEnvInfo();
HOSTVECTOR_TYPE * SpMVM_createGlobalHostVector(int nRows, data_t (*fp)(int));
void              SpMVM_referenceSolver(CR_TYPE *lcrp, data_t *rhs, data_t *lhs, int nIter, int spmvmOptions);
void              SpMVM_zeroVector(VECTOR_TYPE *vec);
HOSTVECTOR_TYPE*  SpMVM_newHostVector( const int nRows, data_t (*fp)(int));
VECTOR_TYPE*      SpMVM_newVector( const int nRows );
void              SpMVM_swapVectors(VECTOR_TYPE *v1, VECTOR_TYPE *v2);
void              SpMVM_normalizeVector( VECTOR_TYPE *vec);
void              SpMVM_normalizeHostVector( HOSTVECTOR_TYPE *vec);
char * SpMVM_workdistName(int options);
char * SpMVM_kernelName(int kernel);
void SpMVM_abort(char *s);

/******************************************************************************
  * Distribute a CRS matrix from the master node to all worker nodes.
  *
  * Arguments:
  *   - CR_TYPE *cr
  *     The CRS matrix data which are present on the master node.
  *   - void *deviceFormats
  *     If OpenCL is enabled, this has to be a pointer to a SPM_GPUFORMATS
  *     data structure, holding information about the desired GPU matrix format.
  *     In the non-OpenCL case, this argument is NULL.
  *****************************************************************************/
LCRP_TYPE * SpMVM_distributeCRS (CR_TYPE *cr, void *deviceFormats, int options);

/******************************************************************************
  * Create a CRS matrix on the master node from a given path.
  *
  * Arguments:
  *   - char *matrixPath
  *     The full path to the matrix file. The matrix may either be present in
  *     MatrixMarket format or a binary CRS format which is explained in the
  *     README file.
  *
  * Returns:
  *   a pointer to a CR_TYPE which holds the data of the CRS matrix on the
  *   master node. On the other nodes, a dummy CRS matrix is created.
  *
  * Note that the CR_TYPE created by this functions has to be freed manually by
  * calling SpMVM_freeCRS(CR_TYPE *).
  *****************************************************************************/
CR_TYPE * SpMVM_createGlobalCRS (char *matrixPath);


VECTOR_TYPE * SpMVM_distributeVector(LCRP_TYPE *lcrp, HOSTVECTOR_TYPE *vec);
void SpMVM_collectVectors(LCRP_TYPE *lcrp, VECTOR_TYPE *vec, 
		HOSTVECTOR_TYPE *totalVec, int kernel);
void SpMVM_freeVector( VECTOR_TYPE* const vec );
void SpMVM_freeHostVector( HOSTVECTOR_TYPE* const vec );
void SpMVM_freeCRS( CR_TYPE* const cr );
void SpMVM_freeLCRP( LCRP_TYPE* const );
void SpMVM_permuteVector( data_t* vec, int* perm, int len);
#endif
