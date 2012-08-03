#ifndef _OCLFUN_H_
#define _OCLFUN_H_

#include "spmvm_globals.h"
#include <CL/cl.h>
#include <sys/param.h>



void CL_init(SPM_GPUFORMATS *);
cl_program CL_registerProgram(char *filename, const char *options);
void CL_bindMatrixToKernel(void *mat, int format, int T, int kernelIdx);

void CL_uploadCRS (LCRP_TYPE *lcrp, SPM_GPUFORMATS *matrixFormats);
void CL_uploadVector( VECTOR_TYPE *vec );
void CL_downloadVector( VECTOR_TYPE *vec );

cl_mem CL_allocDeviceMemory( size_t );
cl_mem CL_allocDeviceMemoryMapped( size_t bytesize, void *hostPtr );
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
void CL_vecscal(cl_mem a, real s, int nRows);
void CL_axpy(cl_mem a, cl_mem b, real s, int nRows);
void CL_dotprod(cl_mem a, cl_mem b, real *out, int nRows);
void CL_setup_communication(LCRP_TYPE* lcrp, SPM_GPUFORMATS *matrixFormats);

#endif
