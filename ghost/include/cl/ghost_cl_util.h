#ifndef __GHOST_CL_UTIL_H__
#define __GHOST_CL_UTIL_H__

void CL_init();
cl_program CL_registerProgram(const char *filename, const char *options);
void CL_bindMatrixToKernel(void *mat, int format, int T, int kernelIdx, int spmvmOptions);

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
void CL_finish();

void CL_enqueueKernel(cl_kernel kernel, cl_uint dim, size_t *gSize, size_t *lSize);
const char * CL_errorString(cl_int err);
 
size_t CL_getLocalSize(cl_kernel kernel);
CL_DEVICE_INFO * CL_getDeviceInfo();
void destroyCLdeviceInfo(CL_DEVICE_INFO * di);
void CL_barrier();
const char * CL_getVersion();

#endif
