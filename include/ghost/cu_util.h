#ifndef GHOST_CU_UTIL_H
#define GHOST_CU_UTIL_H

#include "config.h"
#include "types.h"
#include "vec.h"
#include "error.h"

typedef struct
{
    int nDistinctDevices;
    int *nDevices;
    char **names;
}ghost_gpu_info_t;

#ifdef __cplusplus
extern "C" {
#endif

void ghost_CUDA_init(int dev);
void * CU_allocDeviceMemory( size_t bytesize );
void CU_copyDeviceToHost(void * hostmem, void * devmem, size_t bytesize);
void CU_copyHostToDeviceOffset(void * devmem, void *hostmem, size_t bytesize, size_t offset);
void CU_copyHostToDevice(void * devmem, void *hostmem, size_t bytesize);
void CU_copyDeviceToDevice(void *dest, void *src, size_t bytesize);
void CU_freeDeviceMemory(void * mem);
void CU_memset(void *s, int c, size_t n);
void CU_barrier();
void CU_finish();
void CU_uploadVector( ghost_vec_t *vec );
void CU_downloadVector( ghost_vec_t *vec );
int CU_getDeviceCount(int *devcount);
const char * CU_getVersion();
ghost_error_t CU_getDeviceInfo(ghost_gpu_info_t **);
void *ghost_cu_malloc_mapped(const size_t size);

#ifdef __cplusplus
}
#endif

#endif 
