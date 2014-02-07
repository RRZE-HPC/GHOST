#ifndef GHOST_ghost_cu_UTIL_H
#define GHOST_ghost_cu_UTIL_H

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

ghost_error_t ghost_cu_init(int dev);
ghost_error_t ghost_cu_allocDeviceMemory(void **mem, size_t bytesize);
ghost_error_t ghost_cu_copyDeviceToHost(void * hostmem, void * devmem, size_t bytesize);
ghost_error_t ghost_cu_copyHostToDeviceOffset(void * devmem, void *hostmem, size_t bytesize, size_t offset);
ghost_error_t ghost_cu_copyHostToDevice(void * devmem, void *hostmem, size_t bytesize);
ghost_error_t ghost_cu_copyDeviceToDevice(void *dest, void *src, size_t bytesize);
ghost_error_t ghost_cu_freeDeviceMemory(void * mem);
ghost_error_t ghost_cu_memset(void *s, int c, size_t n);
ghost_error_t ghost_cu_barrier();
ghost_error_t ghost_cu_finish();
ghost_error_t ghost_cu_uploadVector( ghost_vec_t *vec );
ghost_error_t ghost_cu_downloadVector( ghost_vec_t *vec );
ghost_error_t ghost_cu_getDeviceCount(int *devcount);
ghost_error_t ghost_cu_getVersion(int *ver);
ghost_error_t ghost_cu_getDeviceInfo(ghost_gpu_info_t **);
ghost_error_t ghost_cu_malloc_mapped(void **mem, const size_t size);

#ifdef __cplusplus
}
#endif

#endif 
