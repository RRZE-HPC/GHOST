#ifndef GHOST_CU_UTIL_H
#define GHOST_CU_UTIL_H

#include <cublas_v2.h>
#include "config.h"
#include "types.h"
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
ghost_error_t ghost_cu_malloc(void **mem, size_t bytesize);
ghost_error_t ghost_cu_malloc_mapped(void **mem, const size_t size);
ghost_error_t ghost_cu_download(void * hostmem, void * devmem, size_t bytesize);
ghost_error_t ghost_cu_upload(void * devmem, void *hostmem, size_t bytesize);
ghost_error_t ghost_cu_memcpy(void *dest, void *src, size_t bytesize);
ghost_error_t ghost_cu_memset(void *s, int c, size_t n);
ghost_error_t ghost_cu_free(void * mem);
ghost_error_t ghost_cu_barrier();
ghost_error_t ghost_cu_finish();
ghost_error_t ghost_cu_getDeviceCount(int *devcount);
ghost_error_t ghost_cu_getVersion(int *ver);
ghost_error_t ghost_cu_getDeviceInfo(ghost_gpu_info_t **);
ghost_error_t ghost_cu_getDevice(int *device);
ghost_error_t ghost_cu_getCublasHandle(cublasHandle_t *handle);

#ifdef __cplusplus
}
#endif

#endif 
