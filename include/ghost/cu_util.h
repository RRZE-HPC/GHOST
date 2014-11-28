#ifndef GHOST_CU_UTIL_H
#define GHOST_CU_UTIL_H

#include "config.h"
#include "types.h"
#include "error.h"
#ifdef GHOST_HAVE_CUDA
#include <cublas_v2.h>
#include <cusparse_v2.h>
#endif

#ifdef GHOST_HAVE_CUDA
typedef cublasHandle_t ghost_cublas_handle_t;
typedef cusparseHandle_t ghost_cusparse_handle_t;
#else
typedef int ghost_cublas_handle_t;
typedef int ghost_cusparse_handle_t;
#endif

typedef struct
{
    int ndistinctdevice;
    int *ndevice;
    char **names;
}ghost_gpu_info_t;

#ifdef __cplusplus
extern "C" {
#endif

    ghost_error_t ghost_cu_init(int dev);
    ghost_error_t ghost_cu_malloc(void **mem, size_t bytesize);
    ghost_error_t ghost_cu_malloc_mapped(void **mem, const size_t size);
    ghost_error_t ghost_cu_malloc_pinned(void **mem, const size_t size);
    ghost_error_t ghost_cu_download(void * hostmem, void * devmem, size_t bytesize);
    ghost_error_t ghost_cu_download2d(void *dest, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height);
    ghost_error_t ghost_cu_upload(void * devmem, void *hostmem, size_t bytesize);
    ghost_error_t ghost_cu_upload2d(void *dest, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height);
    ghost_error_t ghost_cu_memcpy(void *dest, void *src, size_t bytesize);
    ghost_error_t ghost_cu_memset(void *s, int c, size_t n);
    ghost_error_t ghost_cu_free(void * mem);
    void ghost_cu_free_host(void * mem);
    ghost_error_t ghost_cu_barrier();
    ghost_error_t ghost_cu_ndevice(int *devcount);
    ghost_error_t ghost_cu_version(int *ver);
    ghost_error_t ghost_cu_gpu_info_create(ghost_gpu_info_t **gpu_info);
    ghost_error_t ghost_cu_device(int *device);
    ghost_error_t ghost_cu_cublas_handle(ghost_cublas_handle_t *handle);
    ghost_error_t ghost_cu_cusparse_handle(ghost_cusparse_handle_t *handle);

#ifdef __cplusplus
}
#endif

#endif 
