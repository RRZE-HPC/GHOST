/**
 * @file cu_util.h
 * @brief CUDA utility functions.
 * If CUDA ist disabled, the function are still defined but stubs.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
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
typedef struct cudaDeviceProp ghost_cu_deviceprop_t;
#else
typedef int ghost_cublas_handle_t;
typedef int ghost_cusparse_handle_t;
typedef int ghost_cu_deviceprop_t;
#endif

/**
 * @brief Information about avaiable GPUs.
 */
typedef struct {
    /**
     * @brief The number of distinct CUDA device types.
     */
    int ndistinctdevice;
    /**
     * @brief The number of GPUs of each distinct device type.
     */
    int *ndevice;
    /**
     * @brief The names of each distince device type.
     */
    char **names;
} ghost_gpu_info_t;

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @brief Initalize CUDA on a given device.
     *
     * @param dev The device number.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_cu_init(int dev);
    /**
     * @brief Allocate CUDA device memory.
     *
     * @param mem Where to store the memory.
     * @param bytesize The number of bytes to allocate.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_cu_malloc(void **mem, size_t bytesize);
    /**
     * @brief Allocate mapped host memory.
     *
     * @param mem Where to store the memory.
     * @param size The number of bytes to allocate.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_cu_malloc_mapped(void **mem, const size_t size);
    /**
     * @brief Allocate pinned host memory.
     *
     * @param mem Where to store the memory.
     * @param size The number of bytes to allocate.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_cu_malloc_pinned(void **mem, const size_t size);
    /**
     * @brief Download memory from a GPU to the host.
     *
     * @param hostmem The host side memory location.
     * @param devmem The device side memory location.
     * @param bytesize Number of bytes.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_cu_download(void * hostmem, void * devmem, size_t bytesize);
    /**
     * @brief Download strided memory from a GPU to the host.
     * Copy height rows of width bytes each.
     *
     * @param dest The host memory.
     * @param dpitch The pitch in the host memory.
     * @param src The device memory.
     * @param spitch The pitch in the device memory.
     * @param width The number of bytes per row. 
     * @param height The number of rows.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_cu_download2d(void *dest, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height);
    /**
     * @brief Upload memory from the the host to the GPU.
     *
     * @param devmem The device side memory location.
     * @param hostmem The host side memory location.
     * @param bytesize Number of bytes.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_cu_upload(void * devmem, void *hostmem, size_t bytesize);
    /**
     * @brief Upload strided memory from the host to the GPU.
     * Copy height rows of width bytes each.
     *
     * @param dest The device memory.
     * @param dpitch The pitch in the device memory.
     * @param src The host memory.
     * @param spitch The pitch in the host memory.
     * @param width The number of bytes per row. 
     * @param height The number of rows.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_cu_upload2d(void *dest, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height);
    /**
     * @brief Memcpy GPU memory.
     *
     * @param dest The destination memory location.
     * @param src The source memory location.
     * @param bytesize The number of bytes.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_cu_memcpy(void *dest, void *src, size_t bytesize);
    /**
     * @brief Memset GPU memory.
     *
     * @param s The memory location.
     * @param c What to write to the memory.
     * @param n The number of bytes.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_cu_memset(void *s, int c, size_t n);
    /**
     * @brief Free GPU memory.
     *
     * @param mem The memory location.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_cu_free(void * mem);
    /**
     * @brief Free host memory which has been allocated with 
     * ghost_cu_malloc_pinned() or ghost_cu_malloc_mapped().
     *
     * @param mem The memory location.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_cu_free_host(void * mem);
    /**
     * @brief Wait for any outstanding CUDA kernel.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_cu_barrier();
    /**
     * @brief Get the number of available GPUs.
     *
     * @param devcount Where to store the number.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_cu_ndevice(int *devcount);
    /**
     * @brief Get the CUDA version.
     *
     * @param ver Where to store the version.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_cu_version(int *ver);
    /**
     * @brief Get information about available GPUs.
     *
     * @param gpu_info Where to store the GPU info.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_cu_gpu_info_create(ghost_gpu_info_t **gpu_info);
    /**
     * @brief Get the active GPU. 
     *
     * @param device Where to store the device number.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_cu_device(int *device);
    /**
     * @brief Get the cuBLAS handle. 
     *
     * @param handle Where to store the handle.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_cu_cublas_handle(ghost_cublas_handle_t *handle);
    /**
     * @brief Get the CuSparse handle.
     *
     * @param handle Where to store the handle.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_cu_cusparse_handle(ghost_cusparse_handle_t *handle);
    /**
     * @brief Get the CUDA device properties.
     *
     * @param prop Where to store the properties.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_cu_deviceprop(ghost_cu_deviceprop_t *prop);

#ifdef __cplusplus
}
#endif

#endif 
