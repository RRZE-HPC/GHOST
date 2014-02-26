/**
 * @file cu_util.h
 * @brief CUDA utility functions.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/core.h"
#include "ghost/util.h"
#include "ghost/sparsemat.h"
#include "ghost/locality.h"
#include "ghost/log.h"
#include <string.h>
#include <stdlib.h>
#include <libgen.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define ghost_cu_MAX_DEVICE_NAME_LEN 500

static cublasHandle_t ghost_cublas_handle;
static int ghost_cu_device = -1;

ghost_error_t ghost_cu_init(int dev)
{
    int nDevs = 0;
    CUDA_CALL_RETURN(cudaGetDeviceCount(&nDevs));

    DEBUG_LOG(2,"There are %d CUDA devices attached to the node",nDevs);

    if (dev<nDevs) {
        ghost_cu_device = dev;

        DEBUG_LOG(1,"Selecting CUDA device %d",ghost_cu_device);
        CUDA_CALL_RETURN(cudaSetDevice(ghost_cu_device));
    }
    CUBLAS_CALL_RETURN(cublasCreate(&ghost_cublas_handle));
#ifdef GHOST_HAVE_CUDA_PINNEDMEM
    CUDA_CALL_RETURN(cudaSetDeviceFlags(cudaDeviceMapHost));
#endif

    return GHOST_SUCCESS;
}

ghost_error_t ghost_cu_malloc(void **mem, size_t bytesize)
{
    if (bytesize == 0) {
        return GHOST_SUCCESS;
    }

    CUDA_CALL_RETURN(cudaMalloc(mem,bytesize));

    return GHOST_SUCCESS;
}

ghost_error_t ghost_cu_memcpy(void *dest, void *src, size_t bytesize)
{
    if (bytesize > 0) {
        CUDA_CALL_RETURN(cudaMemcpy(dest,src,bytesize,cudaMemcpyDeviceToDevice));
    }

    return GHOST_SUCCESS;

} 

ghost_error_t ghost_cu_memset(void *s, int c, size_t n)
{
    CUDA_CALL_RETURN(cudaMemset(s,c,n));

    return GHOST_SUCCESS;
} 

ghost_error_t ghost_cu_upload(void * devmem, void *hostmem,
        size_t bytesize)
{
    if (bytesize > 0) {
        CUDA_CALL_RETURN(cudaMemcpy(devmem,hostmem,bytesize,cudaMemcpyHostToDevice));
    }
    return GHOST_SUCCESS;
}

ghost_error_t ghost_cu_download(void *hostmem, void *devmem,
        size_t bytesize)
{
    if (bytesize > 0) {
        CUDA_CALL_RETURN(cudaMemcpy(hostmem,devmem,bytesize,cudaMemcpyDeviceToHost));
    }
    return GHOST_SUCCESS;
}

ghost_error_t ghost_cu_free(void * mem)
{
    CUDA_CALL_RETURN(cudaFree(mem));

    return GHOST_SUCCESS;
}

ghost_error_t ghost_cu_barrier()
{
    CUDA_CALL_RETURN(cudaDeviceSynchronize());

    return GHOST_SUCCESS;
}

ghost_error_t ghost_cu_finish() 
{

    return GHOST_SUCCESS;
}

static int stringcmp(const void *x, const void *y)
{
    return (strcmp((char *)x, (char *)y));
}

ghost_error_t ghost_cu_getDeviceCount(int *devcount)
{
    CUDA_CALL_RETURN(cudaGetDeviceCount(devcount));

    return GHOST_SUCCESS;
}


ghost_error_t ghost_cu_getDeviceInfo(ghost_gpu_info_t **devInfo)
{
    ghost_error_t ret = GHOST_SUCCESS;
    GHOST_CALL_GOTO(ghost_malloc((void **)devInfo,sizeof(ghost_gpu_info_t)),err,ret);
    (*devInfo)->nDistinctDevices = 1;
    (*devInfo)->names = NULL;
    (*devInfo)->nDevices = NULL;

    int me,size,i;
    ghost_type_t ghost_type;
    char name[ghost_cu_MAX_DEVICE_NAME_LEN];
    char *names = NULL;
    int *displs = NULL;
    int *recvcounts = NULL;

    GHOST_CALL_RETURN(ghost_getRank(MPI_COMM_WORLD,&me));
    GHOST_CALL_RETURN(ghost_getNumberOfRanks(MPI_COMM_WORLD,&size));
    GHOST_CALL_RETURN(ghost_type_get(&ghost_type));

    if (ghost_type == GHOST_TYPE_CUDA) {
        struct cudaDeviceProp devProp;
        CUDA_CALL_GOTO(cudaGetDeviceProperties(&devProp,ghost_cu_device),err,ret);
        strncpy(name,devProp.name,ghost_cu_MAX_DEVICE_NAME_LEN);
    } else {
        strncpy(name,"None",5);
    }


    if (me==0) {
        GHOST_CALL_RETURN(ghost_malloc((void **)&names,size*ghost_cu_MAX_DEVICE_NAME_LEN*sizeof(char)));
        GHOST_CALL_RETURN(ghost_malloc((void **)&recvcounts,sizeof(int)*size));
        GHOST_CALL_RETURN(ghost_malloc((void **)&displs,sizeof(int)*size));
        
        for (i=0; i<size; i++) {
            recvcounts[i] = ghost_cu_MAX_DEVICE_NAME_LEN;
            displs[i] = i*ghost_cu_MAX_DEVICE_NAME_LEN;

        }
    }


#ifdef GHOST_HAVE_MPI
    MPI_CALL_RETURN(MPI_Gatherv(name,ghost_cu_MAX_DEVICE_NAME_LEN,MPI_CHAR,names,
                recvcounts,displs,MPI_CHAR,0,MPI_COMM_WORLD));
#else
    strncpy(names,name,ghost_cu_MAX_DEVICE_NAME_LEN);
#endif

    if (me==0) {
        qsort(names,size,ghost_cu_MAX_DEVICE_NAME_LEN*sizeof(char),stringcmp);
        for (i=1; i<size; i++) {
            if (strcmp(names+(i-1)*ghost_cu_MAX_DEVICE_NAME_LEN,
                        names+i*ghost_cu_MAX_DEVICE_NAME_LEN)) {
                (*devInfo)->nDistinctDevices++;
            }
        }
    }
/*
#ifdef GHOST_HAVE_MPI
    MPI_safecall(MPI_Bcast(&((*devInfo)->nDistinctDevices),1,MPI_INT,0,MPI_COMM_WORLD));
#endif
*/
    GHOST_CALL_GOTO(ghost_malloc((void **)&(*devInfo)->nDevices,sizeof(int)*(*devInfo)->nDistinctDevices),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&(*devInfo)->names,sizeof(char *)*(*devInfo)->nDistinctDevices),err,ret);
    for (i=0; i<(*devInfo)->nDistinctDevices; i++) {
        GHOST_CALL_GOTO(ghost_malloc((void **)&(*devInfo)->names[i],sizeof(char)*ghost_cu_MAX_DEVICE_NAME_LEN),err,ret);
        (*devInfo)->nDevices[i] = 1;
    }

    if (me==0) {
        strncpy((*devInfo)->names[0],names,ghost_cu_MAX_DEVICE_NAME_LEN);

        int distIdx = 1;
        for (i=1; i<size; i++) {
            if (strcmp(names+(i-1)*ghost_cu_MAX_DEVICE_NAME_LEN,
                        names+i*ghost_cu_MAX_DEVICE_NAME_LEN)) {
                strncpy((*devInfo)->names[distIdx],names+i*ghost_cu_MAX_DEVICE_NAME_LEN,ghost_cu_MAX_DEVICE_NAME_LEN);
                distIdx++;
            } else {
                (*devInfo)->nDevices[distIdx-1]++;
            }
        }
        free(names);
    }
/*
#ifdef GHOST_HAVE_MPI
    MPI_safecall(MPI_Bcast((*devInfo)->nDevices,(*devInfo)->nDistinctDevices,MPI_INT,0,MPI_COMM_WORLD));

    for (i=0; i<(*devInfo)->nDistinctDevices; i++) {
        MPI_safecall(MPI_Bcast((*devInfo)->names[i],ghost_cu_MAX_DEVICE_NAME_LEN,MPI_CHAR,0,MPI_COMM_WORLD));
    }
#endif
*/

    goto out;
err:
    if (*devInfo) {
        free((*devInfo)->names); ((*devInfo)->names) = NULL;
    }
    free(*devInfo); *devInfo = NULL;
    free(recvcounts); recvcounts = NULL;
    free(displs); displs = NULL;
    free(names); names = NULL;

out:
    return ret;
}

ghost_error_t ghost_cu_malloc_mapped(void **mem, const size_t size)
{

    if (size/(1024.*1024.*1024.) > 1.) {
        DEBUG_LOG(1,"Allocating big array of size %f GB",size/(1024.*1024.*1024.));
    }

    CUDA_CALL_RETURN(cudaHostAlloc(mem,size,cudaHostAllocMapped));

    return GHOST_SUCCESS;
}

ghost_error_t ghost_cu_getDevice(int *device)
{
    if (ghost_cu_device < 0) {
        ERROR_LOG("CUDA not initialized!");
        return GHOST_ERR_CUDA;
    }
    *device = ghost_cu_device;

    return GHOST_SUCCESS;
}

ghost_error_t ghost_cu_getCublasHandle(cublasHandle_t *handle)
{
    if (!ghost_cublas_handle) {
        ERROR_LOG("CUBLAS not initialized!");
        return GHOST_ERR_CUBLAS;
    }
    *handle = ghost_cublas_handle;

    return GHOST_SUCCESS;
}

ghost_error_t ghost_cu_getVersion(int *ver)
{

    CUDA_CALL_RETURN(cudaRuntimeGetVersion(ver));
    
    return GHOST_SUCCESS;
}
