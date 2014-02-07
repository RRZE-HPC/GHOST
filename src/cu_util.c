#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/core.h"
#include "ghost/util.h"
#include "ghost/mat.h"
#include "ghost/affinity.h"
#include "ghost/constants.h"
#include "ghost/log.h"
#include <string.h>
#include <stdlib.h>
#include <libgen.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

#define CU_MAX_DEVICE_NAME_LEN 500

// TODO private variables
cublasHandle_t ghost_cublas_handle;
int ghost_cu_device;

ghost_error_t ghost_CUDA_init(int dev)
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
#if GHOST_HAVE_CUDA_PINNEDMEM
    CUDA_CALL_RETURN(cudaSetDeviceFlags(cudaDeviceMapHost));
#endif

    return GHOST_SUCCESS;
}

ghost_error_t CU_allocDeviceMemory(void **mem, size_t bytesize)
{
    if (bytesize == 0) {
        return GHOST_SUCCESS;
    }

    CUDA_CALL_RETURN(cudaMalloc(mem,bytesize));

    return GHOST_SUCCESS;
}

ghost_error_t CU_copyDeviceToHost(void * hostmem, void * devmem, size_t bytesize) 
{
    if (bytesize > 0) {
        CUDA_CALL_RETURN(cudaMemcpy(hostmem,devmem,bytesize,cudaMemcpyDeviceToHost));
    }

    return GHOST_SUCCESS;
}

ghost_error_t CU_copyDeviceToDevice(void *dest, void *src, size_t bytesize)
{
    if (bytesize > 0) {
        CUDA_CALL_RETURN(cudaMemcpy(dest,src,bytesize,cudaMemcpyDeviceToDevice));
    }

    return GHOST_SUCCESS;

} 

ghost_error_t CU_memset(void *s, int c, size_t n)
{
    CUDA_CALL_RETURN(cudaMemset(s,c,n));

    return GHOST_SUCCESS;
} 

ghost_error_t CU_copyHostToDeviceOffset(void * devmem, void *hostmem,
        size_t bytesize, size_t offset)
{
    if (bytesize > 0) {
        CUDA_CALL_RETURN(cudaMemcpy(((char *)devmem)+offset,((char *)hostmem)+offset,bytesize,cudaMemcpyHostToDevice));
    }
    return GHOST_SUCCESS;
}

ghost_error_t CU_copyHostToDevice(void * devmem, void *hostmem, size_t bytesize)
{
    if (bytesize > 0) {
        CUDA_CALL_RETURN(cudaMemcpy(devmem,hostmem,bytesize,cudaMemcpyHostToDevice));
    }

    return GHOST_SUCCESS;
}

ghost_error_t CU_freeDeviceMemory(void * mem)
{
    CUDA_CALL_RETURN(cudaFree(mem));

    return GHOST_SUCCESS;
}

ghost_error_t CU_barrier()
{
    CUDA_CALL_RETURN(cudaDeviceSynchronize());

    return GHOST_SUCCESS;
}

ghost_error_t CU_finish() 
{

    return GHOST_SUCCESS;
}


ghost_error_t CU_uploadVector( ghost_vec_t *vec )
{
    return CU_copyHostToDevice(vec->CU_val,vec->val,vec->traits->nrows*ghost_sizeofDataType(vec->traits->datatype));
}

ghost_error_t CU_downloadVector( ghost_vec_t *vec )
{
    return CU_copyDeviceToHost(vec->val,vec->CU_val,vec->traits->nrows*ghost_sizeofDataType(vec->traits->datatype));
}

static int stringcmp(const void *x, const void *y)
{
    return (strcmp((char *)x, (char *)y));
}

ghost_error_t CU_getDeviceCount(int *devcount)
{
    CUDA_CALL_RETURN(cudaGetDeviceCount(devcount));

    return GHOST_SUCCESS;
}


ghost_error_t CU_getDeviceInfo(ghost_gpu_info_t **devInfo)
{
    ghost_error_t ret = GHOST_SUCCESS;
    (*devInfo) = ghost_malloc(sizeof(ghost_gpu_info_t));
    (*devInfo)->nDistinctDevices = 1;
    (*devInfo)->names = NULL;
    (*devInfo)->nDevices = NULL;

    int me,size,i;
    ghost_type_t ghost_type;
    char name[CU_MAX_DEVICE_NAME_LEN];
    char *names = NULL;
    int *displs = NULL;
    int *recvcounts = NULL;

    GHOST_CALL_RETURN(ghost_getRank(MPI_COMM_WORLD,&me));
    GHOST_CALL_RETURN(ghost_getNumberOfRanks(MPI_COMM_WORLD,&size));
    GHOST_CALL_RETURN(ghost_getType(&ghost_type));

    if (ghost_type == GHOST_TYPE_CUDAMGMT) {
        struct cudaDeviceProp devProp;
        CUDA_CALL_GOTO(cudaGetDeviceProperties(&devProp,ghost_cu_device),err,ret);
        strncpy(name,devProp.name,CU_MAX_DEVICE_NAME_LEN);
    } else {
        strncpy(name,"None",5);
    }


    if (me==0) {
        names = (char *)ghost_malloc(size*CU_MAX_DEVICE_NAME_LEN*sizeof(char));
        recvcounts = (int *)ghost_malloc(sizeof(int)*size);
        displs = (int *)ghost_malloc(sizeof(int)*size);
        
        for (i=0; i<size; i++) {
            recvcounts[i] = CU_MAX_DEVICE_NAME_LEN;
            displs[i] = i*CU_MAX_DEVICE_NAME_LEN;

        }
    }


#ifdef GHOST_HAVE_MPI
    MPI_safecall(MPI_Gatherv(name,CU_MAX_DEVICE_NAME_LEN,MPI_CHAR,names,
                recvcounts,displs,MPI_CHAR,0,MPI_COMM_WORLD));
#else
    strncpy(names,name,CU_MAX_DEVICE_NAME_LEN);
#endif

    if (me==0) {
        qsort(names,size,CU_MAX_DEVICE_NAME_LEN*sizeof(char),stringcmp);
        for (i=1; i<size; i++) {
            if (strcmp(names+(i-1)*CU_MAX_DEVICE_NAME_LEN,
                        names+i*CU_MAX_DEVICE_NAME_LEN)) {
                (*devInfo)->nDistinctDevices++;
            }
        }
    }
/*
#if GHOST_HAVE_MPI
    MPI_safecall(MPI_Bcast(&((*devInfo)->nDistinctDevices),1,MPI_INT,0,MPI_COMM_WORLD));
#endif
*/
    (*devInfo)->nDevices = ghost_malloc(sizeof(int)*(*devInfo)->nDistinctDevices);
    (*devInfo)->names = ghost_malloc(sizeof(char *)*(*devInfo)->nDistinctDevices);
    for (i=0; i<(*devInfo)->nDistinctDevices; i++) {
        (*devInfo)->names[i] = ghost_malloc(sizeof(char)*CU_MAX_DEVICE_NAME_LEN);
        (*devInfo)->nDevices[i] = 1;
    }

    if (me==0) {
        strncpy((*devInfo)->names[0],names,CU_MAX_DEVICE_NAME_LEN);

        int distIdx = 1;
        for (i=1; i<size; i++) {
            if (strcmp(names+(i-1)*CU_MAX_DEVICE_NAME_LEN,
                        names+i*CU_MAX_DEVICE_NAME_LEN)) {
                strncpy((*devInfo)->names[distIdx],names+i*CU_MAX_DEVICE_NAME_LEN,CU_MAX_DEVICE_NAME_LEN);
                distIdx++;
            } else {
                (*devInfo)->nDevices[distIdx-1]++;
            }
        }
        free(names);
    }
/*
#if GHOST_HAVE_MPI
    MPI_safecall(MPI_Bcast((*devInfo)->nDevices,(*devInfo)->nDistinctDevices,MPI_INT,0,MPI_COMM_WORLD));

    for (i=0; i<(*devInfo)->nDistinctDevices; i++) {
        MPI_safecall(MPI_Bcast((*devInfo)->names[i],CU_MAX_DEVICE_NAME_LEN,MPI_CHAR,0,MPI_COMM_WORLD));
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

ghost_error_t CU_getVersion(int *ver)
{

    CUDA_CALL_RETURN(cudaRuntimeGetVersion(ver));
    
    return GHOST_SUCCESS;
}
