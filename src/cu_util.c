#include "ghost/context.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/mat.h"
#include "ghost/affinity.h"
#include "ghost/constants.h"
#include <string.h>
#include <stdlib.h>
#include <libgen.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

#define CU_MAX_DEVICE_NAME_LEN 500


cublasHandle_t ghost_cublas_handle;
int ghost_cu_device;

void ghost_CUDA_init(int dev)
{
    int nDevs = 0;
    CU_safecall(cudaGetDeviceCount(&nDevs));

    DEBUG_LOG(2,"There are %d CUDA devices attached to the node",nDevs);

    if (dev<nDevs) {
        ghost_cu_device = dev;

        DEBUG_LOG(1,"Selecting CUDA device %d",ghost_cu_device);
        CU_safecall(cudaSetDevice(ghost_cu_device));
    }
    CUBLAS_safecall(cublasCreate(&ghost_cublas_handle));
#if GHOST_HAVE_CUDA_PINNEDMEM
    CU_safecall(cudaSetDeviceFlags(cudaDeviceMapHost));
#endif
}

void * CU_allocDeviceMemory( size_t bytesize )
{
    if (bytesize == 0)
        return NULL;

    void *ret;
    CU_safecall(cudaMalloc(&ret,bytesize));

    return ret;
}

void CU_copyDeviceToHost(void * hostmem, void * devmem, size_t bytesize) 
{
    if (bytesize > 0)
        CU_safecall(cudaMemcpy(hostmem,devmem,bytesize,cudaMemcpyDeviceToHost));
}

void CU_copyDeviceToDevice(void *dest, void *src, size_t bytesize)
{
    if (bytesize > 0)
        CU_safecall(cudaMemcpy(dest,src,bytesize,cudaMemcpyDeviceToDevice));

} 

void CU_memset(void *s, int c, size_t n)
{
    CU_safecall(cudaMemset(s,c,n));
} 

void CU_copyHostToDeviceOffset(void * devmem, void *hostmem,
        size_t bytesize, size_t offset)
{
    if (bytesize > 0)
        CU_safecall(cudaMemcpy(((char *)devmem)+offset,((char *)hostmem)+offset,bytesize,cudaMemcpyHostToDevice));
}

void CU_copyHostToDevice(void * devmem, void *hostmem, size_t bytesize)
{
    //WARNING_LOG("%p %p %lu %d",devmem,hostmem,bytesize,device);
    if (bytesize > 0)
        CU_safecall(cudaMemcpy(devmem,hostmem,bytesize,cudaMemcpyHostToDevice));
}

void CU_freeDeviceMemory(void * mem)
{
    CU_safecall(cudaFree(mem));
}

void CU_barrier()
{
    CU_safecall(cudaDeviceSynchronize());
}

void CU_finish() 
{

}


void CU_uploadVector( ghost_vec_t *vec )
{
    CU_copyHostToDevice(vec->CU_val,vec->val,vec->traits->nrows*ghost_sizeofDataType(vec->traits->datatype));
}

void CU_downloadVector( ghost_vec_t *vec )
{
    CU_copyDeviceToHost(vec->val,vec->CU_val,vec->traits->nrows*ghost_sizeofDataType(vec->traits->datatype));
}

static int stringcmp(const void *x, const void *y)
{
    return (strcmp((char *)x, (char *)y));
}

int CU_getDeviceCount(int *devcount)
{
    CU_safecall(cudaGetDeviceCount(devcount));

    return GHOST_SUCCESS;
}


ghost_acc_info_t *CU_getDeviceInfo() 
{
    ghost_acc_info_t *devInfo = ghost_malloc(sizeof(ghost_acc_info_t));
    devInfo->nDistinctDevices = 1;

    int me,size,i;
    char name[CU_MAX_DEVICE_NAME_LEN];
    char *names = NULL;

    me = ghost_getRank(MPI_COMM_WORLD);
    size = ghost_getNumberOfRanks(MPI_COMM_WORLD);

    if (ghost_type == GHOST_TYPE_CUDAMGMT) {
        struct cudaDeviceProp devProp;
        CU_safecall(cudaGetDeviceProperties(&devProp,ghost_cu_device));
        strncpy(name,devProp.name,CU_MAX_DEVICE_NAME_LEN);
    } else {
        strncpy(name,"None",5);
    }

    int *displs;
    int *recvcounts;

    if (me==0) {
        names = (char *)ghost_malloc(size*CU_MAX_DEVICE_NAME_LEN*sizeof(char));
        recvcounts = (int *)ghost_malloc(sizeof(int)*ghost_getNumberOfRanks(MPI_COMM_WORLD));
        displs = (int *)ghost_malloc(sizeof(int)*ghost_getNumberOfRanks(MPI_COMM_WORLD));
        
        for (i=0; i<ghost_getNumberOfRanks(MPI_COMM_WORLD); i++) {
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
                devInfo->nDistinctDevices++;
            }
        }
    }
/*
#if GHOST_HAVE_MPI
    MPI_safecall(MPI_Bcast(&(devInfo->nDistinctDevices),1,MPI_INT,0,MPI_COMM_WORLD));
#endif
*/
    devInfo->nDevices = ghost_malloc(sizeof(int)*devInfo->nDistinctDevices);
    devInfo->names = ghost_malloc(sizeof(char *)*devInfo->nDistinctDevices);
    for (i=0; i<devInfo->nDistinctDevices; i++) {
        devInfo->names[i] = ghost_malloc(sizeof(char)*CU_MAX_DEVICE_NAME_LEN);
        devInfo->nDevices[i] = 1;
    }

    if (me==0) {
        strncpy(devInfo->names[0],names,CU_MAX_DEVICE_NAME_LEN);

        int distIdx = 1;
        for (i=1; i<size; i++) {
            if (strcmp(names+(i-1)*CU_MAX_DEVICE_NAME_LEN,
                        names+i*CU_MAX_DEVICE_NAME_LEN)) {
                strncpy(devInfo->names[distIdx],names+i*CU_MAX_DEVICE_NAME_LEN,CU_MAX_DEVICE_NAME_LEN);
                distIdx++;
            } else {
                devInfo->nDevices[distIdx-1]++;
            }
        }
        free(names);
    }
/*
#if GHOST_HAVE_MPI
    MPI_safecall(MPI_Bcast(devInfo->nDevices,devInfo->nDistinctDevices,MPI_INT,0,MPI_COMM_WORLD));

    for (i=0; i<devInfo->nDistinctDevices; i++) {
        MPI_safecall(MPI_Bcast(devInfo->names[i],CU_MAX_DEVICE_NAME_LEN,MPI_CHAR,0,MPI_COMM_WORLD));
    }
#endif
*/
    return devInfo;
}

const char * CU_getVersion()
{
    int rtVersion, drVersion;
    CU_safecall(cudaRuntimeGetVersion(&rtVersion));
    CU_safecall(cudaDriverGetVersion(&drVersion));
    char *version = (char *)malloc(1024); // TODO as parameter, else: leak
    snprintf(version,1024,"Runtime: %d, Driver: %d",rtVersion,drVersion);
    return version;
}

void *ghost_cu_malloc_mapped(const size_t size)
{
    void *mem = NULL;

    if (size/(1024.*1024.*1024.) > 1.) {
        DEBUG_LOG(1,"Allocating big array of size %f GB",size/(1024.*1024.*1024.));
    }

    cudaHostAlloc(&mem,size,cudaHostAllocMapped);

    if( ! mem ) {
        ABORT("Error in memory allocation of %zu bytes: %s",size,strerror(errno));
    }
    return mem;
}
