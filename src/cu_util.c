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
#include "ghost/rand.h"
#include <string.h>
#include <stdlib.h>
#include <libgen.h>
#include <unistd.h>
#ifdef GHOST_HAVE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#endif

#define ghost_cu_MAX_DEVICE_NAME_LEN 500

#ifdef GHOST_HAVE_CUDA
static cublasHandle_t ghost_cublas_handle = NULL;
static cusparseHandle_t ghost_cusparse_handle = NULL;
static struct cudaDeviceProp ghost_cu_device_prop;
static ghost_cu_rand_generator cu_rand_generator = NULL;
static int cu_device = -1;
#endif

ghost_error ghost_cu_init(int dev)
{
#ifdef GHOST_HAVE_CUDA
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_SETUP);
    int nDevs = 0;
    CUDA_CALL_RETURN(cudaGetDeviceCount(&nDevs));

    DEBUG_LOG(2,"There are %d CUDA devices attached to the node",nDevs);

    if (dev<nDevs) {
        cu_device = dev;

        DEBUG_LOG(1,"Selecting CUDA device %d",cu_device);
        CUDA_CALL_RETURN(cudaSetDevice(cu_device));
    } else {
        ERROR_LOG("CUDA device out of range!");
        return GHOST_ERR_CUDA;
    }
    CUBLAS_CALL_RETURN(cublasCreate(&ghost_cublas_handle));
    CUSPARSE_CALL_RETURN(cusparseCreate(&ghost_cusparse_handle));
    CUDA_CALL_RETURN(cudaGetDeviceProperties(&ghost_cu_device_prop,cu_device));

#ifdef GHOST_HAVE_CUDA_PINNEDMEM
    CUDA_CALL_RETURN(cudaSetDeviceFlags(cudaDeviceMapHost));
#endif
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_SETUP);
#else
    UNUSED(dev);
#endif


    return GHOST_SUCCESS;
}

ghost_error ghost_cu_malloc(void **mem, size_t bytesize)
{
    ghost_error ret = GHOST_SUCCESS;
#ifdef GHOST_HAVE_CUDA
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    CUDA_CALL_GOTO(cudaMalloc(mem,bytesize),err,ret);
    goto out;
err:
    ERROR_LOG("CUDA malloc with %zu bytes failed!",bytesize);
out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
#else
    UNUSED(mem);
    UNUSED(bytesize);
#endif
    return ret;
}

ghost_error ghost_cu_memcpy(void *dest, void *src, size_t bytesize)
{
#ifdef GHOST_HAVE_CUDA
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    if (bytesize > 0) {
        CUDA_CALL_RETURN(cudaMemcpy(dest,src,bytesize,cudaMemcpyDeviceToDevice));
    }
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
#else
    UNUSED(dest);
    UNUSED(src);
    UNUSED(bytesize);
#endif

    return GHOST_SUCCESS;

} 

ghost_error ghost_cu_memcpy2d(void *dest, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height)
{
#ifdef GHOST_HAVE_CUDA
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    if (width > 0 && height > 0) {
        CUDA_CALL_RETURN(cudaMemcpy2D(dest,dpitch,src,spitch,width,height,cudaMemcpyDeviceToDevice));
    }
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
#else
    UNUSED(dest);
    UNUSED(dpitch);
    UNUSED(src);
    UNUSED(spitch);
    UNUSED(width);
    UNUSED(height);
#endif
    return GHOST_SUCCESS;
}

ghost_error ghost_cu_memset(void *s, int c, size_t n)
{
#ifdef GHOST_HAVE_CUDA
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    CUDA_CALL_RETURN(cudaMemset(s,c,n));
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
#else
    UNUSED(s);
    UNUSED(c);
    UNUSED(n);
#endif

    return GHOST_SUCCESS;
} 

ghost_error ghost_cu_upload(void * devmem, void *hostmem,
        size_t bytesize)
{
#ifdef GHOST_HAVE_CUDA
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_COMMUNICATION);
    if (bytesize > 0) {
        CUDA_CALL_RETURN(cudaMemcpy(devmem,hostmem,bytesize,cudaMemcpyHostToDevice));
    }
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_COMMUNICATION);
#else
    UNUSED(devmem);
    UNUSED(hostmem);
    UNUSED(bytesize);
#endif
    return GHOST_SUCCESS;
}

ghost_error ghost_cu_upload2d(void *dest, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height)
{
#ifdef GHOST_HAVE_CUDA
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_COMMUNICATION);
    if (width > 0 && height > 0) {
        CUDA_CALL_RETURN(cudaMemcpy2D(dest,dpitch,src,spitch,width,height,cudaMemcpyHostToDevice));
    }
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_COMMUNICATION);
#else
    UNUSED(dest);
    UNUSED(dpitch);
    UNUSED(src);
    UNUSED(spitch);
    UNUSED(width);
    UNUSED(height);
#endif
    return GHOST_SUCCESS;
}

ghost_error ghost_cu_download2d(void *dest, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height)
{
#ifdef GHOST_HAVE_CUDA
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_COMMUNICATION);
    if (width > 0 && height > 0) {
        CUDA_CALL_RETURN(cudaMemcpy2D(dest,dpitch,src,spitch,width,height,cudaMemcpyDeviceToHost));
    }
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_COMMUNICATION);
#else
    UNUSED(dest);
    UNUSED(dpitch);
    UNUSED(src);
    UNUSED(spitch);
    UNUSED(width);
    UNUSED(height);
#endif
    return GHOST_SUCCESS;
}

ghost_error ghost_cu_download(void *hostmem, void *devmem,
        size_t bytesize)
{
#ifdef GHOST_HAVE_CUDA
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_COMMUNICATION);
    if (bytesize > 0) {
        CUDA_CALL_RETURN(cudaMemcpy(hostmem,devmem,bytesize,cudaMemcpyDeviceToHost));
    }
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_COMMUNICATION);
#else
    UNUSED(devmem);
    UNUSED(hostmem);
    UNUSED(bytesize);
#endif
    return GHOST_SUCCESS;
}

ghost_error ghost_cu_free(void * mem)
{
#ifdef GHOST_HAVE_CUDA
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    CUDA_CALL_RETURN(cudaFree(mem));
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
#else
    UNUSED(mem);
#endif

    return GHOST_SUCCESS;
}

ghost_error ghost_cu_barrier()
{
#ifdef GHOST_HAVE_CUDA
    CUDA_CALL_RETURN(cudaDeviceSynchronize());
#endif

    return GHOST_SUCCESS;
}

static int stringcmp(const void *x, const void *y)
{
    return (strcmp((char *)x, (char *)y));
}

ghost_error ghost_cu_ndevice(int *devcount)
{
#ifdef GHOST_HAVE_CUDA
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    if (cudaGetDeviceCount(devcount) == cudaErrorNoDevice) {
        *devcount = 0;
    }
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
#else
    *devcount = 0;
#endif

    return GHOST_SUCCESS;
}


ghost_error ghost_cu_gpu_info_create(ghost_gpu_info **devInfo)
{
    ghost_error ret = GHOST_SUCCESS;
#ifdef GHOST_HAVE_CUDA
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_COMMUNICATION);
    ghost_mpi_comm ghost_cuda_comm;
    GHOST_CALL_GOTO(ghost_cuda_comm_get(&ghost_cuda_comm),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)devInfo,sizeof(ghost_gpu_info)),err,ret);
    (*devInfo)->ndistinctdevice = 1;
    (*devInfo)->names = NULL;
    (*devInfo)->ndevice = NULL;

    int me,size,i;
    ghost_type ghost_type;
    char name[ghost_cu_MAX_DEVICE_NAME_LEN];
    char *names = NULL;
    int *displs = NULL;
    int *recvcounts = NULL;

    GHOST_CALL_RETURN(ghost_rank(&me, ghost_cuda_comm));
    GHOST_CALL_RETURN(ghost_nrank(&size, ghost_cuda_comm));
    GHOST_CALL_RETURN(ghost_type_get(&ghost_type));

    if (ghost_type == GHOST_TYPE_CUDA) {
        struct cudaDeviceProp devProp;
        CUDA_CALL_GOTO(cudaGetDeviceProperties(&devProp,cu_device),err,ret);
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
                recvcounts,displs,MPI_CHAR,0,ghost_cuda_comm));
#else
    strncpy(names,name,ghost_cu_MAX_DEVICE_NAME_LEN);
#endif

    if (me==0) {
        qsort(names,size,ghost_cu_MAX_DEVICE_NAME_LEN*sizeof(char),stringcmp);
        for (i=1; i<size; i++) {
            if (strcmp(names+(i-1)*ghost_cu_MAX_DEVICE_NAME_LEN,
                        names+i*ghost_cu_MAX_DEVICE_NAME_LEN)) {
                (*devInfo)->ndistinctdevice++;
            }
        }
    }

#ifdef GHOST_HAVE_MPI
    MPI_CALL_GOTO(MPI_Bcast(&((*devInfo)->ndistinctdevice),1,MPI_INT,0,ghost_cuda_comm),err,ret);
#endif

    GHOST_CALL_GOTO(ghost_malloc((void **)&(*devInfo)->ndevice,sizeof(int)*(*devInfo)->ndistinctdevice),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&(*devInfo)->names,sizeof(char *)*(*devInfo)->ndistinctdevice),err,ret);
    for (i=0; i<(*devInfo)->ndistinctdevice; i++) {
        GHOST_CALL_GOTO(ghost_malloc((void **)&(*devInfo)->names[i],sizeof(char)*ghost_cu_MAX_DEVICE_NAME_LEN),err,ret);
        (*devInfo)->ndevice[i] = 1;
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
                (*devInfo)->ndevice[distIdx-1]++;
            }
        }
        free(names);
    }

#ifdef GHOST_HAVE_MPI
    MPI_CALL_GOTO(MPI_Bcast((*devInfo)->ndevice,(*devInfo)->ndistinctdevice,MPI_INT,0,ghost_cuda_comm),err,ret);

    for (i=0; i<(*devInfo)->ndistinctdevice; i++) {
        MPI_CALL_GOTO(MPI_Bcast((*devInfo)->names[i],ghost_cu_MAX_DEVICE_NAME_LEN,MPI_CHAR,0,ghost_cuda_comm),err,ret);
    }
#endif


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
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_COMMUNICATION);
#else
    UNUSED(stringcmp);
    *devInfo = NULL;
#endif
    return ret;
}

ghost_error ghost_cu_malloc_mapped(void **mem, const size_t size)
{
#ifdef GHOST_HAVE_CUDA

    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    if (size/(1024.*1024.*1024.) > 1.) {
        DEBUG_LOG(1,"Allocating big array of size %f GB",size/(1024.*1024.*1024.));
    }

    CUDA_CALL_RETURN(cudaHostAlloc(mem,size,cudaHostAllocMapped));
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
#else
    UNUSED(mem);
    UNUSED(size);
#endif

    return GHOST_SUCCESS;
}

ghost_error ghost_cu_malloc_pinned(void **mem, const size_t size)
{
#ifdef GHOST_HAVE_CUDA

    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    if (size/(1024.*1024.*1024.) > 1.) {
        DEBUG_LOG(1,"Allocating big array of size %f GB",size/(1024.*1024.*1024.));
    }

    CUDA_CALL_RETURN(cudaHostAlloc(mem,size,cudaHostAllocDefault));
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
#else
    UNUSED(mem);
    UNUSED(size);
#endif

    return GHOST_SUCCESS;
}

ghost_error ghost_cu_device(int *device)
{
#ifdef GHOST_HAVE_CUDA
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    if (cu_device < 0) {
        ERROR_LOG("CUDA not initialized!");
        return GHOST_ERR_CUDA;
    }
    *device = cu_device;
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
#else
    UNUSED(device);
#endif

    return GHOST_SUCCESS;
}

ghost_error ghost_cu_cublas_handle(ghost_cublas_handle_t *handle)
{
#ifdef GHOST_HAVE_CUDA
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    if (!ghost_cublas_handle) {
        ERROR_LOG("CUBLAS not initialized!");
        return GHOST_ERR_CUBLAS;
    }
    *handle = ghost_cublas_handle;
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
#else
    UNUSED(handle);
#endif

    return GHOST_SUCCESS;
}

ghost_error ghost_cu_cusparse_handle(ghost_cusparse_handle_t *handle)
{
#ifdef GHOST_HAVE_CUDA
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    if (!ghost_cusparse_handle) {
        ERROR_LOG("CUSPARSE not initialized!");
        return GHOST_ERR_CUSPARSE;
    }
    *handle = ghost_cusparse_handle;
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
#else
    UNUSED(handle);
#endif

    return GHOST_SUCCESS;
}

ghost_error ghost_cu_version(int *ver)
{
#ifdef GHOST_HAVE_CUDA
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    CUDA_CALL_RETURN(cudaRuntimeGetVersion(ver));
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
#else
    UNUSED(ver);
#endif
    
    return GHOST_SUCCESS;
}
    
ghost_error ghost_cu_free_host(void * mem)
{
#ifdef GHOST_HAVE_CUDA
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    CUDA_CALL_RETURN(cudaFreeHost(mem));
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
#else
    UNUSED(mem);
#endif
    return GHOST_SUCCESS;
}

ghost_error ghost_cu_deviceprop_get(ghost_cu_deviceprop *prop)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
#ifdef GHOST_HAVE_CUDA
    *prop = ghost_cu_device_prop;
#else
    *prop = -1;
#endif

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}

ghost_error ghost_cu_rand_generator_get(ghost_cu_rand_generator *gen)
{
#ifdef GHOST_HAVE_CUDA
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    if (cu_rand_generator == NULL) {
        CURAND_CALL_RETURN(curandCreateGenerator(&cu_rand_generator,CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CALL_RETURN(curandSetPseudoRandomGeneratorSeed(cu_rand_generator,ghost_rand_cu_seed_get()));
    }

    *gen = cu_rand_generator;
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
#else
    UNUSED(gen);
#endif

    return GHOST_SUCCESS;
}

ghost_error ghost_cu_finalize()
{
#ifdef GHOST_HAVE_CUDA
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_TEARDOWN);
    if (ghost_cublas_handle) {
        CUBLAS_CALL_RETURN(cublasDestroy(ghost_cublas_handle));
    }
    if (ghost_cusparse_handle) {
        CUSPARSE_CALL_RETURN(cusparseDestroy(ghost_cusparse_handle));
    }
    if (cu_rand_generator) {
        curandDestroyGenerator(cu_rand_generator);
    }
    cudaDeviceReset();
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_TEARDOWN);
#endif

    return GHOST_SUCCESS;
}

ghost_error ghost_cu_memtranspose(int m, int n, void *to, int ldto, const void *from, int ldfrom, ghost_datatype dt) 
{
#ifdef GHOST_HAVE_CUDA
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    if (dt & GHOST_DT_COMPLEX) {
        if (dt & GHOST_DT_DOUBLE) {
            const cuDoubleComplex alpha = make_cuDoubleComplex(1.,0.), beta = make_cuDoubleComplex(0.,0.);
            CUBLAS_CALL_RETURN(cublasZgeam(ghost_cublas_handle,CUBLAS_OP_T,CUBLAS_OP_T,m,n,&alpha,(const cuDoubleComplex *)from,ldfrom,&beta,(const cuDoubleComplex *)from,ldfrom,(cuDoubleComplex *)to,ldto));
        } else {
            const cuFloatComplex alpha = make_cuFloatComplex(1.,0.), beta = make_cuFloatComplex(0.,0.);
            CUBLAS_CALL_RETURN(cublasCgeam(ghost_cublas_handle,CUBLAS_OP_T,CUBLAS_OP_T,m,n,&alpha,(const cuFloatComplex *)from,ldfrom,&beta,(const cuFloatComplex *)from,ldfrom,(cuFloatComplex *)to,ldto));
        }
    } else {
        if (dt & GHOST_DT_DOUBLE) {
            const double alpha = 1., beta = 0.;
            CUBLAS_CALL_RETURN(cublasDgeam(ghost_cublas_handle,CUBLAS_OP_T,CUBLAS_OP_T,m,n,&alpha,(const double *)from,ldfrom,&beta,(const double *)from,ldfrom,(double *)to,ldto));
        } else {
            const float alpha = 1., beta = 0.;
            CUBLAS_CALL_RETURN(cublasSgeam(ghost_cublas_handle,CUBLAS_OP_T,CUBLAS_OP_T,m,n,&alpha,(const float *)from,ldfrom,&beta,(const float *)from,ldfrom,(float *)to,ldto));
        }
    }
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
#else
    UNUSED(m);
    UNUSED(n);
    UNUSED(to);
    UNUSED(ldto);
    UNUSED(from);
    UNUSED(ldfrom);
    UNUSED(dt);
#endif
    return GHOST_SUCCESS;
}

