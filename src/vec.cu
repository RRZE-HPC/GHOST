#include "ghost/config.h"
#undef GHOST_HAVE_MPI
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/constants.h"
#include "ghost/vec.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <curand.h>
#include <sys/types.h>
#include <unistd.h>


#include "ghost/cu_complex_helper.h"


#define THREADSPERBLOCK 256

extern cublasHandle_t ghost_cublas_handle;

template<typename T>  
__global__ static void cu_vaxpy_kernel(T *v1, T *v2, T *a, ghost_vidx_t nrows, ghost_vidx_t nvecs, ghost_vidx_t nrowspadded)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    for (;idx < nrows; idx+=gridDim.x*blockDim.x)
    {
        ghost_vidx_t v;
        for (v=0; v<nvecs; v++) {
            v1[v*nrowspadded+idx] = axpy<T,T>(v1[v*nrowspadded+idx],v2[v*nrowspadded+idx],a[v]);
        }
    }
}

template<typename T>  
__global__ static void cu_vaxpby_kernel(T *v1, T *v2, T *a, T *b, ghost_vidx_t nrows, ghost_vidx_t nvecs, ghost_vidx_t nrowspadded)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    for (;idx < nrows; idx+=gridDim.x*blockDim.x)
    {
        ghost_vidx_t v;
        for (v=0; v<nvecs; v++) {
            v1[v*nrowspadded+idx] = axpby<T>(v2[v*nrowspadded+idx],v1[v*nrowspadded+idx],a[v],b[v]);
        }
    }
}

template<typename T>  
__global__ static void cu_axpby_kernel(T *v1, T *v2, T a, T b, ghost_vidx_t nrows, ghost_vidx_t nvecs, ghost_vidx_t nrowspadded)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    for (;idx < nrows; idx+=gridDim.x*blockDim.x)
    {
        ghost_vidx_t v;
        for (v=0; v<nvecs; v++) {
            v1[v*nrowspadded+idx] = axpby<T>(v2[v*nrowspadded+idx],v1[v*nrowspadded+idx],a,b);
        }
    }
}

template<typename T>  
__global__ static void cu_vscale_kernel(T *vec, T *a, ghost_vidx_t nrows, ghost_vidx_t nvecs, ghost_vidx_t nrowspadded)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    for (;idx < nrows; idx+=gridDim.x*blockDim.x)
    {
        ghost_vidx_t v;
        for (v=0; v<nvecs; v++) {
            vec[v*nrowspadded+idx] = scale<T>(a[v],vec[v*nrowspadded+idx]);
        }
    }
}

template<typename T>  
__global__ static void cu_fromscalar_kernel(T *vec, T a, ghost_vidx_t nrows, ghost_vidx_t nvecs, ghost_vidx_t nrowspadded)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    for (;idx < nrows; idx+=gridDim.x*blockDim.x)
    {
        ghost_vidx_t v;
        for (v=0; v<nvecs; v++) {
            vec[v*nrowspadded+idx] = a;
        }
    }
}

extern "C" void ghost_vec_cu_vaxpy(ghost_vec_t *v1, ghost_vec_t *v2, void *a)
{
    void *d_a = CU_allocDeviceMemory(v1->traits->nvecs*ghost_sizeofDataType(v1->traits->datatype));
    CU_copyHostToDevice(d_a,a,v1->traits->nvecs*ghost_sizeofDataType(v1->traits->datatype));
    if (v1->traits->datatype != v2->traits->datatype)
    {
        WARNING_LOG("Cannot VAXPY vectors with different data types");
        return;
    }

    if (v1->traits->datatype & GHOST_BINCRS_DT_COMPLEX)
    {
        if (v1->traits->datatype & GHOST_BINCRS_DT_DOUBLE)
        {
            cu_vaxpy_kernel<cuDoubleComplex><<< (int)ceil((double)v1->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>((cuDoubleComplex *)v1->CU_val, (cuDoubleComplex *)v2->CU_val,(cuDoubleComplex *)d_a,v1->traits->nrows,v1->traits->nvecs,v1->traits->nrowspadded);
        } 
        else 
        {
            cu_vaxpy_kernel<cuFloatComplex><<< (int)ceil((double)v1->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>((cuFloatComplex *)v1->CU_val, (cuFloatComplex *)v2->CU_val,(cuFloatComplex *)d_a,v1->traits->nrows,v1->traits->nvecs,v1->traits->nrowspadded);
        }
    }
    else
    {
        if (v1->traits->datatype & GHOST_BINCRS_DT_DOUBLE)
        {
            cu_vaxpy_kernel<double><<< (int)ceil((double)v1->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>((double *)v1->CU_val, (double *)v2->CU_val,(double *)d_a,v1->traits->nrows,v1->traits->nvecs,v1->traits->nrowspadded);
        } 
        else 
        {
            cu_vaxpy_kernel<float><<< (int)ceil((double)v1->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>((float *)v1->CU_val, (float *)v2->CU_val,(float *)d_a,v1->traits->nrows,v1->traits->nvecs,v1->traits->nrowspadded);
        }
    }
}
    
extern "C" void ghost_vec_cu_vaxpby(ghost_vec_t *v1, ghost_vec_t *v2, void *a, void *b)
{
    void *d_a = CU_allocDeviceMemory(v1->traits->nvecs*ghost_sizeofDataType(v1->traits->datatype));
    void *d_b = CU_allocDeviceMemory(v1->traits->nvecs*ghost_sizeofDataType(v1->traits->datatype));
    CU_copyHostToDevice(d_a,a,v1->traits->nvecs*ghost_sizeofDataType(v1->traits->datatype));
    CU_copyHostToDevice(d_b,b,v1->traits->nvecs*ghost_sizeofDataType(v1->traits->datatype));
    if (v1->traits->datatype != v2->traits->datatype)
    {
        WARNING_LOG("Cannot VAXPBY vectors with different data types");
        return;
    }
    if (v1->traits->datatype & GHOST_BINCRS_DT_COMPLEX)
    {
        if (v1->traits->datatype & GHOST_BINCRS_DT_DOUBLE)
        {
            cu_vaxpby_kernel<cuDoubleComplex><<< (int)ceil((double)v1->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                (cuDoubleComplex *)v1->CU_val, (cuDoubleComplex *)v2->CU_val,(cuDoubleComplex *)d_a,(cuDoubleComplex *)d_b,
                 v1->traits->nrows,v1->traits->nvecs,v1->traits->nrowspadded);
        } 
        else 
        {
            cu_vaxpby_kernel<cuFloatComplex><<< (int)ceil((double)v1->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                (cuFloatComplex *)v1->CU_val, (cuFloatComplex *)v2->CU_val,(cuFloatComplex *)d_a,(cuFloatComplex *)d_b,
                 v1->traits->nrows,v1->traits->nvecs,v1->traits->nrowspadded);
        }
    }
    else
    {
        if (v1->traits->datatype & GHOST_BINCRS_DT_DOUBLE)
        {
            cu_vaxpby_kernel<double><<< (int)ceil((double)v1->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (double *)v1->CU_val, (double *)v2->CU_val,(double *)d_a,(double *)d_b,
                    v1->traits->nrows,v1->traits->nvecs,v1->traits->nrowspadded);
        } 
        else 
        {
            cu_vaxpby_kernel<float><<< (int)ceil((double)v1->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                (float *)v1->CU_val, (float *)v2->CU_val,(float *)d_a,(float *)d_b,
                 v1->traits->nrows,v1->traits->nvecs,v1->traits->nrowspadded);
        }
    }
}

extern "C" void ghost_vec_cu_dotprod(ghost_vec_t *vec, ghost_vec_t *vec2, void *res)
{
    if (vec->traits->datatype != vec2->traits->datatype)
    {
        WARNING_LOG("Cannot DOT vectors with different data types");
        return;
    }
    
    ghost_vidx_t v;
    for (v=0; v<vec->traits->nvecs; v++)
    {
        char *v1 = &vec->CU_val[v*vec->traits->nrowspadded*ghost_sizeofDataType(vec->traits->datatype)];
        char *v2 = &vec2->CU_val[v*vec->traits->nrowspadded*ghost_sizeofDataType(vec->traits->datatype)];
        if (vec->traits->datatype & GHOST_BINCRS_DT_COMPLEX)
        {
            if (vec->traits->datatype & GHOST_BINCRS_DT_DOUBLE)
            {
                CUBLAS_safecall(cublasZdotc(ghost_cublas_handle,vec->traits->nrows,
                            (const cuDoubleComplex *)v1,1,(const cuDoubleComplex *)v2,1,&((cuDoubleComplex *)res)[v]));
            } 
            else 
            {
                CUBLAS_safecall(cublasCdotc(ghost_cublas_handle,vec->traits->nrows,
                            (const cuFloatComplex *)v1,1,(const cuFloatComplex *)v2,1,&((cuFloatComplex *)res)[v]));
            }
        }
        else
        {
            if (vec->traits->datatype & GHOST_BINCRS_DT_DOUBLE)
            {
                CUBLAS_safecall(cublasDdot(ghost_cublas_handle,vec->traits->nrows,
                            (const double *)v1,1,(const double *)v2,1,&((double *)res)[v]));
            } 
            else 
            {
                CUBLAS_safecall(cublasSdot(ghost_cublas_handle,vec->traits->nrows,
                            (const float *)v1,1,(const float *)v2,1,&((float *)res)[v]));
            }
        }
    }
}

extern "C" void ghost_vec_cu_axpy(ghost_vec_t *vec, ghost_vec_t *vec2, void *a)
{
    if (vec->traits->datatype != vec2->traits->datatype)
    {
        WARNING_LOG("Cannot AXPY vectors with different data types");
        return;
    }
    if (vec->traits->datatype & GHOST_BINCRS_DT_COMPLEX)
    {
        if (vec->traits->datatype & GHOST_BINCRS_DT_DOUBLE)
        {
            CUBLAS_safecall(cublasZaxpy(ghost_cublas_handle,vec->traits->nrows,
                        (const cuDoubleComplex *)a,
                        (const cuDoubleComplex *)vec2->CU_val,1,
                        (cuDoubleComplex *)vec->CU_val,1));
        } 
        else 
        {
            CUBLAS_safecall(cublasCaxpy(ghost_cublas_handle,vec->traits->nrows,
                        (const cuFloatComplex *)a,
                        (const cuFloatComplex *)vec2->CU_val,1,
                        (cuFloatComplex *)vec->CU_val,1));
        }
    }
    else
    {
        if (vec->traits->datatype & GHOST_BINCRS_DT_DOUBLE)
        {
            CUBLAS_safecall(cublasDaxpy(ghost_cublas_handle,vec->traits->nrows,
                        (const double *)a,
                        (const double *)vec2->CU_val,1,
                        (double *)vec->CU_val,1));
        } 
        else 
        {
            CUBLAS_safecall(cublasSaxpy(ghost_cublas_handle,vec->traits->nrows,
                        (const float *)a,
                        (const float *)vec2->CU_val,1,
                        (float *)vec->CU_val,1));
        }
    }
}

extern "C" void ghost_vec_cu_axpby(ghost_vec_t *v1, ghost_vec_t *v2, void *a, void *b)
{
    if (v1->traits->datatype != v2->traits->datatype)
    {
        WARNING_LOG("Cannot AXPY vectors with different data types");
        return;
    }
    if (v1->traits->datatype & GHOST_BINCRS_DT_COMPLEX)
    {
        if (v1->traits->datatype & GHOST_BINCRS_DT_DOUBLE)
        {
            cu_axpby_kernel<cuDoubleComplex><<< (int)ceil((double)v1->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>
                ((cuDoubleComplex *)v1->CU_val, (cuDoubleComplex *)v2->CU_val,*((cuDoubleComplex *)a),*((cuDoubleComplex *)b),
                 v1->traits->nrows,v1->traits->nvecs,v1->traits->nrowspadded);
        } 
        else 
        {
            cu_axpby_kernel<cuFloatComplex><<< (int)ceil((double)v1->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>
                ((cuFloatComplex *)v1->CU_val, (cuFloatComplex *)v2->CU_val,*((cuFloatComplex *)a),*((cuFloatComplex *)b),
                 v1->traits->nrows,v1->traits->nvecs,v1->traits->nrowspadded);
        }
    }
    else
    {
        if (v1->traits->datatype & GHOST_BINCRS_DT_DOUBLE)
        {
            cu_axpby_kernel<double><<< (int)ceil((double)v1->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>
                ((double *)v1->CU_val, (double *)v2->CU_val,*((double *)a),*((double *)b),
                 v1->traits->nrows,v1->traits->nvecs,v1->traits->nrowspadded);
        } 
        else 
        {
            cu_axpby_kernel<float><<< (int)ceil((double)v1->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>
                ((float *)v1->CU_val, (float *)v2->CU_val,*((float *)a),*((float *)b),
                 v1->traits->nrows,v1->traits->nvecs,v1->traits->nrowspadded);
        }
    }
}

extern "C" void ghost_vec_cu_scale(ghost_vec_t *vec, void *a)
{
    if (vec->traits->datatype & GHOST_BINCRS_DT_COMPLEX)
    {
        if (vec->traits->datatype & GHOST_BINCRS_DT_DOUBLE)
        {
            CUBLAS_safecall(cublasZscal(ghost_cublas_handle,vec->traits->nrows,
                        (const cuDoubleComplex *)a,
                        (cuDoubleComplex *)vec->CU_val,1));
        } 
        else 
        {
            CUBLAS_safecall(cublasCscal(ghost_cublas_handle,vec->traits->nrows,
                        (const cuFloatComplex *)a,
                        (cuFloatComplex *)vec->CU_val,1));
        }
    }
    else
    {
        if (vec->traits->datatype & GHOST_BINCRS_DT_DOUBLE)
        {
            CUBLAS_safecall(cublasDscal(ghost_cublas_handle,vec->traits->nrows,
                        (const double *)a,
                        (double *)vec->CU_val,1));
        } 
        else 
        {
            CUBLAS_safecall(cublasSscal(ghost_cublas_handle,vec->traits->nrows,
                        (const float *)a,
                        (float *)vec->CU_val,1));
        }
    }
}

extern "C" void ghost_vec_cu_vscale(ghost_vec_t *vec, void *a)
{
    void *d_a = CU_allocDeviceMemory(vec->traits->nvecs*ghost_sizeofDataType(vec->traits->datatype));
    CU_copyHostToDevice(d_a,a,vec->traits->nvecs*ghost_sizeofDataType(vec->traits->datatype));
    if (vec->traits->datatype & GHOST_BINCRS_DT_COMPLEX)
    {
        if (vec->traits->datatype & GHOST_BINCRS_DT_DOUBLE)
        {
            cu_vscale_kernel<cuDoubleComplex><<< (int)ceil((double)vec->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (cuDoubleComplex *)vec->CU_val, (cuDoubleComplex *)d_a,
                    vec->traits->nrows,vec->traits->nvecs,vec->traits->nrowspadded);
        } 
        else 
        {
            cu_vscale_kernel<cuFloatComplex><<< (int)ceil((double)vec->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (cuFloatComplex *)vec->CU_val, (cuFloatComplex *)d_a,
                    vec->traits->nrows,vec->traits->nvecs,vec->traits->nrowspadded);
        }
    }
    else
    {
        if (vec->traits->datatype & GHOST_BINCRS_DT_DOUBLE)
        {
            cu_vscale_kernel<double><<< (int)ceil((double)vec->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (double *)vec->CU_val, (double *)d_a,
                    vec->traits->nrows,vec->traits->nvecs,vec->traits->nrowspadded);
        } 
        else 
        {
            cu_vscale_kernel<float><<< (int)ceil((double)vec->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (float *)vec->CU_val, (float *)d_a,
                    vec->traits->nrows,vec->traits->nvecs,vec->traits->nrowspadded);
        }
    }
}

extern "C" void ghost_vec_cu_fromScalar(ghost_vec_t *vec, void *a)
{
    ghost_vec_malloc(vec);
    if (vec->traits->datatype & GHOST_BINCRS_DT_COMPLEX)
    {
        if (vec->traits->datatype & GHOST_BINCRS_DT_DOUBLE)
        {
            cu_fromscalar_kernel<cuDoubleComplex><<< (int)ceil((double)vec->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (cuDoubleComplex *)vec->CU_val, *(cuDoubleComplex *)a,
                    vec->traits->nrows,vec->traits->nvecs,vec->traits->nrowspadded);
        } 
        else 
        {
            cu_fromscalar_kernel<cuFloatComplex><<< (int)ceil((double)vec->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (cuFloatComplex *)vec->CU_val, *(cuFloatComplex *)a,
                    vec->traits->nrows,vec->traits->nvecs,vec->traits->nrowspadded);
        }
    }
    else
    {
        if (vec->traits->datatype & GHOST_BINCRS_DT_DOUBLE)
        {
            cu_fromscalar_kernel<double><<< (int)ceil((double)vec->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (double *)vec->CU_val, *(double *)a,
                    vec->traits->nrows,vec->traits->nvecs,vec->traits->nrowspadded);
        } 
        else 
        {
            cu_fromscalar_kernel<float><<< (int)ceil((double)vec->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (float *)vec->CU_val, *(float *)a,
                    vec->traits->nrows,vec->traits->nvecs,vec->traits->nrowspadded);
        }
    }
}

void ghost_vec_cu_fromRand(ghost_vec_t *vec)
{
    long pid = getpid();
    ghost_vec_malloc(vec);
    curandGenerator_t gen;
    CURAND_safecall(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_safecall(curandSetPseudoRandomGeneratorSeed(gen,ghost_hash(int(ghost_wctimemilli()),clock(),ghost_ompGetThreadNum())));

    ghost_vidx_t v;
    for (v=0; v<vec->traits->nvecs; v++)
    {
        if (vec->traits->datatype & GHOST_BINCRS_DT_COMPLEX)
        {
            if (vec->traits->datatype & GHOST_BINCRS_DT_DOUBLE)
            {
                CURAND_safecall(curandGenerateUniformDouble(gen,
                            &((double *)(vec->CU_val))[v*vec->traits->nrowspadded],
                            vec->traits->nrows*2));
            } 
            else 
            {
                CURAND_safecall(curandGenerateUniform(gen,
                            &((float *)(vec->CU_val))[v*vec->traits->nrowspadded],
                            vec->traits->nrows*2));
            }
        }
        else
        {
            if (vec->traits->datatype & GHOST_BINCRS_DT_DOUBLE)
            {
                CURAND_safecall(curandGenerateUniformDouble(gen,
                            &((double *)(vec->CU_val))[v*vec->traits->nrowspadded],
                            vec->traits->nrows));
            } 
            else 
            {
                CURAND_safecall(curandGenerateUniform(gen,
                            &((float *)(vec->CU_val))[v*vec->traits->nrowspadded],
                            vec->traits->nrows));
            }
        }
    }
}
