#include <ghost_config.h>
#undef GHOST_HAVE_MPI
#include <ghost_types.h>
#include <ghost_util.h>
#include <ghost_constants.h>
#include <ghost_vec.h>

#include <cuda_runtime.h>
#include <stdio.h>
#include <cublas_v2.h>

#include <ghost_cu_complex_helper.h>


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
            v1[v*nrowspadded+idx] = axpy<T,T>(v1[v*nrowspadded],v2[v*nrowspadded+idx],a[v]);
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
    if (v1->traits->datatype != v2->traits->datatype)
    {
        WARNING_LOG("Cannot VAXPY vectors with different data types");
        return;
    }

    if (v1->traits->datatype & GHOST_BINCRS_DT_COMPLEX)
    {
        if (v1->traits->datatype & GHOST_BINCRS_DT_DOUBLE)
        {
            cu_vaxpy_kernel<cuDoubleComplex><<< (int)ceil((double)v1->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>((cuDoubleComplex *)v1->CU_val, (cuDoubleComplex *)v2->CU_val,(cuDoubleComplex *)a,v1->traits->nrows,v1->traits->nvecs,v1->traits->nrowspadded);
        } 
        else 
        {
            cu_vaxpy_kernel<cuFloatComplex><<< (int)ceil((double)v1->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>((cuFloatComplex *)v1->CU_val, (cuFloatComplex *)v2->CU_val,(cuFloatComplex *)a,v1->traits->nrows,v1->traits->nvecs,v1->traits->nrowspadded);
        }
    }
    else
    {
        if (v1->traits->datatype & GHOST_BINCRS_DT_DOUBLE)
        {
            cu_vaxpy_kernel<double><<< (int)ceil((double)v1->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>((double *)v1->CU_val, (double *)v2->CU_val,(double *)a,v1->traits->nrows,v1->traits->nvecs,v1->traits->nrowspadded);
        } 
        else 
        {
            cu_vaxpy_kernel<float><<< (int)ceil((double)v1->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>((float *)v1->CU_val, (float *)v2->CU_val,(float *)a,v1->traits->nrows,v1->traits->nvecs,v1->traits->nrowspadded);
        }
    }
}
    
extern "C" void ghost_vec_cu_vaxpby(ghost_vec_t *v1, ghost_vec_t *v2, void *a, void *b)
{
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
                (cuDoubleComplex *)v1->CU_val, (cuDoubleComplex *)v2->CU_val,(cuDoubleComplex *)a,(cuDoubleComplex *)b,
                 v1->traits->nrows,v1->traits->nvecs,v1->traits->nrowspadded);
        } 
        else 
        {
            cu_vaxpby_kernel<cuFloatComplex><<< (int)ceil((double)v1->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                (cuFloatComplex *)v1->CU_val, (cuFloatComplex *)v2->CU_val,(cuFloatComplex *)a,(cuFloatComplex *)b,
                 v1->traits->nrows,v1->traits->nvecs,v1->traits->nrowspadded);
        }
    }
    else
    {
        if (v1->traits->datatype & GHOST_BINCRS_DT_DOUBLE)
        {
            cu_vaxpby_kernel<double><<< (int)ceil((double)v1->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (double *)v1->CU_val, (double *)v2->CU_val,(double *)a,(double *)b,
                    v1->traits->nrows,v1->traits->nvecs,v1->traits->nrowspadded);
        } 
        else 
        {
            cu_vaxpby_kernel<float><<< (int)ceil((double)v1->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                (float *)v1->CU_val, (float *)v2->CU_val,(float *)a,(float *)b,
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
    if (vec->traits->datatype & GHOST_BINCRS_DT_COMPLEX)
    {
        if (vec->traits->datatype & GHOST_BINCRS_DT_DOUBLE)
        {
            CUBLAS_safecall(cublasZdotc(ghost_cublas_handle,vec->traits->nrows,
                        (const cuDoubleComplex *)vec->CU_val,1,(const cuDoubleComplex *)vec2->CU_val,1,(cuDoubleComplex *)res));
        } 
        else 
        {
            CUBLAS_safecall(cublasCdotc(ghost_cublas_handle,vec->traits->nrows,
                        (const cuFloatComplex *)vec->CU_val,1,(const cuFloatComplex *)vec2->CU_val,1,(cuFloatComplex *)res));
        }
    }
    else
    {
        if (vec->traits->datatype & GHOST_BINCRS_DT_DOUBLE)
        {
            CUBLAS_safecall(cublasDdot(ghost_cublas_handle,vec->traits->nrows,
                        (const double *)vec->CU_val,1,(const double *)vec2->CU_val,1,(double *)res));
        } 
        else 
        {
            CUBLAS_safecall(cublasSdot(ghost_cublas_handle,vec->traits->nrows,
                        (const float *)vec->CU_val,1,(const float *)vec2->CU_val,1,(float *)res));
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
    if (vec->traits->datatype & GHOST_BINCRS_DT_COMPLEX)
    {
        if (vec->traits->datatype & GHOST_BINCRS_DT_DOUBLE)
        {
            cu_vscale_kernel<cuDoubleComplex><<< (int)ceil((double)vec->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (cuDoubleComplex *)vec->CU_val, (cuDoubleComplex *)a,
                    vec->traits->nrows,vec->traits->nvecs,vec->traits->nrowspadded);
        } 
        else 
        {
            cu_vscale_kernel<cuFloatComplex><<< (int)ceil((double)vec->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (cuFloatComplex *)vec->CU_val, (cuFloatComplex *)a,
                    vec->traits->nrows,vec->traits->nvecs,vec->traits->nrowspadded);
        }
    }
    else
    {
        if (vec->traits->datatype & GHOST_BINCRS_DT_DOUBLE)
        {
            cu_vscale_kernel<double><<< (int)ceil((double)vec->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (double *)vec->CU_val, (double *)a,
                    vec->traits->nrows,vec->traits->nvecs,vec->traits->nrowspadded);
        } 
        else 
        {
            cu_vscale_kernel<float><<< (int)ceil((double)vec->traits->nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (float *)vec->CU_val, (float *)a,
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
