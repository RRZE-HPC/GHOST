#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/densemat_cm.h"
#include "ghost/log.h"
#include "ghost/timing.h"
#include "ghost/locality.h"
#include "ghost/instr.h"
#include "ghost/rand.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <curand.h>
#include <sys/types.h>
#include <unistd.h>
#include <complex.h>

#include "ghost/cu_complex.h"


#define THREADSPERBLOCK 256


template<typename T>  
__global__ static void cu_vaxpby_kernel(T *v1, T *v2, T *a, T *b, ghost_lidx_t nrows, ghost_lidx_t ncols, ghost_lidx_t ld1, ghost_lidx_t ld2)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    for (;idx < nrows; idx+=gridDim.x*blockDim.x) {
        ghost_lidx_t v;
        for (v=0; v<ncols; v++) {
            v1[v*ld1+idx] = axpby<T>(v2[v*ld2+idx],v1[v*ld1+idx],a[v],b[v]);
        }
    }
}

template<typename T>  
__global__ static void cu_axpby_kernel(T *v1, T *v2, T a, T b, ghost_lidx_t nrows, ghost_lidx_t ncols, ghost_lidx_t ld1, ghost_lidx_t ld2) 
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    for (;idx < nrows; idx+=gridDim.x*blockDim.x) {
        ghost_lidx_t v;
        for (v=0; v<ncols; v++) {
            v1[v*ld1+idx] = axpby<T>(v2[v*ld2+idx],v1[v*ld1+idx],a,b);
        }
    }
}

template<typename T>  
__global__ static void cu_axpbypcz_kernel(T *v1, T *v2, T *v3, T a, T b, T c, ghost_lidx_t nrows, ghost_lidx_t ncols, ghost_lidx_t ld1, ghost_lidx_t ld2, ghost_lidx_t ld3) 
{
    T o;
    one(o);
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    for (;idx < nrows; idx+=gridDim.x*blockDim.x) {
        ghost_lidx_t v;
        for (v=0; v<ncols; v++) {
            v1[v*ld1+idx] = axpby<T>(axpby<T>(v2[v*ld2+idx],v1[v*ld1+idx],a,b),v3[v*ld3+idx],o,c);
        }
    }
}

template<typename T>  
__global__ static void cu_vaxpbypcz_kernel(T *v1, T *v2, T *v3, T *a, T *b, T *c, ghost_lidx_t nrows, ghost_lidx_t ncols, ghost_lidx_t ld1, ghost_lidx_t ld2, ghost_lidx_t ld3) 
{
    T o;
    one(o);
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    for (;idx < nrows; idx+=gridDim.x*blockDim.x) {
        ghost_lidx_t v;
        for (v=0; v<ncols; v++) {
            v1[v*ld1+idx] = axpby<T>(axpby<T>(v2[v*ld2+idx],v1[v*ld1+idx],a[v],b[v]),v3[v*ld3+idx],o,c[v]);
        }
    }
}

template<typename T>  
__global__ static void cu_scale_kernel(T *vec, T a, ghost_lidx_t nrows, ghost_lidx_t ncols, ghost_lidx_t ld)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    for (;idx < nrows; idx+=gridDim.x*blockDim.x) {
        ghost_lidx_t v;
        for (v=0; v<ncols; v++) {
            vec[v*ld+idx] = scale<T>(a,vec[v*ld+idx]);
        }
    }

}

template<typename T>  
__global__ static void cu_conj_kernel(T *vec, ghost_lidx_t nrows, ghost_lidx_t ncols, ghost_lidx_t ld)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    for (;idx < nrows; idx+=gridDim.x*blockDim.x) {
        ghost_lidx_t v;
        for (v=0; v<ncols; v++) {
            vec[v*ld+idx] = conj(vec[v*ld+idx]);
        }
    }

}

template<typename T>  
__global__ static void cu_vscale_kernel(T *vec, T *a, ghost_lidx_t nrows, ghost_lidx_t ncols, ghost_lidx_t ld)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    for (;idx < nrows; idx+=gridDim.x*blockDim.x) {
        ghost_lidx_t v;
        for (v=0; v<ncols; v++) {
            vec[v*ld+idx] = scale<T>(a[v],vec[v*ld+idx]);
        }
    }
}

template<typename T>  
__global__ static void cu_fromscalar_kernel(T *vec, T a, ghost_lidx_t nrows, ghost_lidx_t ncols, ghost_lidx_t ld)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    for (;idx < nrows; idx+=gridDim.x*blockDim.x) {
        ghost_lidx_t v;
        for (v=0; v<ncols; v++) {
            vec[v*ld+idx] = a;
        }
    }
}

template<typename T>  
__global__ static void cu_communicationassembly_kernel(T *vec, T *work, ghost_lidx_t offs, ghost_lidx_t *duelist, ghost_lidx_t ncols, ghost_lidx_t ndues, ghost_lidx_t nrowspadded, ghost_lidx_t *perm)
{
    int due = blockIdx.x*blockDim.x+threadIdx.x;
    int col = threadIdx.y;

    if (perm) {
        for (;due < ndues; due+=gridDim.x*blockDim.x) {
            work[(offs+due)*ncols+col] = vec[col*nrowspadded+perm[duelist[due]]];
        }
    } else {
        for (;due < ndues; due+=gridDim.x*blockDim.x) {
            work[(offs+due)*ncols+col] = vec[col*nrowspadded+duelist[due]];
        }
    }
}

extern "C" ghost_error_t ghost_densemat_cm_cu_communicationassembly(void * work, ghost_lidx_t *dueptr, ghost_densemat_t *vec, ghost_lidx_t *perm)
{
  
    if (!vec->context->cu_duelist) {
       ERROR_LOG("cu_duelist must not be NULL!");
       return GHOST_ERR_INVALID_ARG;
    }
    if (!dueptr) {
       ERROR_LOG("dueptr must not be NULL!");
       return GHOST_ERR_INVALID_ARG;
    }


    int nrank, proc;
    dim3 block((int)ceil((double)THREADSPERBLOCK/vec->traits.ncols),vec->traits.ncols,1);
    ghost_context_t *ctx = vec->context;
    
    ghost_nrank(&nrank,ctx->mpicomm); 
            
    for (proc=0 ; proc<nrank ; proc++){
        if (vec->traits.datatype & GHOST_DT_COMPLEX)
        {
            if (vec->traits.datatype & GHOST_DT_DOUBLE)
            {
                if (ctx->dues[proc]) {
                    cu_communicationassembly_kernel<cuDoubleComplex><<< (int)ceil((double)ctx->dues[proc]/THREADSPERBLOCK)*vec->traits.ncols,block >>>((cuDoubleComplex *)vec->cu_val, ((cuDoubleComplex *)work),dueptr[proc],ctx->cu_duelist[proc],vec->traits.ncols,ctx->dues[proc],vec->stride,perm);
                }
            } 
            else 
            {
                if (ctx->dues[proc]) {
                    cu_communicationassembly_kernel<cuFloatComplex><<< (int)ceil((double)ctx->dues[proc]/THREADSPERBLOCK)*vec->traits.ncols,block >>>((cuFloatComplex *)vec->cu_val, ((cuFloatComplex *)work),dueptr[proc],ctx->cu_duelist[proc],vec->traits.ncols,ctx->dues[proc],vec->stride,perm);
                }
            }
        }
        else
        {
            if (vec->traits.datatype & GHOST_DT_DOUBLE)
            {
                if (ctx->dues[proc]) {
                    cu_communicationassembly_kernel<double><<< (int)ceil((double)ctx->dues[proc]/THREADSPERBLOCK)*vec->traits.ncols,block >>>((double *)vec->cu_val, ((double *)work),dueptr[proc],ctx->cu_duelist[proc],vec->traits.ncols,ctx->dues[proc],vec->stride,perm);
                }
            } 
            else 
            {
                if (ctx->dues[proc]) {
                    cu_communicationassembly_kernel<float><<< (int)ceil((double)ctx->dues[proc]/THREADSPERBLOCK)*vec->traits.ncols,block >>>((float *)vec->cu_val, ((float *)work),dueptr[proc],ctx->cu_duelist[proc],vec->traits.ncols,ctx->dues[proc],vec->stride,perm);
                }
            }
        }
    }
    cudaDeviceSynchronize();

    if (cudaPeekAtLastError() != cudaSuccess) {
        ERROR_LOG("Error in kernel");
        return GHOST_ERR_CUDA;
    }

    return GHOST_SUCCESS;

}


extern "C" ghost_error_t ghost_densemat_cm_cu_vaxpy(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret = GHOST_SUCCESS; 
    
    if (v1->traits.datatype & GHOST_DT_COMPLEX)
    {
        if (v1->traits.datatype & GHOST_DT_DOUBLE)
        {
            complex double *one;
            GHOST_CALL_RETURN(ghost_malloc((void **)&one,v1->traits.ncols*sizeof(complex double)));
            int v;
            for (v=0; v<v1->traits.ncols; v++) {
                one[v] = 1.+I*0.;
            }
            ret =  ghost_densemat_cm_cu_vaxpby(v1,v2,a,one);
        } 
        else 
        {
            complex float *one;
            GHOST_CALL_RETURN(ghost_malloc((void **)&one,v1->traits.ncols*sizeof(complex float)));
            int v;
            for (v=0; v<v1->traits.ncols; v++) {
                one[v] = 1.+I*0.;
            }
            ret =  ghost_densemat_cm_cu_vaxpby(v1,v2,a,one);
        }
    }
    else
    {
        if (v1->traits.datatype & GHOST_DT_DOUBLE)
        {
            double *one;
            GHOST_CALL_RETURN(ghost_malloc((void **)&one,v1->traits.ncols*sizeof(double)));
            int v;
            for (v=0; v<v1->traits.ncols; v++) {
                one[v] = 1.;
            }
            ret =  ghost_densemat_cm_cu_vaxpby(v1,v2,a,one);
        } 
        else 
        {
            float *one;
            GHOST_CALL_RETURN(ghost_malloc((void **)&one,v1->traits.ncols*sizeof(float)));
            int v;
            for (v=0; v<v1->traits.ncols; v++) {
                one[v] = 1.;
            }
            ret =  ghost_densemat_cm_cu_vaxpby(v1,v2,a,one);
        }
    }
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
}
    
extern "C" ghost_error_t ghost_densemat_cm_cu_vaxpby(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a, void *b)
{
    if (v1->traits.datatype != v2->traits.datatype)
    {
        ERROR_LOG("Cannot VAXPBY vectors with different data types");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
    
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret = GHOST_SUCCESS;

    void *d_a;
    void *d_b;
    size_t sizeofdt;

    ghost_datatype_size(&sizeofdt,v1->traits.datatype);
    
    GHOST_CALL_GOTO(ghost_cu_malloc(&d_a,v1->traits.ncols*sizeofdt),err,ret);
    GHOST_CALL_GOTO(ghost_cu_malloc(&d_b,v1->traits.ncols*sizeofdt),err,ret);
   
    ghost_lidx_t c; 
    for (c=0; c<v1->traits.ncols; c++) {
            GHOST_CALL_GOTO(ghost_cu_upload(&((char *)d_a)[c*sizeofdt],&((char *)a)[c*sizeofdt],sizeofdt),err,ret);
            GHOST_CALL_GOTO(ghost_cu_upload(&((char *)d_b)[c*sizeofdt],&((char *)b)[c*sizeofdt],sizeofdt),err,ret);
    }
    
    
    void *v1val, *v2val;
    ghost_densemat_t *v1compact, *v2compact;
    
    if (v1->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        INFO_LOG("Cloning (and compressing) v1 before operation");
        GHOST_CALL_GOTO(v1->clone(v1,&v1compact,v1->traits.nrows,0,v1->traits.ncols,0),err,ret);
    } else {
        v1compact = v1;
    }
    if (v2->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        INFO_LOG("Cloning (and compressing) v2 before operation");
        GHOST_CALL_GOTO(v2->clone(v2,&v2compact,v2->traits.nrows,0,v2->traits.ncols,0),err,ret);
    } else {
        v2compact = v2;
    }
    GHOST_CALL_GOTO(ghost_densemat_cu_valptr(v1compact,&v1val),err,ret);
    GHOST_CALL_GOTO(ghost_densemat_cu_valptr(v2compact,&v2val),err,ret);
    

    if (v1->traits.datatype & GHOST_DT_COMPLEX)
    {
        if (v1->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_vaxpby_kernel<cuDoubleComplex><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                (cuDoubleComplex *)v1val, (cuDoubleComplex *)v2val,(cuDoubleComplex *)d_a,(cuDoubleComplex *)d_b,
                 v1->traits.nrows,v1->traits.ncols,v1->stride,v2->stride);
        } 
        else 
        {
            cu_vaxpby_kernel<cuFloatComplex><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                (cuFloatComplex *)v1val, (cuFloatComplex *)v2val,(cuFloatComplex *)d_a,(cuFloatComplex *)d_b,
                 v1->traits.nrows,v1->traits.ncols,v1->stride,v2->stride);
        }
    }
    else
    {
        if (v1->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_vaxpby_kernel<double><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                (double *)v1val, (double *)v2val,(double *)d_a,(double *)d_b,
                 v1->traits.nrows,v1->traits.ncols,v1->stride,v2->stride);
        } 
        else 
        {
            cu_vaxpby_kernel<float><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                (float *)v1val, (float *)v2val,(float *)d_a,(float *)d_b,
                 v1->traits.nrows,v1->traits.ncols,v1->stride,v2->stride);
        }
    }
    if (v1compact != v1) {
        GHOST_CALL_GOTO(v1->fromVec(v1,v1compact,0,0),err,ret);
        v1compact->destroy(v1compact);
    }
    if (v2compact != v2) {
        v2compact->destroy(v2compact);
    }
    
    goto out;
err:
out:
    GHOST_CALL_RETURN(ghost_cu_free(d_a));
    GHOST_CALL_RETURN(ghost_cu_free(d_b));
    cudaDeviceSynchronize();
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_cm_cu_vaxpbypcz(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a, void *b, ghost_densemat_t *v3, void *c)
{
    if (v1->traits.datatype != v2->traits.datatype || v1->traits.datatype != v3->traits.datatype)
    {
        ERROR_LOG("Cannot VAXPBYPCZ vectors with different data types");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
    
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret = GHOST_SUCCESS;

    void *d_a;
    void *d_b;
    void *d_c;
    size_t sizeofdt;

    ghost_datatype_size(&sizeofdt,v1->traits.datatype);
    
    GHOST_CALL_GOTO(ghost_cu_malloc(&d_a,v1->traits.ncols*sizeofdt),err,ret);
    GHOST_CALL_GOTO(ghost_cu_malloc(&d_b,v1->traits.ncols*sizeofdt),err,ret);
    GHOST_CALL_GOTO(ghost_cu_malloc(&d_c,v1->traits.ncols*sizeofdt),err,ret);
   
    ghost_lidx_t col; 
    for (col=0; col<v1->traits.ncols; col++) {
            GHOST_CALL_GOTO(ghost_cu_upload(&((char *)d_a)[col*sizeofdt],&((char *)a)[col*sizeofdt],sizeofdt),err,ret);
            GHOST_CALL_GOTO(ghost_cu_upload(&((char *)d_b)[col*sizeofdt],&((char *)b)[col*sizeofdt],sizeofdt),err,ret);
            GHOST_CALL_GOTO(ghost_cu_upload(&((char *)d_c)[col*sizeofdt],&((char *)c)[col*sizeofdt],sizeofdt),err,ret);
    }
    
    
    void *v1val, *v2val, *v3val;
    ghost_densemat_t *v1compact, *v2compact, *v3compact;
    
    if (v1->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        INFO_LOG("Cloning (and compressing) v1 before operation");
        GHOST_CALL_GOTO(v1->clone(v1,&v1compact,v1->traits.nrows,0,v1->traits.ncols,0),err,ret);
    } else {
        v1compact = v1;
    }
    if (v2->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        INFO_LOG("Cloning (and compressing) v2 before operation");
        GHOST_CALL_GOTO(v2->clone(v2,&v2compact,v2->traits.nrows,0,v2->traits.ncols,0),err,ret);
    } else {
        v2compact = v2;
    }
    if (v3->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        INFO_LOG("Cloning (and compressing) v3 before operation");
        GHOST_CALL_GOTO(v3->clone(v3,&v3compact,v3->traits.nrows,0,v3->traits.ncols,0),err,ret);
    } else {
        v3compact = v3;
    }
    GHOST_CALL_GOTO(ghost_densemat_cu_valptr(v1compact,&v1val),err,ret);
    GHOST_CALL_GOTO(ghost_densemat_cu_valptr(v2compact,&v2val),err,ret);
    GHOST_CALL_GOTO(ghost_densemat_cu_valptr(v3compact,&v3val),err,ret);
    

    if (v1->traits.datatype & GHOST_DT_COMPLEX)
    {
        if (v1->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_vaxpbypcz_kernel<cuDoubleComplex><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                (cuDoubleComplex *)v1val, (cuDoubleComplex *)v2val, (cuDoubleComplex *)v3val, (cuDoubleComplex *)d_a,(cuDoubleComplex *)d_b,
                (cuDoubleComplex *)d_c, v1->traits.nrows,v1->traits.ncols,v1->stride,v2->stride, v3->stride);
        } 
        else 
        {
            cu_vaxpbypcz_kernel<cuFloatComplex><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                (cuFloatComplex *)v1val, (cuFloatComplex *)v2val, (cuFloatComplex *)v3val, (cuFloatComplex *)d_a,(cuFloatComplex *)d_b,
                (cuFloatComplex *)d_c, v1->traits.nrows,v1->traits.ncols,v1->stride,v2->stride, v3->stride);
        }
    }
    else
    {
        if (v1->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_vaxpbypcz_kernel<double><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                (double *)v1val, (double *)v2val, (double *)v3val, (double *)d_a,(double *)d_b,
                (double *)d_c, v1->traits.nrows,v1->traits.ncols,v1->stride,v2->stride, v3->stride);
        } 
        else 
        {
            cu_vaxpbypcz_kernel<float><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                (float *)v1val, (float *)v2val, (float *)v3val, (float *)d_a,(float *)d_b,
                (float *)d_c, v1->traits.nrows,v1->traits.ncols,v1->stride,v2->stride, v3->stride);
        }
    }
    if (v1compact != v1) {
        GHOST_CALL_GOTO(v1->fromVec(v1,v1compact,0,0),err,ret);
        v1compact->destroy(v1compact);
    }
    if (v2compact != v2) {
        v2compact->destroy(v2compact);
    }
    if (v3compact != v3) {
        v3compact->destroy(v3compact);
    }
    
    goto out;
err:
out:
    GHOST_CALL_RETURN(ghost_cu_free(d_a));
    GHOST_CALL_RETURN(ghost_cu_free(d_b));
    GHOST_CALL_RETURN(ghost_cu_free(d_c));
    cudaDeviceSynchronize();
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_cm_cu_dotprod(ghost_densemat_t *vec, void *res, ghost_densemat_t *vec2)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret = GHOST_SUCCESS;
    
    if (vec->traits.datatype != vec2->traits.datatype)
    {
        ERROR_LOG("Cannot DOT vectors with different data types (%s and %s)",ghost_datatype_string(vec->traits.datatype),ghost_datatype_string(vec2->traits.datatype));
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
    size_t sizeofdt;
    ghost_datatype_size(&sizeofdt,vec->traits.datatype);
    ghost_densemat_t *veccompact;
    ghost_densemat_t *vec2compact;

    if (vec->traits.flags & GHOST_DENSEMAT_VIEW) {
        INFO_LOG("Cloning (and compressing) vec1 before dotproduct");
        vec->clone(vec,&veccompact,vec->traits.nrows,0,vec->traits.ncols,0);
    } else {
        veccompact = vec;
    }
    if (vec2->traits.flags & GHOST_DENSEMAT_VIEW) {
        INFO_LOG("Cloning (and compressing) vec1 before dotproduct");
        vec2->clone(vec2,&vec2compact,vec2->traits.nrows,0,vec2->traits.ncols,0);
    } else {
        vec2compact = vec2;
    }
  
     
    cublasHandle_t ghost_cublas_handle;
    GHOST_CALL_GOTO(ghost_cu_cublas_handle(&ghost_cublas_handle),err,ret); 
    ghost_lidx_t v;
    for (v=0; v<veccompact->traits.ncols; v++)
    {
        char *v1 = veccompact->cu_val + v*veccompact->stride*veccompact->elSize;
        char *v2 = vec2compact->cu_val + v*vec2compact->stride*vec2compact->elSize;
        if (vec->traits.datatype & GHOST_DT_COMPLEX)
        {
            if (vec->traits.datatype & GHOST_DT_DOUBLE)
            {
                CUBLAS_CALL_GOTO(cublasZdotc(ghost_cublas_handle,vec->traits.nrows,
                            (const cuDoubleComplex *)v2,1,(const cuDoubleComplex *)v1,1,&((cuDoubleComplex *)res)[v]),err,ret);
            } 
            else 
            {
                CUBLAS_CALL_GOTO(cublasCdotc(ghost_cublas_handle,vec->traits.nrows,
                            (const cuFloatComplex *)v1,1,(const cuFloatComplex *)v2,1,&((cuFloatComplex *)res)[v]),err,ret);
            }
        }
        else
        {
            if (vec->traits.datatype & GHOST_DT_DOUBLE)
            {
                CUBLAS_CALL_GOTO(cublasDdot(ghost_cublas_handle,vec->traits.nrows,
                            (const double *)v1,1,(const double *)v2,1,&((double *)res)[v]),err,ret);
            } 
            else 
            {
                CUBLAS_CALL_GOTO(cublasSdot(ghost_cublas_handle,vec->traits.nrows,
                            (const float *)v1,1,(const float *)v2,1,&((float *)res)[v]),err,ret);
            }
        }
    }
    if (veccompact != vec) {
        veccompact->destroy(veccompact);
    }
    if (vec2compact != vec2) {
        vec2compact->destroy(vec2compact);
    }

    goto out;
err:
out:
    cudaDeviceSynchronize();
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_cm_cu_axpy(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret = GHOST_SUCCESS; 
    
    if (v1->traits.datatype & GHOST_DT_COMPLEX)
    {
        if (v1->traits.datatype & GHOST_DT_DOUBLE)
        {
            const cuDoubleComplex one = make_cuDoubleComplex(1.,0);
            ret =  ghost_densemat_cm_cu_axpby(v1,v2,a,(void *)&one);
        } 
        else 
        {
            const cuFloatComplex one = make_cuFloatComplex(1.,0.);
            ret = ghost_densemat_cm_cu_axpby(v1,v2,a,(void *)&one);
        }
    }
    else
    {
        if (v1->traits.datatype & GHOST_DT_DOUBLE)
        {
            const double one = 1.;
            ret = ghost_densemat_cm_cu_axpby(v1,v2,a,(void *)&one);
        } 
        else 
        {
            const float one = 1.f;
            ret = ghost_densemat_cm_cu_axpby(v1,v2,a,(void *)&one);
        }
    }
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_cm_cu_axpby(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a, void *b)
{
    if (v1->traits.datatype != v2->traits.datatype)
    {
        ERROR_LOG("Cannot AXPBY vectors with different data types");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret = GHOST_SUCCESS;
    void *v1val, *v2val;
    ghost_densemat_t *v1compact, *v2compact;
    
    if (v1->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        INFO_LOG("Cloning (and compressing) v1 before operation");
        GHOST_CALL_GOTO(v1->clone(v1,&v1compact,v1->traits.nrows,0,v1->traits.ncols,0),err,ret);
    } else {
        v1compact = v1;
    }
    if (v2->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        INFO_LOG("Cloning (and compressing) v2 before operation");
        GHOST_CALL_GOTO(v2->clone(v2,&v2compact,v2->traits.nrows,0,v2->traits.ncols,0),err,ret);
    } else {
        v2compact = v2;
    }
    GHOST_CALL_GOTO(ghost_densemat_cu_valptr(v1compact,&v1val),err,ret);
    GHOST_CALL_GOTO(ghost_densemat_cu_valptr(v2compact,&v2val),err,ret);

    if (v1->traits.datatype & GHOST_DT_COMPLEX)
    {
        if (v1->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_axpby_kernel<cuDoubleComplex><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>
                ((cuDoubleComplex *)v1val, (cuDoubleComplex *)v2val,*((cuDoubleComplex *)a),*((cuDoubleComplex *)b),v1->traits.nrows,v1->traits.ncols,v1->stride,v2->stride);
        } 
        else 
        {
            cu_axpby_kernel<cuFloatComplex><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>
                ((cuFloatComplex *)v1val, (cuFloatComplex *)v2val,*((cuFloatComplex *)a),*((cuFloatComplex *)b),v1->traits.nrows,v1->traits.ncols,v1->stride,v2->stride);
            
        }
    }
    else
    {
        if (v1->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_axpby_kernel<double><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>
                ((double *)v1val, (double *)v2val,*((double *)a),*((double *)b),v1->traits.nrows,v1->traits.ncols,v1->stride,v2->stride);
        } 
        else 
        {
            cu_axpby_kernel<float><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>
                ((float *)v1val, (float *)v2val,*((float *)a),*((float *)b),v1->traits.nrows,v1->traits.ncols,v1->stride,v2->stride);
        }
    }
    if (v1compact != v1) {
        GHOST_CALL_GOTO(v1->fromVec(v1,v1compact,0,0),err,ret);
        v1compact->destroy(v1compact);
    }
    if (v2compact != v2) {
        v2compact->destroy(v2compact);
    }

    goto out;
err:
out:
    cudaDeviceSynchronize();
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_cm_cu_axpbypcz(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a, void *b, ghost_densemat_t *v3, void *c)
{
    if (v1->traits.datatype != v2->traits.datatype || v1->traits.datatype != v3->traits.datatype)
    {
        ERROR_LOG("Cannot VAXPBYPCZ vectors with different data types");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
    
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret = GHOST_SUCCESS;

    size_t sizeofdt;

    ghost_datatype_size(&sizeofdt,v1->traits.datatype);
    
    void *v1val, *v2val, *v3val;
    ghost_densemat_t *v1compact, *v2compact, *v3compact;
    
    if (v1->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        INFO_LOG("Cloning (and compressing) v1 before operation");
        GHOST_CALL_GOTO(v1->clone(v1,&v1compact,v1->traits.nrows,0,v1->traits.ncols,0),err,ret);
    } else {
        v1compact = v1;
    }
    if (v2->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        INFO_LOG("Cloning (and compressing) v2 before operation");
        GHOST_CALL_GOTO(v2->clone(v2,&v2compact,v2->traits.nrows,0,v2->traits.ncols,0),err,ret);
    } else {
        v2compact = v2;
    }
    if (v3->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        INFO_LOG("Cloning (and compressing) v3 before operation");
        GHOST_CALL_GOTO(v3->clone(v3,&v3compact,v3->traits.nrows,0,v3->traits.ncols,0),err,ret);
    } else {
        v3compact = v3;
    }
    GHOST_CALL_GOTO(ghost_densemat_cu_valptr(v1compact,&v1val),err,ret);
    GHOST_CALL_GOTO(ghost_densemat_cu_valptr(v2compact,&v2val),err,ret);
    GHOST_CALL_GOTO(ghost_densemat_cu_valptr(v3compact,&v3val),err,ret);
    

    if (v1->traits.datatype & GHOST_DT_COMPLEX)
    {
        if (v1->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_axpbypcz_kernel<cuDoubleComplex><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                (cuDoubleComplex *)v1val, (cuDoubleComplex *)v2val, (cuDoubleComplex *)v3val, *(cuDoubleComplex *)a,*(cuDoubleComplex *)b,
                *(cuDoubleComplex *)c, v1->traits.nrows,v1->traits.ncols,v1->stride,v2->stride, v3->stride);
        } 
        else 
        {
            cu_axpbypcz_kernel<cuFloatComplex><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                (cuFloatComplex *)v1val, (cuFloatComplex *)v2val, (cuFloatComplex *)v3val, *(cuFloatComplex *)a,*(cuFloatComplex *)b,
                *(cuFloatComplex *)c, v1->traits.nrows,v1->traits.ncols,v1->stride,v2->stride, v3->stride);
        }
    }
    else
    {
        if (v1->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_axpbypcz_kernel<double><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                (double *)v1val, (double *)v2val, (double *)v3val, *(double *)a,*(double *)b,
                *(double *)c, v1->traits.nrows,v1->traits.ncols,v1->stride,v2->stride, v3->stride);
        } 
        else 
        {
            cu_axpbypcz_kernel<float><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                (float *)v1val, (float *)v2val, (float *)v3val, *(float *)a,*(float *)b,
                *(float *)c, v1->traits.nrows,v1->traits.ncols,v1->stride,v2->stride, v3->stride);
        }
    }
    if (v1compact != v1) {
        GHOST_CALL_GOTO(v1->fromVec(v1,v1compact,0,0),err,ret);
        v1compact->destroy(v1compact);
    }
    if (v2compact != v2) {
        v2compact->destroy(v2compact);
    }
    if (v3compact != v3) {
        v3compact->destroy(v3compact);
    }
    
    goto out;
err:
out:
    cudaDeviceSynchronize();
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_cm_cu_scale(ghost_densemat_t *vec, void *a)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret = GHOST_SUCCESS;
    
    void *vecval;
    ghost_densemat_t *veccompact;
    
    if (vec->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        INFO_LOG("Cloning (and compressing) vec before operation");
        GHOST_CALL_GOTO(vec->clone(vec,&veccompact,vec->traits.nrows,0,vec->traits.ncols,0),err,ret);
    } else {
        veccompact = vec;
    }
    GHOST_CALL_GOTO(ghost_densemat_cu_valptr(veccompact,&vecval),err,ret);
    
    if (vec->traits.datatype & GHOST_DT_COMPLEX)
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_scale_kernel<cuDoubleComplex><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (cuDoubleComplex *)vecval, *(cuDoubleComplex *)a,
                    vec->traits.nrows,vec->traits.ncols,vec->stride);
        } 
        else 
        {
            cu_scale_kernel<cuFloatComplex><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (cuFloatComplex *)vecval, *(cuFloatComplex *)a,
                    vec->traits.nrows,vec->traits.ncols,vec->stride);
        }
    }
    else
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_scale_kernel<double><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (double *)vecval, *(double *)a,
                    vec->traits.nrows,vec->traits.ncols,vec->stride);
        } 
        else 
        {
            cu_scale_kernel<float><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (float *)vecval, *(float *)a,
                    vec->traits.nrows,vec->traits.ncols,vec->stride);
        }
    }
    if (veccompact != vec) {
        INFO_LOG("Transform back");
        GHOST_CALL_GOTO(vec->fromVec(vec,veccompact,0,0),err,ret);
        veccompact->destroy(veccompact);
    }
    
    goto out;

err:

out:
    cudaDeviceSynchronize();
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    
    return ret;
}

extern "C" ghost_error_t ghost_densemat_cm_cu_vscale(ghost_densemat_t *vec, void *a)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret = GHOST_SUCCESS;

    void *d_a;
    ghost_idx_t c;
    void *vecval;
    ghost_densemat_t *veccompact;
    
    if (vec->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        INFO_LOG("Cloning (and compressing) vec before operation");
        GHOST_CALL_GOTO(vec->clone(vec,&veccompact,vec->traits.nrows,0,vec->traits.ncols,0),err,ret);
    } else {
        veccompact = vec;
    }
    GHOST_CALL_GOTO(ghost_densemat_cu_valptr(veccompact,&vecval),err,ret);

    GHOST_CALL_GOTO(ghost_cu_malloc(&d_a,vec->traits.ncols*vec->elSize),err,ret);
    
    for (c=0; c<vec->traits.ncols; c++) {
        GHOST_CALL_GOTO(ghost_cu_upload(&((char *)d_a)[c*vec->elSize],&((char *)a)[c*vec->elSize],vec->elSize),err,ret);
    }
    
    if (vec->traits.datatype & GHOST_DT_COMPLEX)
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_vscale_kernel<cuDoubleComplex><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (cuDoubleComplex *)vecval, (cuDoubleComplex *)d_a,
                    vec->traits.nrows,vec->traits.ncols,vec->stride);
        } 
        else 
        {
            cu_vscale_kernel<cuFloatComplex><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (cuFloatComplex *)vecval, (cuFloatComplex *)d_a,
                    vec->traits.nrows,vec->traits.ncols,vec->stride);
        }
    }
    else
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_vscale_kernel<double><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (double *)vecval, (double *)d_a,
                    vec->traits.nrows,vec->traits.ncols,vec->stride);
        } 
        else 
        {
            cu_vscale_kernel<float><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (float *)vecval, (float *)d_a,
                    vec->traits.nrows,vec->traits.ncols,vec->stride);
        }
    }
    if (veccompact != vec) {
        INFO_LOG("Transform back");
        GHOST_CALL_GOTO(vec->fromVec(vec,veccompact,0,0),err,ret);
        veccompact->destroy(veccompact);
    }

    goto out;
err:
out:
    cudaDeviceSynchronize();
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_cm_cu_fromScalar(ghost_densemat_t *vec, void *a)
{
    ghost_error_t ret = GHOST_SUCCESS;
    int needInit = 0;
    ghost_densemat_cm_malloc(vec,&needInit);
    
    void *vecval;
    ghost_densemat_t *veccompact;
    
    if (vec->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        INFO_LOG("Cloning (and compressing) vec before operation");
        GHOST_CALL_GOTO(vec->clone(vec,&veccompact,vec->traits.nrows,0,vec->traits.ncols,0),err,ret);
    } else {
        veccompact = vec;
    }
    GHOST_CALL_GOTO(ghost_densemat_cu_valptr(veccompact,&vecval),err,ret);

    if (vec->traits.datatype & GHOST_DT_COMPLEX)
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_fromscalar_kernel<cuDoubleComplex><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (cuDoubleComplex *)vecval, *(cuDoubleComplex *)a,
                    vec->traits.nrows,vec->traits.ncols,vec->stride);
        } 
        else 
        {
            cu_fromscalar_kernel<cuFloatComplex><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (cuFloatComplex *)vecval, *(cuFloatComplex *)a,
                    vec->traits.nrows,vec->traits.ncols,vec->stride);
        }
    }
    else
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_fromscalar_kernel<double><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (double *)vecval, *(double *)a,
                    vec->traits.nrows,vec->traits.ncols,vec->stride);
        } 
        else 
        {
            cu_fromscalar_kernel<float><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (float *)vecval, *(float *)a,
                    vec->traits.nrows,vec->traits.ncols,vec->stride);
        }
    }
    if (veccompact != vec) {
        INFO_LOG("Transform back");
        GHOST_CALL_GOTO(vec->fromVec(vec,veccompact,0,0),err,ret);
        veccompact->destroy(veccompact);
    }
    
    goto out;
err:
out:
    return ret;
}

extern "C" ghost_error_t ghost_densemat_cm_cu_fromRand(ghost_densemat_t *vec)
{
    ghost_error_t ret = GHOST_SUCCESS;

    ghost_densemat_t *onevec;
    long pid = getpid();
    double time;
    double one[] = {1.,1.};
    float fone[] = {1.,0.};
    double minusahalf[] = {-0.5,0.};
    float fminusahalf[] = {-0.5,0.};
    
    ghost_timing_wcmilli(&time);
    int needInit = 0;
    ghost_densemat_cm_malloc(vec,&needInit);
    curandGenerator_t gen;
    CURAND_CALL_GOTO(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT),err,ret);
    CURAND_CALL_GOTO(curandSetPseudoRandomGeneratorSeed(gen,ghost_rand_cu_seed_get()),err,ret);

    vec->clone(vec,&onevec,vec->traits.nrows,0,vec->traits.ncols,0);
    onevec->fromScalar(onevec,one);

    one[1] = 0.;
    void *valptr;
    ghost_densemat_t *compactvec;

    if ((vec->traits.ncolsorig != vec->traits.ncols) || (vec->traits.flags & GHOST_DENSEMAT_SCATTERED)) {
        INFO_LOG("Cloning (and compressing) vec before operation");
        vec->clone(vec,&compactvec,vec->traits.nrows,0,vec->traits.ncols,0);
    } else {
        compactvec = vec;
    }
    ghost_densemat_cu_valptr(compactvec,&valptr);


    if (vec->traits.datatype & GHOST_DT_COMPLEX)
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            CURAND_CALL_GOTO(curandGenerateUniformDouble(gen,
                        (double *)valptr,
                        compactvec->stride*compactvec->traits.ncols*2),err,ret);
        } 
        else 
        {
            CURAND_CALL_GOTO(curandGenerateUniform(gen,
                        (float *)valptr,
                        compactvec->stride*compactvec->traits.ncols*2),err,ret);
        }
    }
    else
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            CURAND_CALL_GOTO(curandGenerateUniformDouble(gen,
                        (double *)valptr,
                        compactvec->stride*compactvec->traits.ncols),err,ret);
        } 
        else 
        {
            CURAND_CALL_GOTO(curandGenerateUniform(gen,
                        (float *)valptr,
                        compactvec->stride*compactvec->traits.ncols),err,ret);
        }
    }
    if (compactvec->traits.datatype & GHOST_DT_DOUBLE) {
        compactvec->axpby(compactvec,onevec,minusahalf,one);
    } else {
        compactvec->axpby(compactvec,onevec,fminusahalf,fone);
    }
    if (compactvec != vec) {
        vec->fromVec(vec,compactvec,0,0);
        compactvec->destroy(compactvec);
    }
    goto out;
err:
out:
    CURAND_CALL_RETURN(curandDestroyGenerator(gen));
    onevec->destroy(onevec);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_cm_cu_conj(ghost_densemat_t *vec)
{
    if (vec->traits.datatype & GHOST_DT_REAL) {
        return GHOST_SUCCESS;
    }
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret = GHOST_SUCCESS;
    
    void *vecval;
    ghost_densemat_t *veccompact;
    
    if (vec->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        INFO_LOG("Cloning (and compressing) vec before operation");
        GHOST_CALL_GOTO(vec->clone(vec,&veccompact,vec->traits.nrows,0,vec->traits.ncols,0),err,ret);
    } else {
        veccompact = vec;
    }
    GHOST_CALL_GOTO(ghost_densemat_cu_valptr(veccompact,&vecval),err,ret);
    
    if (vec->traits.datatype & GHOST_DT_DOUBLE)
    {
        cu_conj_kernel<cuDoubleComplex><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                (cuDoubleComplex *)vecval,
                vec->traits.nrows,vec->traits.ncols,vec->stride);
    } 
    else 
    {
        cu_conj_kernel<cuFloatComplex><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                (cuFloatComplex *)vecval,
                vec->traits.nrows,vec->traits.ncols,vec->stride);
    }
    if (veccompact != vec) {
        INFO_LOG("Transform back");
        GHOST_CALL_GOTO(vec->fromVec(vec,veccompact,0,0),err,ret);
        veccompact->destroy(veccompact);
    }
    
    goto out;

err:

out:
    cudaDeviceSynchronize();
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    
    return ret;
}
