#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/densemat_rm.h"
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


#define THREADSPERBLOCK 1024


template<typename T>  
__global__ static void cu_vaxpby_kernel(T *v1, T *v2, T *a, T *b, ghost_lidx_t nrows, ghost_lidx_t ncols, ghost_lidx_t ld1, ghost_lidx_t ld2)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    for (;idx < nrows; idx+=gridDim.x*blockDim.x) {
        ghost_lidx_t v;
        for (v=0; v<ncols; v++) {
            v1[idx*ld1+v] = axpby<T>(v2[idx*ld2+v],v1[idx*ld1+v],a[v],b[v]);
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
            v1[idx*ld1+v] = axpby<T>(v2[idx*ld2+v],v1[idx*ld1+v],a,b);
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
            vec[idx*ld+v] = scale<T>(a,vec[idx*ld+v]);
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
            vec[idx*ld+v] = scale<T>(a[v],vec[idx*ld+v]);
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
            vec[idx*ld+v] = a;
        }
    }
}


template<typename T>  
__global__ static void cu_communicationassembly_kernel(T *vec, T *work, ghost_lidx_t offs, ghost_lidx_t *duelist, ghost_lidx_t ncols, ghost_lidx_t ndues, ghost_lidx_t ncolspadded, ghost_lidx_t *perm)
{
    int due = blockIdx.x*blockDim.x+threadIdx.x;
    int col = threadIdx.y;

    if (perm) {
        for (;due < ndues; due+=gridDim.x*blockDim.x) {
            work[(offs+due)*ncols+col] = vec[perm[duelist[due]]*ncolspadded+col];
        }
    } else {
        for (;due < ndues; due+=gridDim.x*blockDim.x) {
            work[(offs+due)*ncols+col] = vec[duelist[due]*ncolspadded+col];
        }
    }
}

extern "C" ghost_error_t ghost_densemat_rm_cu_communicationassembly(void * work, ghost_lidx_t *dueptr, ghost_densemat_t *vec, ghost_lidx_t *perm)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
  
    if (!vec->context->cu_duelist) {
       ERROR_LOG("cu_duelist must not be NULL!");
       return GHOST_ERR_INVALID_ARG;
    }
    if (!dueptr) {
       ERROR_LOG("dueptr must not be NULL!");
       return GHOST_ERR_INVALID_ARG;
    }


    int nrank, proc, me;
    ghost_context_t *ctx = vec->context;
    
    ghost_nrank(&nrank,ctx->mpicomm); 
    ghost_rank(&me,ctx->mpicomm);
            
    for (proc=0 ; proc<nrank ; proc++){
        dim3 block((int)ceil((double)THREADSPERBLOCK/vec->traits.ncols),vec->traits.ncols);
        dim3 grid((int)ceil((double)ctx->dues[proc]/block.x));
        DEBUG_LOG(1,"communication assembly with grid %d block %dx%d %d->%d",grid.x,block.x,block.y,me,proc);
        if (vec->traits.datatype & GHOST_DT_COMPLEX)
        {
            if (vec->traits.datatype & GHOST_DT_DOUBLE)
            {
                if (ctx->dues[proc]) {
                    cu_communicationassembly_kernel<cuDoubleComplex><<< grid,block >>>((cuDoubleComplex *)vec->cu_val, ((cuDoubleComplex *)work),dueptr[proc],ctx->cu_duelist[proc],vec->traits.ncols,ctx->dues[proc],vec->stride,perm);
                }
            } 
            else 
            {
                if (ctx->dues[proc]) {
                    cu_communicationassembly_kernel<cuFloatComplex><<< grid,block >>>((cuFloatComplex *)vec->cu_val, ((cuFloatComplex *)work),dueptr[proc],ctx->cu_duelist[proc],vec->traits.ncols,ctx->dues[proc],vec->stride,perm);
                }
            }
        }
        else
        {
            if (vec->traits.datatype & GHOST_DT_DOUBLE)
            {
                if (ctx->dues[proc]) {
                    cu_communicationassembly_kernel<double><<< grid,block >>>((double *)vec->cu_val, ((double *)work),dueptr[proc],ctx->cu_duelist[proc],vec->traits.ncols,ctx->dues[proc],vec->stride,perm);
                }
            } 
            else 
            {
                if (ctx->dues[proc]) {
                    cu_communicationassembly_kernel<float><<< grid,block >>>((float *)vec->cu_val, ((float *)work),dueptr[proc],ctx->cu_duelist[proc],vec->traits.ncols,ctx->dues[proc],vec->stride,perm);
                }
            }
        }
    }
    cudaDeviceSynchronize();

    if (cudaPeekAtLastError() != cudaSuccess) {
        ERROR_LOG("Error in kernel");
        return GHOST_ERR_CUDA;
    }
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return GHOST_SUCCESS;

}

extern "C" ghost_error_t ghost_densemat_rm_cu_vaxpy(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a)
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
            ret =  ghost_densemat_rm_cu_vaxpby(v1,v2,a,one);
        } 
        else 
        {
            complex float *one;
            GHOST_CALL_RETURN(ghost_malloc((void **)&one,v1->traits.ncols*sizeof(complex float)));
            int v;
            for (v=0; v<v1->traits.ncols; v++) {
                one[v] = 1.+I*0.;
            }
            ret =  ghost_densemat_rm_cu_vaxpby(v1,v2,a,one);
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
            ret =  ghost_densemat_rm_cu_vaxpby(v1,v2,a,one);
        } 
        else 
        {
            float *one;
            GHOST_CALL_RETURN(ghost_malloc((void **)&one,v1->traits.ncols*sizeof(float)));
            int v;
            for (v=0; v<v1->traits.ncols; v++) {
                one[v] = 1.;
            }
            ret =  ghost_densemat_rm_cu_vaxpby(v1,v2,a,one);
        }
    }
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
}
    
extern "C" ghost_error_t ghost_densemat_rm_cu_vaxpby(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a, void *b)
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

extern "C" ghost_error_t ghost_densemat_rm_cu_dotprod(ghost_densemat_t *vec, void *res, ghost_densemat_t *vec2)
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
        INFO_LOG("Cloning (and compressing) vec2 before dotproduct");
        vec2->clone(vec2,&vec2compact,vec2->traits.nrows,0,vec2->traits.ncols,0);
    } else {
        vec2compact = vec2;
    }
  
     
    cublasHandle_t ghost_cublas_handle;
    GHOST_CALL_GOTO(ghost_cu_cublas_handle(&ghost_cublas_handle),err,ret); 
    ghost_lidx_t v;
    for (v=0; v<veccompact->traits.ncols; v++)
    {
        char *v1 = veccompact->cu_val+v*veccompact->elSize;
        char *v2 = vec2compact->cu_val+v*veccompact->elSize;
        if (vec->traits.datatype & GHOST_DT_COMPLEX)
        {
            if (vec->traits.datatype & GHOST_DT_DOUBLE)
            {
                CUBLAS_CALL_GOTO(cublasZdotc(ghost_cublas_handle,vec->traits.nrows,
                            (const cuDoubleComplex *)v1,veccompact->stride,(const cuDoubleComplex *)v2,vec2compact->stride,&((cuDoubleComplex *)res)[v]),err,ret);
            } 
            else 
            {
                CUBLAS_CALL_GOTO(cublasCdotc(ghost_cublas_handle,vec->traits.nrows,
                            (const cuFloatComplex *)v1,veccompact->stride,(const cuFloatComplex *)v2,vec2compact->stride,&((cuFloatComplex *)res)[v]),err,ret);
            }
        }
        else
        {
            if (vec->traits.datatype & GHOST_DT_DOUBLE)
            {
                CUBLAS_CALL_GOTO(cublasDdot(ghost_cublas_handle,vec->traits.nrows,
                            (const double *)v1,veccompact->stride,(const double *)v2,vec2compact->stride,&((double *)res)[v]),err,ret);
            } 
            else 
            {
                CUBLAS_CALL_GOTO(cublasSdot(ghost_cublas_handle,vec->traits.nrows,
                            (const float *)v1,veccompact->stride,(const float *)v2,vec2compact->stride,&((float *)res)[v]),err,ret);
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

extern "C" ghost_error_t ghost_densemat_rm_cu_axpy(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret = GHOST_SUCCESS; 
    
    if (v1->traits.datatype & GHOST_DT_COMPLEX)
    {
        if (v1->traits.datatype & GHOST_DT_DOUBLE)
        {
            const cuDoubleComplex one = make_cuDoubleComplex(1.,0);
            ret =  ghost_densemat_rm_cu_axpby(v1,v2,a,(void *)&one);
        } 
        else 
        {
            const cuFloatComplex one = make_cuFloatComplex(1.,0.);
            ret = ghost_densemat_rm_cu_axpby(v1,v2,a,(void *)&one);
        }
    }
    else
    {
        if (v1->traits.datatype & GHOST_DT_DOUBLE)
        {
            const double one = 1.;
            ret = ghost_densemat_rm_cu_axpby(v1,v2,a,(void *)&one);
        } 
        else 
        {
            const float one = 1.f;
            ret = ghost_densemat_rm_cu_axpby(v1,v2,a,(void *)&one);
        }
    }
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_rm_cu_axpby(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a, void *b)
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

extern "C" ghost_error_t ghost_densemat_rm_cu_scale(ghost_densemat_t *vec, void *a)
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

extern "C" ghost_error_t ghost_densemat_rm_cu_vscale(ghost_densemat_t *vec, void *a)
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

extern "C" ghost_error_t ghost_densemat_rm_cu_fromScalar(ghost_densemat_t *vec, void *a)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_INITIALIZATION);
    ghost_error_t ret = GHOST_SUCCESS;
    int needInit = 0;
    ghost_densemat_rm_malloc(vec,&needInit);
    
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
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_INITIALIZATION);
    return ret;
}

extern "C" ghost_error_t ghost_densemat_rm_cu_fromRand(ghost_densemat_t *vec)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_INITIALIZATION);
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
    ghost_densemat_rm_malloc(vec,&needInit);
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
                        compactvec->traits.nrowspadded*compactvec->traits.ncols*2),err,ret);
        } 
        else 
        {
            CURAND_CALL_GOTO(curandGenerateUniform(gen,
                        (float *)valptr,
                        compactvec->traits.nrowspadded*compactvec->traits.ncols*2),err,ret);
        }
    }
    else
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            CURAND_CALL_GOTO(curandGenerateUniformDouble(gen,
                        (double *)valptr,
                        compactvec->traits.nrowspadded*compactvec->traits.ncols),err,ret);
        } 
        else 
        {
            CURAND_CALL_GOTO(curandGenerateUniform(gen,
                        (float *)valptr,
                        compactvec->traits.nrowspadded*compactvec->traits.ncols),err,ret);
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
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_INITIALIZATION);
    CURAND_CALL_RETURN(curandDestroyGenerator(gen));
    onevec->destroy(onevec);

    return ret;
}
#if 0

extern "C" ghost_error_t ghost_densemat_rm_cu_vaxpy(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret = GHOST_SUCCESS;
    void *d_a;
    size_t sizeofdt;
    char colfield[v1->traits.ncolsorig];
    char rowfield[v1->traits.nrowsorig];
    char *cucolfield = NULL, *curowfield = NULL;
    int grid = (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK);
    dim3 block (THREADSPERBLOCK/v1->traits.ncolsorig,v1->traits.ncolsorig);
    ghost_datatype_size(&sizeofdt,v1->traits.datatype);
    
    GHOST_CALL_GOTO(ghost_cu_malloc(&d_a,v1->traits.ncols*sizeofdt),err,ret);
    
    ghost_cu_upload(d_a,a,v1->traits.ncols*sizeofdt);
    if (v1->traits.datatype != v2->traits.datatype)
    {
        ERROR_LOG("Cannot VAXPY vectors with different data types");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }

    
    if (ghost_bitmap_weight(v1->ldmask) != v1->traits.ncolsorig || 
            ghost_bitmap_weight(v1->trmask) != v1->traits.nrowsorig ||
            ghost_bitmap_weight(v2->ldmask) != v2->traits.ncolsorig ||
            ghost_bitmap_weight(v2->trmask) != v2->traits.nrowsorig) { 
        
        if (!ghost_bitmap_isequal(v1->ldmask,v2->ldmask) || !ghost_bitmap_isequal(v1->trmask,v2->trmask)) {
            ERROR_LOG("The masks have to be equal!");
            ret = GHOST_ERR_INVALID_ARG;
            goto err;
        }
       
        WARNING_LOG("Potentially slow VAXPY operation because some rows or columns are masked out!");
        
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&cucolfield,v1->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&curowfield,v1->traits.nrowsorig),err,ret);

        ghost_densemat_mask2charfield(v1->ldmask,v1->traits.ncolsorig,colfield);
        ghost_densemat_mask2charfield(v1->trmask,v1->traits.nrowsorig,rowfield);

        GHOST_CALL_GOTO(ghost_cu_upload(cucolfield,colfield,v1->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_upload(curowfield,rowfield,v1->traits.nrowsorig),err,ret);
    }

    if (v1->traits.datatype & GHOST_DT_COMPLEX)
    {
        if (v1->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_vaxpy_kernel<cuDoubleComplex><<< grid,block >>>((cuDoubleComplex *)v1->cu_val, (cuDoubleComplex *)v2->cu_val,(cuDoubleComplex *)d_a,v1->traits.nrowsorig,curowfield,v1->traits.ncolsorig,cucolfield,v1->stride);
        } 
        else 
        {
            cu_vaxpy_kernel<cuFloatComplex><<< grid,block >>>((cuFloatComplex *)v1->cu_val, (cuFloatComplex *)v2->cu_val,(cuFloatComplex *)d_a,v1->traits.nrowsorig,curowfield,v1->traits.ncolsorig,cucolfield,v1->stride);
        }
    }
    else
    {
        if (v1->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_vaxpy_kernel<double><<< grid,block >>>((double *)v1->cu_val, (double *)v2->cu_val,(double *)d_a,v1->traits.nrowsorig,curowfield,v1->traits.ncolsorig,cucolfield,v1->stride);
        } 
        else 
        {
            cu_vaxpy_kernel<float><<< grid,block >>>((float *)v1->cu_val, (float *)v2->cu_val,(float *)d_a,v1->traits.nrowsorig,curowfield,v1->traits.ncolsorig,cucolfield,v1->stride);
        }
    }
    
    goto out;
err:
out:
    GHOST_CALL_RETURN(ghost_cu_free(cucolfield));
    GHOST_CALL_RETURN(ghost_cu_free(curowfield));
    GHOST_CALL_RETURN(ghost_cu_free(d_a));
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
}
    
extern "C" ghost_error_t ghost_densemat_rm_cu_vaxpby(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a, void *b)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret = GHOST_SUCCESS;

    void *d_a;
    void *d_b;
    size_t sizeofdt;
    char colfield[v1->traits.ncolsorig];
    char rowfield[v1->traits.nrowsorig];

    char *cucolfield = NULL, *curowfield = NULL;
    int grid = (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK);
    dim3 block (THREADSPERBLOCK/v1->traits.ncolsorig,v1->traits.ncolsorig);
    
    ghost_datatype_size(&sizeofdt,v1->traits.datatype);
    
    GHOST_CALL_GOTO(ghost_cu_malloc(&d_a,v1->traits.ncols*sizeofdt),err,ret);
    GHOST_CALL_GOTO(ghost_cu_malloc(&d_b,v1->traits.ncols*sizeofdt),err,ret);
    
    ghost_cu_upload(d_b,b,v1->traits.ncols*sizeofdt);
    
    if (v1->traits.datatype != v2->traits.datatype)
    {
        ERROR_LOG("Cannot VAXPBY vectors with different data types");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
    
    if (ghost_bitmap_weight(v1->ldmask) != v1->traits.ncolsorig || 
            ghost_bitmap_weight(v1->trmask) != v1->traits.nrowsorig ||
            ghost_bitmap_weight(v2->ldmask) != v2->traits.ncolsorig ||
            ghost_bitmap_weight(v2->trmask) != v2->traits.nrowsorig) { 
        
        if (!ghost_bitmap_isequal(v1->ldmask,v2->ldmask) || !ghost_bitmap_isequal(v1->trmask,v2->trmask)) {
            ERROR_LOG("The masks have to be equal!");
            ret = GHOST_ERR_INVALID_ARG;
            goto err;
        }
       
        WARNING_LOG("Potentially slow VAXPBY operation because some rows or columns are masked out!");
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&cucolfield,v1->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&curowfield,v1->traits.nrowsorig),err,ret);

        ghost_densemat_mask2charfield(v1->ldmask,v1->traits.ncolsorig,colfield);
        ghost_densemat_mask2charfield(v1->trmask,v1->traits.nrowsorig,rowfield);

        GHOST_CALL_GOTO(ghost_cu_upload(cucolfield,colfield,v1->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_upload(curowfield,rowfield,v1->traits.nrowsorig),err,ret);
    }

    if (v1->traits.datatype & GHOST_DT_COMPLEX)
    {
        if (v1->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_vaxpby_kernel<cuDoubleComplex><<< grid,block >>>(
                (cuDoubleComplex *)v1->cu_val, (cuDoubleComplex *)v2->cu_val,(cuDoubleComplex *)d_a,(cuDoubleComplex *)d_b,
                 v1->traits.nrowsorig,curowfield,v1->traits.ncolsorig,cucolfield,v1->stride);
        } 
        else 
        {
            cu_vaxpby_kernel<cuFloatComplex><<< grid,block >>>(
                (cuFloatComplex *)v1->cu_val, (cuFloatComplex *)v2->cu_val,(cuFloatComplex *)d_a,(cuFloatComplex *)d_b,
                 v1->traits.nrowsorig,curowfield,v1->traits.ncolsorig,cucolfield,v1->stride);
        }
    }
    else
    {
        if (v1->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_vaxpby_kernel<double><<< grid,block >>>(
                 (double *)v1->cu_val, (double *)v2->cu_val,(double *)d_a,(double *)d_b,
                 v1->traits.nrowsorig,curowfield,v1->traits.ncolsorig,cucolfield,v1->stride);
        } 
        else 
        {
            cu_vaxpby_kernel<float><<< grid,block >>>(
                (float *)v1->cu_val, (float *)v2->cu_val,(float *)d_a,(float *)d_b,
                 v1->traits.nrowsorig,curowfield,v1->traits.ncolsorig,cucolfield,v1->stride);
        }
    }
    goto out;
err:
out:
    GHOST_CALL_RETURN(ghost_cu_free(cucolfield));
    GHOST_CALL_RETURN(ghost_cu_free(curowfield));
    GHOST_CALL_RETURN(ghost_cu_free(d_a));
    GHOST_CALL_RETURN(ghost_cu_free(d_a));
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_rm_cu_dotprod(ghost_densemat_t *vec, void *res, ghost_densemat_t *vec2)
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
    ghost_densemat_t *vecclone;
    ghost_densemat_t *vec2clone;

    if (ghost_bitmap_weight(vec->ldmask) != vec->traits.ncolsorig || 
            ghost_bitmap_weight(vec->trmask) != vec->traits.nrowsorig) {
        INFO_LOG("Cloning (and compressing) vec1 before dotproduct");
        vec->clone(vec,&vecclone,vec->traits.nrows,0,vec->traits.ncols,0);
    } else {
        vecclone = vec;
    }
    if (ghost_bitmap_weight(vec2->ldmask) != vec2->traits.ncolsorig || 
            ghost_bitmap_weight(vec2->trmask) != vec2->traits.nrowsorig) {
        INFO_LOG("Cloning (and compressing) vec1 before dotproduct");
        vec2->clone(vec2,&vec2clone,vec2->traits.nrows,0,vec2->traits.ncols,0);
    } else {
        vec2clone = vec2;
    }
  
     
    cublasHandle_t ghost_cublas_handle;
    GHOST_CALL_GOTO(ghost_cu_cublas_handle(&ghost_cublas_handle),err,ret); 
    ghost_lidx_t v;
    for (v=0; v<vecclone->traits.ncols; v++)
    {
        char *v1 = &((char *)(vecclone->cu_val))[v*sizeofdt];
        char *v2 = &((char *)(vec2clone->cu_val))[v*sizeofdt];
        if (vec->traits.datatype & GHOST_DT_COMPLEX)
        {
            if (vec->traits.datatype & GHOST_DT_DOUBLE)
            {
                CUBLAS_CALL_GOTO(cublasZdotc(ghost_cublas_handle,vec->traits.nrows,
                            (const cuDoubleComplex *)v1,vec->stride,(const cuDoubleComplex *)v2,vec2->stride,&((cuDoubleComplex *)res)[v]),err,ret);
            } 
            else 
            {
                CUBLAS_CALL_GOTO(cublasCdotc(ghost_cublas_handle,vec->traits.nrows,
                            (const cuFloatComplex *)v1,vec->stride,(const cuFloatComplex *)v2,vec2->stride,&((cuFloatComplex *)res)[v]),err,ret);
            }
        }
        else
        {
            if (vec->traits.datatype & GHOST_DT_DOUBLE)
            {
                CUBLAS_CALL_GOTO(cublasDdot(ghost_cublas_handle,vec->traits.nrows,
                            (const double *)v1,vec->stride,(const double *)v2,vec2->stride,&((double *)res)[v]),err,ret);
            } 
            else 
            {
                CUBLAS_CALL_GOTO(cublasSdot(ghost_cublas_handle,vec->traits.nrows,
                            (const float *)v1,vec->stride,(const float *)v2,vec2->stride,&((float *)res)[v]),err,ret);
            }
        }
    }

    goto out;
err:
out:
    if (!ghost_bitmap_iscompact(vec->ldmask) || 
            !ghost_bitmap_iscompact(vec->trmask)) {
        vecclone->destroy(vecclone);
    }
    
    if (!ghost_bitmap_iscompact(vec2->ldmask) || 
            !ghost_bitmap_iscompact(vec2->trmask)) {
        vec2clone->destroy(vec2clone);
    }
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_rm_cu_axpy(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *a)
{
    if (vec->traits.datatype != vec2->traits.datatype)
    {
        ERROR_LOG("Cannot AXPY vectors with different data types");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret = GHOST_SUCCESS;
    
    char colfield[vec->traits.ncolsorig];
    char rowfield[vec->traits.nrowsorig];

    char *cucolfield = NULL, *curowfield = NULL;
    int grid = (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK);
    dim3 block (THREADSPERBLOCK/vec->traits.ncolsorig,vec->traits.ncolsorig); 

    if (ghost_bitmap_weight(vec->ldmask) != vec->traits.ncolsorig || 
            ghost_bitmap_weight(vec->trmask) != vec->traits.nrowsorig ||
            ghost_bitmap_weight(vec2->ldmask) != vec2->traits.ncolsorig ||
            ghost_bitmap_weight(vec2->trmask) != vec2->traits.nrowsorig) {

        if (!ghost_bitmap_isequal(vec->ldmask,vec2->ldmask) || !ghost_bitmap_isequal(vec->trmask,vec2->trmask)) {
            ERROR_LOG("The masks have to be equal!");
            ret = GHOST_ERR_INVALID_ARG;
            goto err;
        }
       
        WARNING_LOG("Potentially slow AXPY operation because some rows or columns are masked out!");
        
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&cucolfield,vec->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&curowfield,vec->traits.nrowsorig),err,ret);

        ghost_densemat_mask2charfield(vec->ldmask,vec->traits.ncolsorig,colfield);
        ghost_densemat_mask2charfield(vec->trmask,vec->traits.nrowsorig,rowfield);

        GHOST_CALL_GOTO(ghost_cu_upload(cucolfield,colfield,vec->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_upload(curowfield,rowfield,vec->traits.nrowsorig),err,ret);
    }

    if (vec->traits.datatype & GHOST_DT_COMPLEX)
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            const cuDoubleComplex one = make_cuDoubleComplex(1.,1.);
            cu_axpby_kernel<cuDoubleComplex><<< grid,block >>>
                ((cuDoubleComplex *)vec->cu_val, (cuDoubleComplex *)vec2->cu_val,*((cuDoubleComplex *)a),one,
                 vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->stride);
        } 
        else 
        {
            const cuFloatComplex one = make_cuFloatComplex(1.,1.);
            cu_axpby_kernel<cuFloatComplex><<< grid,block >>>
                ((cuFloatComplex *)vec->cu_val, (cuFloatComplex *)vec2->cu_val,*((cuFloatComplex *)a),one,
                 vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->stride);
        }
    }
    else
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_axpby_kernel<double><<< grid,block >>>
                ((double *)vec->cu_val, (double *)vec2->cu_val,*((double *)a),(double)1.,
                 vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->stride);
        } 
        else 
        {
            cu_axpby_kernel<float><<< grid,block >>>
                ((float *)vec->cu_val, (float *)vec2->cu_val,*((float *)a),(float)1.,
                 vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->stride);
        }
    }

    
    goto out;
err:
out:
    
    GHOST_CALL_RETURN(ghost_cu_free(cucolfield));
    GHOST_CALL_RETURN(ghost_cu_free(curowfield));
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_rm_cu_axpby(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a, void *b)
{
    if (v1->traits.datatype != v2->traits.datatype)
    {
        ERROR_LOG("Cannot AXPBY vectors with different data types");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret = GHOST_SUCCESS;

    char colfield[v1->traits.ncolsorig];
    char rowfield[v1->traits.nrowsorig];

    char *cucolfield = NULL, *curowfield = NULL;
    int grid = (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK);
    dim3 block (THREADSPERBLOCK/v1->traits.ncolsorig,v1->traits.ncolsorig); 
    INFO_LOG("block %dx%d",block.x,block.y);

    if (ghost_bitmap_weight(v1->ldmask) != v1->traits.ncolsorig || 
            ghost_bitmap_weight(v1->trmask) != v1->traits.nrowsorig ||
            ghost_bitmap_weight(v2->ldmask) != v2->traits.ncolsorig ||
            ghost_bitmap_weight(v2->trmask) != v2->traits.nrowsorig) { 
        
        if (!ghost_bitmap_isequal(v1->ldmask,v2->ldmask) || !ghost_bitmap_isequal(v1->trmask,v2->trmask)) {
            ERROR_LOG("The masks have to be equal!");
            ret = GHOST_ERR_INVALID_ARG;
            goto err;
        }
        
        WARNING_LOG("Potentially slow AXPBY operation because some rows or columns are masked out!");

        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&cucolfield,v1->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&curowfield,v1->traits.nrowsorig),err,ret);

        ghost_densemat_mask2charfield(v1->ldmask,v1->traits.ncolsorig,colfield);
        ghost_densemat_mask2charfield(v1->trmask,v1->traits.nrowsorig,rowfield);

        GHOST_CALL_GOTO(ghost_cu_upload(cucolfield,colfield,v1->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_upload(curowfield,rowfield,v1->traits.nrowsorig),err,ret);
    }


    if (v1->traits.datatype & GHOST_DT_COMPLEX)
    {
        if (v1->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_axpby_kernel<cuDoubleComplex><<< grid,block >>>
                ((cuDoubleComplex *)v1->cu_val, (cuDoubleComplex *)v2->cu_val,*((cuDoubleComplex *)a),*((cuDoubleComplex *)b),
                 v1->traits.nrowsorig,curowfield,v1->traits.ncolsorig,cucolfield,v1->stride);
        } 
        else 
        {
            cu_axpby_kernel<cuFloatComplex><<< grid,block >>>
                ((cuFloatComplex *)v1->cu_val, (cuFloatComplex *)v2->cu_val,*((cuFloatComplex *)a),*((cuFloatComplex *)b),
                 v1->traits.nrowsorig,curowfield,v1->traits.ncolsorig,cucolfield,v1->stride);
        }
    }
    else
    {
        if (v1->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_axpby_kernel<double><<< grid,block >>>
                ((double *)v1->cu_val, (double *)v2->cu_val,*((double *)a),*((double *)b),
                 v1->traits.nrowsorig,curowfield,v1->traits.ncolsorig,cucolfield,v1->stride);
        } 
        else 
        {
            cu_axpby_kernel<float><<< grid,block >>>
                ((float *)v1->cu_val, (float *)v2->cu_val,*((float *)a),*((float *)b),
                 v1->traits.nrowsorig,curowfield,v1->traits.ncolsorig,cucolfield,v1->stride);
        }
    }
    
    goto out;
err:
out:
    GHOST_CALL_RETURN(ghost_cu_free(cucolfield));
    GHOST_CALL_RETURN(ghost_cu_free(curowfield));
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_rm_cu_scale(ghost_densemat_t *vec, void *a)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret = GHOST_SUCCESS;
    
    char colfield[vec->traits.ncolsorig];
    char rowfield[vec->traits.nrowsorig];

    char *cucolfield = NULL, *curowfield = NULL;
    int grid = (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK);
    dim3 block (THREADSPERBLOCK/vec->traits.ncolsorig,vec->traits.ncolsorig);

    if (ghost_bitmap_weight(vec->ldmask) != vec->traits.ncolsorig || 
            ghost_bitmap_weight(vec->trmask) != vec->traits.nrowsorig) { 
        WARNING_LOG("Potentially slow SCAL operation because some rows or columns are masked out!");
        
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&cucolfield,vec->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&curowfield,vec->traits.nrowsorig),err,ret);

        ghost_densemat_mask2charfield(vec->ldmask,vec->traits.ncolsorig,colfield);
        ghost_densemat_mask2charfield(vec->trmask,vec->traits.nrowsorig,rowfield);

        GHOST_CALL_GOTO(ghost_cu_upload(cucolfield,colfield,vec->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_upload(curowfield,rowfield,vec->traits.nrowsorig),err,ret);
    }


    if (vec->traits.datatype & GHOST_DT_COMPLEX)
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_scale_kernel<cuDoubleComplex><<< grid,block >>>(
                    (cuDoubleComplex *)vec->cu_val, *(cuDoubleComplex *)a,
                    vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->stride);
        } 
        else 
        {
            cu_scale_kernel<cuFloatComplex><<< grid,block >>>(
                    (cuFloatComplex *)vec->cu_val, *(cuFloatComplex *)a,
                    vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->stride);
        }
    }
    else
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_scale_kernel<double><<< grid,block >>>(
                    (double *)vec->cu_val, *(double *)a,
                    vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->stride);
        } 
        else 
        {
            cu_scale_kernel<float><<< grid,block >>>(
                    (float *)vec->cu_val, *(float *)a,
                    vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->stride);
        }
    }
    goto out;

err:

out:
    GHOST_CALL_RETURN(ghost_cu_free(cucolfield));
    GHOST_CALL_RETURN(ghost_cu_free(curowfield));
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    
    return ret;
}

extern "C" ghost_error_t ghost_densemat_rm_cu_vscale(ghost_densemat_t *vec, void *a)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret = GHOST_SUCCESS;

    void *d_a;
    size_t sizeofdt;
    char colfield[vec->traits.ncolsorig];
    char rowfield[vec->traits.nrowsorig];
    ghost_idx_t c,v=0;

    char *cucolfield = NULL, *curowfield = NULL;
    int grid = (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK);
    dim3 block (THREADSPERBLOCK/vec->traits.ncolsorig,vec->traits.ncolsorig); 
    
    ghost_datatype_size(&sizeofdt,vec->traits.datatype);
    GHOST_CALL_GOTO(ghost_cu_malloc(&d_a,vec->traits.ncolsorig*sizeofdt),err,ret);
    GHOST_CALL_GOTO(ghost_cu_memset(d_a,0,vec->traits.ncolsorig*sizeofdt),err,ret);
    
    for (c=0; c<vec->traits.ncolsorig; c++) {
        if (ghost_bitmap_isset(vec->ldmask,c)) {
            GHOST_CALL_GOTO(ghost_cu_upload(&((char *)d_a)[c*sizeofdt],&((char *)a)[v*sizeofdt],sizeofdt),err,ret);
            v++;
        }
    }
    
    if (ghost_bitmap_weight(vec->ldmask) != vec->traits.ncolsorig || 
            ghost_bitmap_weight(vec->trmask) != vec->traits.nrowsorig) { 
        
        WARNING_LOG("Potentially slow VSCALE operation because some rows or columns are masked out!");
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&cucolfield,vec->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&curowfield,vec->traits.nrowsorig),err,ret);

        ghost_densemat_mask2charfield(vec->ldmask,vec->traits.ncolsorig,colfield);
        ghost_densemat_mask2charfield(vec->trmask,vec->traits.nrowsorig,rowfield);

        GHOST_CALL_GOTO(ghost_cu_upload(cucolfield,colfield,vec->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_upload(curowfield,rowfield,vec->traits.nrowsorig),err,ret);
    }


    if (vec->traits.datatype & GHOST_DT_COMPLEX)
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_vscale_kernel<cuDoubleComplex><<< grid,block >>>(
                    (cuDoubleComplex *)vec->cu_val, (cuDoubleComplex *)d_a,
                    vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->stride);
        } 
        else 
        {
            cu_vscale_kernel<cuFloatComplex><<< grid,block >>>(
                    (cuFloatComplex *)vec->cu_val, (cuFloatComplex *)d_a,
                    vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->stride);
        }
    }
    else
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_vscale_kernel<double><<< grid,block >>>(
                    (double *)vec->cu_val, (double *)d_a,
                    vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->stride);
        } 
        else 
        {
            cu_vscale_kernel<float><<< grid,block >>>(
                    (float *)vec->cu_val, (float *)d_a,
                    vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->stride);
        }
    }

    goto out;
err:
out:
    GHOST_CALL_RETURN(ghost_cu_free(cucolfield));
    GHOST_CALL_RETURN(ghost_cu_free(curowfield));
    GHOST_CALL_RETURN(ghost_cu_free(d_a));
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_rm_cu_fromScalar(ghost_densemat_t *vec, void *a)
{
    ghost_error_t ret = GHOST_SUCCESS;
    
    char colfield[vec->traits.ncolsorig];
    char rowfield[vec->traits.nrowsorig];

    char *cucolfield = NULL, *curowfield = NULL;
    int grid = (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK);
    dim3 block (THREADSPERBLOCK/vec->traits.ncolsorig,vec->traits.ncolsorig); 

    if (ghost_bitmap_weight(vec->ldmask) != vec->traits.ncolsorig || 
            ghost_bitmap_weight(vec->trmask) != vec->traits.nrowsorig) { 
        
        WARNING_LOG("Potentially slow fromScalar operation because some rows or columns are masked out!");
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&cucolfield,vec->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&curowfield,vec->traits.nrowsorig),err,ret);

        ghost_densemat_mask2charfield(vec->ldmask,vec->traits.ncolsorig,colfield);
        ghost_densemat_mask2charfield(vec->trmask,vec->traits.nrowsorig,rowfield);

        GHOST_CALL_GOTO(ghost_cu_upload(cucolfield,colfield,vec->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_upload(curowfield,rowfield,vec->traits.nrowsorig),err,ret);
    }
    
    ghost_densemat_rm_malloc(vec);
    if (vec->traits.datatype & GHOST_DT_COMPLEX)
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_fromscalar_kernel<cuDoubleComplex><<< grid,block >>>(
                    (cuDoubleComplex *)vec->cu_val, *(cuDoubleComplex *)a,
                    vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->stride);
        } 
        else 
        {
            cu_fromscalar_kernel<cuFloatComplex><<< grid,block >>>(
                    (cuFloatComplex *)vec->cu_val, *(cuFloatComplex *)a,
                    vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->stride);
        }
    }
    else
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_fromscalar_kernel<double><<< grid,block >>>(
                    (double *)vec->cu_val, *(double *)a,
                    vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->stride);
        } 
        else 
        {
            cu_fromscalar_kernel<float><<< grid,block >>>(
                    (float *)vec->cu_val, *(float *)a,
                    vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->stride);
        }
    }
    
    goto out;
err:
out:
    GHOST_CALL_RETURN(ghost_cu_free(cucolfield));
    GHOST_CALL_RETURN(ghost_cu_free(curowfield));

    return ret;
}

extern "C" ghost_error_t ghost_densemat_rm_cu_fromRand(ghost_densemat_t *vec)
{
    ghost_error_t ret = GHOST_SUCCESS;

    ghost_densemat_t *onevec, *onevecview;
    long pid = getpid();
    double time;
    double one[] = {1.,1.};
    float fone[] = {1.,0.};
    double minusahalf[] = {-0.5,0.};
    float fminusahalf[] = {-0.5,0.};
    
    ghost_timing_wcmilli(&time);
    ghost_densemat_rm_malloc(vec);
    curandGenerator_t gen;
    CURAND_CALL_GOTO(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT),err,ret);
    CURAND_CALL_GOTO(curandSetPseudoRandomGeneratorSeed(gen,ghost_hash(int(time),clock(),pid)),err,ret);

    vec->clone(vec,&onevec,vec->traits.nrowsorig,0,vec->traits.ncolsorig,0);
    onevec->fromScalar(onevec,one);
    onevec->viewVec(onevec,&onevecview,vec->traits.nrows,ghost_bitmap_first(vec->trmask),vec->traits.ncols,ghost_bitmap_first(vec->ldmask));

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
                        (double *)(valptr),
                        compactvec->traits.ncolsorig*compactvec->traits.nrows*2),err,ret);
        } 
        else 
        {
            CURAND_CALL_GOTO(curandGenerateUniform(gen,
                        (float *)(valptr),
                        compactvec->traits.ncolsorig*compactvec->traits.nrows*2),err,ret);
        }
    }
    else
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            CURAND_CALL_GOTO(curandGenerateUniformDouble(gen,
                        (double *)(valptr),
                        compactvec->traits.ncolsorig*compactvec->traits.nrows),err,ret);
        } 
        else 
        {
            CURAND_CALL_GOTO(curandGenerateUniform(gen,
                        (float *)(valptr),
                        compactvec->traits.ncolsorig*compactvec->traits.nrows),err,ret);
        }
    }
    if (compactvec->traits.datatype & GHOST_DT_DOUBLE) {
        compactvec->axpby(compactvec,onevecview,minusahalf,one);
    } else {
        compactvec->axpby(compactvec,onevecview,fminusahalf,fone);
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
#endif
