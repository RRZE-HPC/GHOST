#include "ghost/config.h"
#undef GHOST_HAVE_MPI
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/densemat_cm.h"
#include "ghost/log.h"
#include "ghost/timing.h"
#include "ghost/locality.h"
#include "ghost/instr.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <curand.h>
#include <sys/types.h>
#include <unistd.h>

#include "ghost/cu_complex.h"


#define THREADSPERBLOCK 256


template<typename T>  
__global__ static void cu_vaxpy_kernel(T *v1, T *v2, T *a, ghost_idx_t nrows, char *rowmask, ghost_idx_t ncols, char *colmask, ghost_idx_t nrowspadded)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    if (rowmask || colmask) {
        for (;idx < nrows; idx+=gridDim.x*blockDim.x) {
            if (rowmask[idx]) {
                ghost_idx_t v;
                for (v=0; v<ncols; v++) {
                    if (colmask[v]) {
                        v1[v*nrowspadded+idx] = axpy<T,T>(v1[v*nrowspadded+idx],v2[v*nrowspadded+idx],a[v]);
                    }
                }
            }
        }
    } else {
        for (;idx < nrows; idx+=gridDim.x*blockDim.x) {
            ghost_idx_t v;
            for (v=0; v<ncols; v++) {
                v1[v*nrowspadded+idx] = axpy<T,T>(v1[v*nrowspadded+idx],v2[v*nrowspadded+idx],a[v]);
            }
        }
    }

}

template<typename T>  
__global__ static void cu_vaxpby_kernel(T *v1, T *v2, T *a, T *b, ghost_idx_t nrows, char *rowmask, ghost_idx_t ncols, char *colmask, ghost_idx_t nrowspadded)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    if (rowmask || colmask) {
        for (;idx < nrows; idx+=gridDim.x*blockDim.x) {
            if (rowmask[idx]) {
                ghost_idx_t v;
                for (v=0; v<ncols; v++) {
                    if (colmask[v]) {
                        v1[v*nrowspadded+idx] = axpby<T>(v2[v*nrowspadded+idx],v1[v*nrowspadded+idx],a[v],b[v]);
                    }
                }
            }
        }
    } else {
        for (;idx < nrows; idx+=gridDim.x*blockDim.x) {
            ghost_idx_t v;
            for (v=0; v<ncols; v++) {
                v1[v*nrowspadded+idx] = axpby<T>(v2[v*nrowspadded+idx],v1[v*nrowspadded+idx],a[v],b[v]);
            }
        }
    }

}

template<typename T>  
__global__ static void cu_axpby_kernel(T *v1, T *v2, T a, T b, ghost_idx_t nrows, char *rowmask, ghost_idx_t ncols, char *colmask, ghost_idx_t nrowspadded)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    if (rowmask || colmask) {
        for (;idx < nrows; idx+=gridDim.x*blockDim.x) {
            if (rowmask[idx]) {
                ghost_idx_t v;
                for (v=0; v<ncols; v++) {
                    if (colmask[v]) {
                        v1[v*nrowspadded+idx] = axpby<T>(v2[v*nrowspadded+idx],v1[v*nrowspadded+idx],a,b);
                    }
                }
            }
        }
    } else {
        for (;idx < nrows; idx+=gridDim.x*blockDim.x) {
            ghost_idx_t v;
            for (v=0; v<ncols; v++) {
                v1[v*nrowspadded+idx] = axpby<T>(v2[v*nrowspadded+idx],v1[v*nrowspadded+idx],a,b);
            }
        }
    }
}

template<typename T>  
__global__ static void cu_scale_kernel(T *vec, T a, ghost_idx_t nrows, char *rowmask, ghost_idx_t ncols, char *colmask, ghost_idx_t nrowspadded)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    if (rowmask || colmask) {
        for (;idx < nrows; idx+=gridDim.x*blockDim.x) {
            if (rowmask[idx]) {
                ghost_idx_t v;
                for (v=0; v<ncols; v++) {
                    if (colmask[v]) {
                        vec[v*nrowspadded+idx] = scale<T>(a,vec[v*nrowspadded+idx]);
                    }
                }
            }
        }
    } else {
        for (;idx < nrows; idx+=gridDim.x*blockDim.x) {
            ghost_idx_t v;
            for (v=0; v<ncols; v++) {
                vec[v*nrowspadded+idx] = scale<T>(a,vec[v*nrowspadded+idx]);
            }
        }
    }

}

template<typename T>  
__global__ static void cu_vscale_kernel(T *vec, T *a, ghost_idx_t nrows, char *rowmask, ghost_idx_t ncols, char *colmask, ghost_idx_t nrowspadded)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    if (rowmask || colmask) {
        int c;
        for (;idx < nrows; idx+=gridDim.x*blockDim.x) {
            if (rowmask[idx]) {
                ghost_idx_t v;
                for (c=0,v=0; v<ncols; v++) {
                    if (colmask[v]) {
                        vec[v*nrowspadded+idx] = scale<T>(a[c],vec[v*nrowspadded+idx]);
                        c++;
                    }
                }
            }
        }
    } else {
        for (;idx < nrows; idx+=gridDim.x*blockDim.x) {
            ghost_idx_t v;
            for (v=0; v<ncols; v++) {
                vec[v*nrowspadded+idx] = scale<T>(a[v],vec[v*nrowspadded+idx]);
            }
        }
    }
}

template<typename T>  
__global__ static void cu_fromscalar_kernel(T *vec, T a, ghost_idx_t nrows, char *rowmask, ghost_idx_t ncols, char *colmask, ghost_idx_t nrowspadded)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    if (rowmask || colmask) {
        for (;idx < nrows; idx+=gridDim.x*blockDim.x) {
            if (rowmask[idx]) {
                ghost_idx_t v;
                for (v=0; v<ncols; v++) {
                    if (colmask[v]) {
                        vec[v*nrowspadded+idx] = a;
                    }
                }
            }
        }
    } else {
        for (;idx < nrows; idx+=gridDim.x*blockDim.x) {
            ghost_idx_t v;
            for (v=0; v<ncols; v++) {
                vec[v*nrowspadded+idx] = a;
            }
        }
    }
}

template<typename T>  
__global__ static void cu_communicationassembly_kernel(T *vec, T *work, ghost_idx_t offs, ghost_idx_t *duelist, ghost_idx_t ncols, ghost_idx_t ndues, ghost_idx_t nrowspadded, ghost_idx_t *perm)
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

extern "C" ghost_error_t ghost_densemat_cm_cu_communicationassembly(void * work, ghost_idx_t *dueptr, ghost_densemat_t *vec, ghost_idx_t *perm)
{
   

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
                    cu_communicationassembly_kernel<cuDoubleComplex><<< (int)ceil((double)ctx->dues[proc]/THREADSPERBLOCK)*vec->traits.ncols,block >>>((cuDoubleComplex *)vec->cu_val, ((cuDoubleComplex *)work),dueptr[proc],ctx->cu_duelist[proc],vec->traits.ncols,ctx->dues[proc],vec->traits.nrowspadded,perm);
                }
            } 
            else 
            {
                if (ctx->dues[proc]) {
                    cu_communicationassembly_kernel<cuFloatComplex><<< (int)ceil((double)ctx->dues[proc]/THREADSPERBLOCK)*vec->traits.ncols,block >>>((cuFloatComplex *)vec->cu_val, ((cuFloatComplex *)work),dueptr[proc],ctx->cu_duelist[proc],vec->traits.ncols,ctx->dues[proc],vec->traits.nrowspadded,perm);
                }
            }
        }
        else
        {
            if (vec->traits.datatype & GHOST_DT_DOUBLE)
            {
                if (ctx->dues[proc]) {
                    cu_communicationassembly_kernel<double><<< (int)ceil((double)ctx->dues[proc]/THREADSPERBLOCK)*vec->traits.ncols,block >>>((double *)vec->cu_val, ((double *)work),dueptr[proc],ctx->cu_duelist[proc],vec->traits.ncols,ctx->dues[proc],vec->traits.nrowspadded,perm);
                }
            } 
            else 
            {
                if (ctx->dues[proc]) {
                    cu_communicationassembly_kernel<float><<< (int)ceil((double)ctx->dues[proc]/THREADSPERBLOCK)*vec->traits.ncols,block >>>((float *)vec->cu_val, ((float *)work),dueptr[proc],ctx->cu_duelist[proc],vec->traits.ncols,ctx->dues[proc],vec->traits.nrowspadded,perm);
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
    GHOST_INSTR_START(vaxpy);
    ghost_error_t ret = GHOST_SUCCESS;
    void *d_a;
    size_t sizeofdt;
    char colfield[v1->traits.ncolsorig];
    char rowfield[v1->traits.nrowsorig];
    char *cucolfield = NULL, *curowfield = NULL;
    ghost_datatype_size(&sizeofdt,v1->traits.datatype);
    
    GHOST_CALL_GOTO(ghost_cu_malloc(&d_a,v1->traits.ncols*sizeofdt),err,ret);
    
    ghost_cu_upload(d_a,a,v1->traits.ncols*sizeofdt);
    if (v1->traits.datatype != v2->traits.datatype)
    {
        ERROR_LOG("Cannot VAXPY vectors with different data types");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }

    
    if (!ghost_bitmap_iscompact(v1->ldmask) || 
            !ghost_bitmap_iscompact(v1->trmask) || 
            !ghost_bitmap_iscompact(v2->ldmask) || 
            !ghost_bitmap_iscompact(v2->trmask)) {
        WARNING_LOG("Potentially slow VAXPY operation because some rows or columns are masked out!");
        
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&cucolfield,v1->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&curowfield,v1->traits.nrowsorig),err,ret);

        ghost_densemat_mask2charfield(v1->trmask,v1->traits.ncolsorig,colfield);
        ghost_densemat_mask2charfield(v1->ldmask,v1->traits.nrowsorig,rowfield);

        GHOST_CALL_GOTO(ghost_cu_upload(cucolfield,colfield,v1->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_upload(curowfield,rowfield,v1->traits.nrowsorig),err,ret);
    }

    if (v1->traits.datatype & GHOST_DT_COMPLEX)
    {
        if (v1->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_vaxpy_kernel<cuDoubleComplex><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>((cuDoubleComplex *)v1->cu_val, (cuDoubleComplex *)v2->cu_val,(cuDoubleComplex *)d_a,v1->traits.nrowsorig,curowfield,v1->traits.ncolsorig,cucolfield,v1->traits.nrowspadded);
        } 
        else 
        {
            cu_vaxpy_kernel<cuFloatComplex><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>((cuFloatComplex *)v1->cu_val, (cuFloatComplex *)v2->cu_val,(cuFloatComplex *)d_a,v1->traits.nrowsorig,curowfield,v1->traits.ncolsorig,cucolfield,v1->traits.nrowspadded);
        }
    }
    else
    {
        if (v1->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_vaxpy_kernel<double><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>((double *)v1->cu_val, (double *)v2->cu_val,(double *)d_a,v1->traits.nrowsorig,curowfield,v1->traits.ncolsorig,cucolfield,v1->traits.nrowspadded);
        } 
        else 
        {
            cu_vaxpy_kernel<float><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>((float *)v1->cu_val, (float *)v2->cu_val,(float *)d_a,v1->traits.nrowsorig,curowfield,v1->traits.ncolsorig,cucolfield,v1->traits.nrowspadded);
        }
    }
    
    goto out;
err:
out:
    GHOST_CALL_RETURN(ghost_cu_free(cucolfield));
    GHOST_CALL_RETURN(ghost_cu_free(curowfield));
    GHOST_CALL_RETURN(ghost_cu_free(d_a));
    GHOST_INSTR_STOP(vaxpy);

    return ret;
}
    
extern "C" ghost_error_t ghost_densemat_cm_cu_vaxpby(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a, void *b)
{
    GHOST_INSTR_START(vaxpby);
    ghost_error_t ret = GHOST_SUCCESS;

    void *d_a;
    void *d_b;
    size_t sizeofdt;
    char colfield[v1->traits.ncolsorig];
    char rowfield[v1->traits.nrowsorig];

    char *cucolfield = NULL, *curowfield = NULL;
    
    ghost_datatype_size(&sizeofdt,v1->traits.datatype);
    
    GHOST_CALL_GOTO(ghost_cu_malloc(&d_a,v1->traits.ncols*sizeofdt),err,ret);
    GHOST_CALL_GOTO(ghost_cu_malloc(&d_b,v1->traits.ncols*sizeofdt),err,ret);
    
    ghost_cu_upload(d_b,b,v1->traits.ncols*sizeofdt);
    
    if (v1->traits.datatype != v2->traits.datatype)
    {
        ERROR_LOG("Cannot VAXPBY vectors with different data types");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
    
    if (!ghost_bitmap_iscompact(v1->ldmask) || 
            !ghost_bitmap_iscompact(v1->trmask) || 
            !ghost_bitmap_iscompact(v2->ldmask) || 
            !ghost_bitmap_iscompact(v2->trmask)) {
        
        WARNING_LOG("Potentially slow VAXPBY operation because some rows or columns are masked out!");
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&cucolfield,v1->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&curowfield,v1->traits.nrowsorig),err,ret);

        ghost_densemat_mask2charfield(v1->trmask,v1->traits.ncolsorig,colfield);
        ghost_densemat_mask2charfield(v1->ldmask,v1->traits.nrowsorig,rowfield);

        GHOST_CALL_GOTO(ghost_cu_upload(cucolfield,colfield,v1->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_upload(curowfield,rowfield,v1->traits.nrowsorig),err,ret);
    }

    if (v1->traits.datatype & GHOST_DT_COMPLEX)
    {
        if (v1->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_vaxpby_kernel<cuDoubleComplex><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                (cuDoubleComplex *)v1->cu_val, (cuDoubleComplex *)v2->cu_val,(cuDoubleComplex *)d_a,(cuDoubleComplex *)d_b,
                 v1->traits.nrowsorig,curowfield,v1->traits.ncolsorig,cucolfield,v1->traits.nrowspadded);
        } 
        else 
        {
            cu_vaxpby_kernel<cuFloatComplex><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                (cuFloatComplex *)v1->cu_val, (cuFloatComplex *)v2->cu_val,(cuFloatComplex *)d_a,(cuFloatComplex *)d_b,
                 v1->traits.nrowsorig,curowfield,v1->traits.ncolsorig,cucolfield,v1->traits.nrowspadded);
        }
    }
    else
    {
        if (v1->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_vaxpby_kernel<double><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                 (double *)v1->cu_val, (double *)v2->cu_val,(double *)d_a,(double *)d_b,
                 v1->traits.nrowsorig,curowfield,v1->traits.ncolsorig,cucolfield,v1->traits.nrowspadded);
        } 
        else 
        {
            cu_vaxpby_kernel<float><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                (float *)v1->cu_val, (float *)v2->cu_val,(float *)d_a,(float *)d_b,
                 v1->traits.nrowsorig,curowfield,v1->traits.ncolsorig,cucolfield,v1->traits.nrowspadded);
        }
    }
    goto out;
err:
out:
    GHOST_CALL_RETURN(ghost_cu_free(cucolfield));
    GHOST_CALL_RETURN(ghost_cu_free(curowfield));
    GHOST_CALL_RETURN(ghost_cu_free(d_a));
    GHOST_CALL_RETURN(ghost_cu_free(d_a));
    GHOST_INSTR_STOP(vaxpby);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_cm_cu_dotprod(ghost_densemat_t *vec, void *res, ghost_densemat_t *vec2)
{
    GHOST_INSTR_START(dot);
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

    if (!ghost_bitmap_iscompact(vec->ldmask) || 
            !ghost_bitmap_iscompact(vec->trmask)) {
        INFO_LOG("Cloning (and compressing) vec1 before dotproduct");
        vec->clone(vec,&vecclone,vec->traits.nrows,0,vec->traits.ncols,0);
    } else {
        vecclone = vec;
    }
    if (!ghost_bitmap_iscompact(vec2->ldmask) || 
            !ghost_bitmap_iscompact(vec2->trmask)) {
        INFO_LOG("Cloning (and compressing) vec1 before dotproduct");
        vec2->clone(vec2,&vec2clone,vec2->traits.nrows,0,vec2->traits.ncols,0);
    } else {
        vec2clone = vec2;
    }
  
     
    cublasHandle_t ghost_cublas_handle;
    GHOST_CALL_GOTO(ghost_cu_cublas_handle(&ghost_cublas_handle),err,ret); 
    ghost_idx_t v;
    for (v=0; v<vecclone->traits.ncols; v++)
    {
        char *v1 = &((char *)(vecclone->cu_val))[v*vecclone->traits.nrowspadded*sizeofdt];
        char *v2 = &((char *)(vec2clone->cu_val))[v*vec2clone->traits.nrowspadded*sizeofdt];
        if (vec->traits.datatype & GHOST_DT_COMPLEX)
        {
            if (vec->traits.datatype & GHOST_DT_DOUBLE)
            {
                CUBLAS_CALL_GOTO(cublasZdotc(ghost_cublas_handle,vec->traits.nrows,
                            (const cuDoubleComplex *)v1,1,(const cuDoubleComplex *)v2,1,&((cuDoubleComplex *)res)[v]),err,ret);
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
    GHOST_INSTR_STOP(dot);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_cm_cu_axpy(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *a)
{
    if (vec->traits.datatype != vec2->traits.datatype)
    {
        ERROR_LOG("Cannot AXPY vectors with different data types");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
    GHOST_INSTR_START(axpy);
    ghost_error_t ret = GHOST_SUCCESS;
    
    char colfield[vec->traits.ncolsorig];
    char rowfield[vec->traits.nrowsorig];

    char *cucolfield = NULL, *curowfield = NULL;

    if (!ghost_bitmap_iscompact(vec->ldmask) || 
            !ghost_bitmap_iscompact(vec->trmask) || 
            !ghost_bitmap_iscompact(vec2->ldmask) || 
            !ghost_bitmap_iscompact(vec2->trmask)) {
        WARNING_LOG("Potentially slow AXPY operation because some rows or columns are masked out!");
        
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&cucolfield,vec->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&curowfield,vec->traits.nrowsorig),err,ret);

        ghost_densemat_mask2charfield(vec->trmask,vec->traits.ncolsorig,colfield);
        ghost_densemat_mask2charfield(vec->ldmask,vec->traits.nrowsorig,rowfield);

        GHOST_CALL_GOTO(ghost_cu_upload(cucolfield,colfield,vec->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_upload(curowfield,rowfield,vec->traits.nrowsorig),err,ret);
    }

    
    if (vec->traits.datatype & GHOST_DT_COMPLEX)
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            const cuDoubleComplex one = make_cuDoubleComplex(1.,1.);
            cu_axpby_kernel<cuDoubleComplex><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>
                ((cuDoubleComplex *)vec->cu_val, (cuDoubleComplex *)vec2->cu_val,*((cuDoubleComplex *)a),one,
                 vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->traits.nrowspadded);
        } 
        else 
        {
            const cuFloatComplex one = make_cuFloatComplex(1.,1.);
            cu_axpby_kernel<cuFloatComplex><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>
                ((cuFloatComplex *)vec->cu_val, (cuFloatComplex *)vec2->cu_val,*((cuFloatComplex *)a),one,
                 vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->traits.nrowspadded);
        }
    }
    else
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_axpby_kernel<double><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>
                ((double *)vec->cu_val, (double *)vec2->cu_val,*((double *)a),(double)1.,
                 vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->traits.nrowspadded);
        } 
        else 
        {
            cu_axpby_kernel<float><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>
                ((float *)vec->cu_val, (float *)vec2->cu_val,*((float *)a),(float)1.,
                 vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->traits.nrowspadded);
        }
    }

    
    /*cublasHandle_t ghost_cublas_handle;
    GHOST_CALL_RETURN(ghost_cu_cublas_handle(&ghost_cublas_handle)); 
    if (vec->traits.datatype & GHOST_DT_COMPLEX)
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            CUBLAS_CALL_RETURN(cublasZaxpy(ghost_cublas_handle,vec->traits.nrows,
                        (const cuDoubleComplex *)a,
                        (const cuDoubleComplex *)vec2->cu_val,1,
                        (cuDoubleComplex *)vec->cu_val,1));
        } 
        else 
        {
            CUBLAS_CALL_RETURN(cublasCaxpy(ghost_cublas_handle,vec->traits.nrows,
                        (const cuFloatComplex *)a,
                        (const cuFloatComplex *)vec2->cu_val,1,
                        (cuFloatComplex *)vec->cu_val,1));
        }
    }
    else
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            CUBLAS_CALL_RETURN(cublasDaxpy(ghost_cublas_handle,vec->traits.nrows,
                        (const double *)a,
                        (const double *)vec2->cu_val,1,
                        (double *)vec->cu_val,1));
        } 
        else 
        {
            CUBLAS_CALL_RETURN(cublasSaxpy(ghost_cublas_handle,vec->traits.nrows,
                        (const float *)a,
                        (const float *)vec2->cu_val,1,
                        (float *)vec->cu_val,1));
        }
    }*/
    goto out;
err:
out:
    
    GHOST_CALL_RETURN(ghost_cu_free(cucolfield));
    GHOST_CALL_RETURN(ghost_cu_free(curowfield));
    GHOST_INSTR_STOP(axpy)

    return ret;
}

extern "C" ghost_error_t ghost_densemat_cm_cu_axpby(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a, void *b)
{
    if (v1->traits.datatype != v2->traits.datatype)
    {
        ERROR_LOG("Cannot AXPBY vectors with different data types");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
    GHOST_INSTR_START(axpby);
    ghost_error_t ret = GHOST_SUCCESS;

    char colfield[v1->traits.ncolsorig];
    char rowfield[v1->traits.nrowsorig];

    char *cucolfield = NULL, *curowfield = NULL;
    
    if (!ghost_bitmap_iscompact(v1->ldmask) || 
            !ghost_bitmap_iscompact(v1->trmask) || 
            !ghost_bitmap_iscompact(v2->ldmask) || 
            !ghost_bitmap_iscompact(v2->trmask)) {
        
        WARNING_LOG("Potentially slow AXPBY operation because some rows or columns are masked out!");
        
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&cucolfield,v1->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&curowfield,v1->traits.nrowsorig),err,ret);

        ghost_densemat_mask2charfield(v1->trmask,v1->traits.ncolsorig,colfield);
        ghost_densemat_mask2charfield(v1->ldmask,v1->traits.nrowsorig,rowfield);

        GHOST_CALL_GOTO(ghost_cu_upload(cucolfield,colfield,v1->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_upload(curowfield,rowfield,v1->traits.nrowsorig),err,ret);
    }

    if (v1->traits.datatype & GHOST_DT_COMPLEX)
    {
        if (v1->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_axpby_kernel<cuDoubleComplex><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>
                ((cuDoubleComplex *)v1->cu_val, (cuDoubleComplex *)v2->cu_val,*((cuDoubleComplex *)a),*((cuDoubleComplex *)b),
                 v1->traits.nrowsorig,curowfield,v1->traits.ncolsorig,cucolfield,v1->traits.nrowspadded);
        } 
        else 
        {
            cu_axpby_kernel<cuFloatComplex><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>
                ((cuFloatComplex *)v1->cu_val, (cuFloatComplex *)v2->cu_val,*((cuFloatComplex *)a),*((cuFloatComplex *)b),
                 v1->traits.nrowsorig,curowfield,v1->traits.ncolsorig,cucolfield,v1->traits.nrowspadded);
        }
    }
    else
    {
        if (v1->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_axpby_kernel<double><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>
                ((double *)v1->cu_val, (double *)v2->cu_val,*((double *)a),*((double *)b),
                 v1->traits.nrowsorig,curowfield,v1->traits.ncolsorig,cucolfield,v1->traits.nrowspadded);
        } 
        else 
        {
            cu_axpby_kernel<float><<< (int)ceil((double)v1->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>
                ((float *)v1->cu_val, (float *)v2->cu_val,*((float *)a),*((float *)b),
                 v1->traits.nrowsorig,curowfield,v1->traits.ncolsorig,cucolfield,v1->traits.nrowspadded);
        }
    }
    
    goto out;
err:
out:
    GHOST_CALL_RETURN(ghost_cu_free(cucolfield));
    GHOST_CALL_RETURN(ghost_cu_free(curowfield));
    GHOST_INSTR_STOP(axpby);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_cm_cu_scale(ghost_densemat_t *vec, void *a)
{
    GHOST_INSTR_START(scale);
    ghost_error_t ret = GHOST_SUCCESS;
    
    char colfield[vec->traits.ncolsorig];
    char rowfield[vec->traits.nrowsorig];

    char *cucolfield = NULL, *curowfield = NULL;
    
    if (!ghost_bitmap_iscompact(vec->ldmask) || 
            !ghost_bitmap_iscompact(vec->trmask)) { 
        WARNING_LOG("Potentially slow SCAL operation because some rows or columns are masked out!");
        
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&cucolfield,vec->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&curowfield,vec->traits.nrowsorig),err,ret);

        ghost_densemat_mask2charfield(vec->trmask,vec->traits.ncolsorig,colfield);
        ghost_densemat_mask2charfield(vec->ldmask,vec->traits.nrowsorig,rowfield);

        GHOST_CALL_GOTO(ghost_cu_upload(cucolfield,colfield,vec->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_upload(curowfield,rowfield,vec->traits.nrowsorig),err,ret);
    }

    if (vec->traits.datatype & GHOST_DT_COMPLEX)
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_scale_kernel<cuDoubleComplex><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (cuDoubleComplex *)vec->cu_val, *(cuDoubleComplex *)a,
                    vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->traits.nrowspadded);
        } 
        else 
        {
            cu_scale_kernel<cuFloatComplex><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (cuFloatComplex *)vec->cu_val, *(cuFloatComplex *)a,
                    vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->traits.nrowspadded);
        }
    }
    else
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_scale_kernel<double><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (double *)vec->cu_val, *(double *)a,
                    vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->traits.nrowspadded);
        } 
        else 
        {
            cu_scale_kernel<float><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (float *)vec->cu_val, *(float *)a,
                    vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->traits.nrowspadded);
        }
    }
    goto out;

err:

out:
    GHOST_CALL_RETURN(ghost_cu_free(cucolfield));
    GHOST_CALL_RETURN(ghost_cu_free(curowfield));
    GHOST_INSTR_STOP(scale);

    
    return ret;
}

extern "C" ghost_error_t ghost_densemat_cm_cu_vscale(ghost_densemat_t *vec, void *a)
{
    GHOST_INSTR_START(vscale);
    ghost_error_t ret = GHOST_SUCCESS;

    void *d_a;
    size_t sizeofdt;
    char colfield[vec->traits.ncolsorig];
    char rowfield[vec->traits.nrowsorig];

    char *cucolfield = NULL, *curowfield = NULL;
    
    ghost_datatype_size(&sizeofdt,vec->traits.datatype);
    GHOST_CALL_GOTO(ghost_cu_malloc(&d_a,vec->traits.ncols*sizeofdt),err,ret);
    ghost_cu_upload(d_a,a,vec->traits.ncols*sizeofdt);
    
    if (!ghost_bitmap_iscompact(vec->ldmask) || 
            !ghost_bitmap_iscompact(vec->trmask)) {
        
        WARNING_LOG("Potentially slow VSCALE operation because some rows or columns are masked out!");
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&cucolfield,vec->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&curowfield,vec->traits.nrowsorig),err,ret);

        ghost_densemat_mask2charfield(vec->trmask,vec->traits.ncolsorig,colfield);
        ghost_densemat_mask2charfield(vec->ldmask,vec->traits.nrowsorig,rowfield);

        GHOST_CALL_GOTO(ghost_cu_upload(cucolfield,colfield,vec->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_upload(curowfield,rowfield,vec->traits.nrowsorig),err,ret);
    }

    if (vec->traits.datatype & GHOST_DT_COMPLEX)
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_vscale_kernel<cuDoubleComplex><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (cuDoubleComplex *)vec->cu_val, (cuDoubleComplex *)d_a,
                    vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->traits.nrowspadded);
        } 
        else 
        {
            cu_vscale_kernel<cuFloatComplex><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (cuFloatComplex *)vec->cu_val, (cuFloatComplex *)d_a,
                    vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->traits.nrowspadded);
        }
    }
    else
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_vscale_kernel<double><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (double *)vec->cu_val, (double *)d_a,
                    vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->traits.nrowspadded);
        } 
        else 
        {
            cu_vscale_kernel<float><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (float *)vec->cu_val, (float *)d_a,
                    vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->traits.nrowspadded);
        }
    }

    goto out;
err:
out:
    GHOST_CALL_RETURN(ghost_cu_free(cucolfield));
    GHOST_CALL_RETURN(ghost_cu_free(curowfield));
    GHOST_CALL_RETURN(ghost_cu_free(d_a));
    GHOST_INSTR_STOP(vscale)

    return ret;
}

extern "C" ghost_error_t ghost_densemat_cm_cu_fromScalar(ghost_densemat_t *vec, void *a)
{
    ghost_error_t ret = GHOST_SUCCESS;
    
    char colfield[vec->traits.ncolsorig];
    char rowfield[vec->traits.nrowsorig];

    char *cucolfield = NULL, *curowfield = NULL;

    if (!ghost_bitmap_iscompact(vec->ldmask) || 
            !ghost_bitmap_iscompact(vec->trmask)) {
        
        WARNING_LOG("Potentially slow fromScalar operation because some rows or columns are masked out!");
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&cucolfield,vec->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&curowfield,vec->traits.nrowsorig),err,ret);

        ghost_densemat_mask2charfield(vec->trmask,vec->traits.ncolsorig,colfield);
        ghost_densemat_mask2charfield(vec->ldmask,vec->traits.nrowsorig,rowfield);

        GHOST_CALL_GOTO(ghost_cu_upload(cucolfield,colfield,vec->traits.ncolsorig),err,ret);
        GHOST_CALL_GOTO(ghost_cu_upload(curowfield,rowfield,vec->traits.nrowsorig),err,ret);
    }
    
    ghost_densemat_cm_malloc(vec);
    if (vec->traits.datatype & GHOST_DT_COMPLEX)
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_fromscalar_kernel<cuDoubleComplex><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (cuDoubleComplex *)vec->cu_val, *(cuDoubleComplex *)a,
                    vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->traits.nrowspadded);
        } 
        else 
        {
            cu_fromscalar_kernel<cuFloatComplex><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (cuFloatComplex *)vec->cu_val, *(cuFloatComplex *)a,
                    vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->traits.nrowspadded);
        }
    }
    else
    {
        if (vec->traits.datatype & GHOST_DT_DOUBLE)
        {
            cu_fromscalar_kernel<double><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (double *)vec->cu_val, *(double *)a,
                    vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->traits.nrowspadded);
        } 
        else 
        {
            cu_fromscalar_kernel<float><<< (int)ceil((double)vec->traits.nrows/THREADSPERBLOCK),THREADSPERBLOCK >>>(
                    (float *)vec->cu_val, *(float *)a,
                    vec->traits.nrowsorig,curowfield,vec->traits.ncolsorig,cucolfield,vec->traits.nrowspadded);
        }
    }
    
    goto out;
err:
out:
    GHOST_CALL_RETURN(ghost_cu_free(cucolfield));
    GHOST_CALL_RETURN(ghost_cu_free(curowfield));

    return ret;
}

extern "C" ghost_error_t ghost_densemat_cm_cu_fromRand(ghost_densemat_t *vec)
{
    ghost_error_t ret = GHOST_SUCCESS;
    if (!ghost_bitmap_iscompact(vec->ldmask) || 
            !ghost_bitmap_iscompact(vec->trmask)) {
        ERROR_LOG("fromRand does currently not consider vector views!");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }

    ghost_densemat_t *onevec;
    long pid = getpid();
    double time;
    double one[] = {1.,1.};
    float fone[] = {1.,0.};
    double minusahalf[] = {-0.5,0.};
    float fminusahalf[] = {-0.5,0.};
    
    ghost_timing_wcmilli(&time);
    ghost_densemat_cm_malloc(vec);
    curandGenerator_t gen;
    CURAND_CALL_GOTO(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT),err,ret);
    CURAND_CALL_GOTO(curandSetPseudoRandomGeneratorSeed(gen,ghost_hash(int(time),clock(),pid)),err,ret);

    vec->clone(vec,&onevec,vec->traits.nrows,0,vec->traits.ncols,0);
    onevec->fromScalar(onevec,one);

    one[1] = 0.;

    ghost_idx_t v;
    for (v=0; v<vec->traits.ncols; v++)
    {
        if (vec->traits.datatype & GHOST_DT_COMPLEX)
        {
            if (vec->traits.datatype & GHOST_DT_DOUBLE)
            {
                CURAND_CALL_GOTO(curandGenerateUniformDouble(gen,
                            &((double *)(vec->cu_val))[v*vec->traits.nrowspadded],
                            vec->traits.nrows*2),err,ret);
            } 
            else 
            {
                CURAND_CALL_GOTO(curandGenerateUniform(gen,
                            &((float *)(vec->cu_val))[v*vec->traits.nrowspadded],
                            vec->traits.nrows*2),err,ret);
            }
        }
        else
        {
            if (vec->traits.datatype & GHOST_DT_DOUBLE)
            {
                CURAND_CALL_GOTO(curandGenerateUniformDouble(gen,
                            &((double *)(vec->cu_val))[v*vec->traits.nrowspadded],
                            vec->traits.nrows),err,ret);
            } 
            else 
            {
                CURAND_CALL_GOTO(curandGenerateUniform(gen,
                            &((float *)(vec->cu_val))[v*vec->traits.nrowspadded],
                            vec->traits.nrows),err,ret);
            }
        }
    }
    if (vec->traits.datatype & GHOST_DT_DOUBLE) {
        vec->axpby(vec,onevec,minusahalf,one);
    } else {
        vec->axpby(vec,onevec,fminusahalf,fone);
    }
    goto out;
err:
out:
    CURAND_CALL_RETURN(curandDestroyGenerator(gen));
    onevec->destroy(onevec);

    return ret;
}
