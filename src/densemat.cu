#include "ghost/config.h"
#undef GHOST_HAVE_MPI
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/densemat_cm.h"
#include "ghost/log.h"
#include "ghost/timing.h"

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
}

template<typename T>  
__global__ static void cu_vaxpby_kernel(T *v1, T *v2, T *a, T *b, ghost_idx_t nrows, char *rowmask, ghost_idx_t ncols, char *colmask, ghost_idx_t nrowspadded)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

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
}

template<typename T>  
__global__ static void cu_axpby_kernel(T *v1, T *v2, T a, T b, ghost_idx_t nrows, char *rowmask, ghost_idx_t ncols, char *colmask, ghost_idx_t nrowspadded)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

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
}

template<typename T>  
__global__ static void cu_scale_kernel(T *vec, T a, ghost_idx_t nrows, char *rowmask, ghost_idx_t ncols, char *colmask, ghost_idx_t nrowspadded)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

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
}

template<typename T>  
__global__ static void cu_vscale_kernel(T *vec, T *a, ghost_idx_t nrows, char *rowmask, ghost_idx_t ncols, char *colmask, ghost_idx_t nrowspadded)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
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
}

template<typename T>  
__global__ static void cu_fromscalar_kernel(T *vec, T a, ghost_idx_t nrows, char *rowmask, ghost_idx_t ncols, char *colmask, ghost_idx_t nrowspadded)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

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
}

extern "C" ghost_error_t ghost_densemat_cm_cu_vaxpy(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a)
{
    void *d_a;
    size_t sizeofdt;
    ghost_datatype_size(&sizeofdt,v1->traits.datatype);
    GHOST_CALL_RETURN(ghost_cu_malloc(&d_a,v1->traits.ncols*sizeofdt));
    ghost_cu_upload(d_a,a,v1->traits.ncols*sizeofdt);
    if (v1->traits.datatype != v2->traits.datatype)
    {
        ERROR_LOG("Cannot VAXPY vectors with different data types");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
    char colfield[v1->traits.ncolsorig];
    char rowfield[v1->traits.nrowsorig];

    char *cucolfield, *curowfield;
    GHOST_CALL_RETURN(ghost_cu_malloc((void **)&cucolfield,v1->traits.ncolsorig));
    GHOST_CALL_RETURN(ghost_cu_malloc((void **)&curowfield,v1->traits.nrowsorig));

    ghost_densemat_mask2charfield(v1->cumask,v1->traits.ncolsorig,colfield);
    ghost_densemat_mask2charfield(v1->mask,v1->traits.nrowsorig,rowfield);

    GHOST_CALL_RETURN(ghost_cu_upload(cucolfield,colfield,v1->traits.ncolsorig));
    GHOST_CALL_RETURN(ghost_cu_upload(curowfield,rowfield,v1->traits.nrowsorig));

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
    return GHOST_SUCCESS;
}
    
extern "C" ghost_error_t ghost_densemat_cm_cu_vaxpby(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a, void *b)
{
    void *d_a;
    void *d_b;
    size_t sizeofdt;
    ghost_datatype_size(&sizeofdt,v1->traits.datatype);
    GHOST_CALL_RETURN(ghost_cu_malloc(&d_a,v1->traits.ncols*sizeofdt)); //TODO goto and free
    GHOST_CALL_RETURN(ghost_cu_malloc(&d_b,v1->traits.ncols*sizeofdt));
    ghost_cu_upload(d_b,b,v1->traits.ncols*sizeofdt);
    if (v1->traits.datatype != v2->traits.datatype)
    {
        ERROR_LOG("Cannot VAXPBY vectors with different data types");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
    char colfield[v1->traits.ncolsorig];
    char rowfield[v1->traits.nrowsorig];

    char *cucolfield, *curowfield;
    GHOST_CALL_RETURN(ghost_cu_malloc((void **)&cucolfield,v1->traits.ncolsorig));
    GHOST_CALL_RETURN(ghost_cu_malloc((void **)&curowfield,v1->traits.nrowsorig));

    ghost_densemat_mask2charfield(v1->cumask,v1->traits.ncolsorig,colfield);
    ghost_densemat_mask2charfield(v1->mask,v1->traits.nrowsorig,rowfield);

    GHOST_CALL_RETURN(ghost_cu_upload(cucolfield,colfield,v1->traits.ncolsorig));
    GHOST_CALL_RETURN(ghost_cu_upload(curowfield,rowfield,v1->traits.nrowsorig));
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

    return GHOST_SUCCESS;
}

extern "C" ghost_error_t ghost_densemat_cm_cu_dotprod(ghost_densemat_t *vec, void *res, ghost_densemat_t *vec2)
{
    if (vec->traits.datatype != vec2->traits.datatype)
    {
        ERROR_LOG("Cannot DOT vectors with different data types (%s and %s)",ghost_datatype_string(vec->traits.datatype),ghost_datatype_string(vec2->traits.datatype));
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
    size_t sizeofdt;
    ghost_datatype_size(&sizeofdt,vec->traits.datatype);
    ghost_densemat_t *vecclone;
    ghost_densemat_t *vec2clone;

    WARNING_LOG("Cloning (and compressing) vectors before dotproduct");
    vec->clone(vec,&vecclone,vec->traits.nrows,0,vec->traits.ncols,0);
    vec2->clone(vec2,&vec2clone,vec2->traits.nrows,0,vec2->traits.ncols,0);
  
     
    cublasHandle_t ghost_cublas_handle;
    GHOST_CALL_RETURN(ghost_cu_cublas_handle(&ghost_cublas_handle)); 
    ghost_idx_t v;
    for (v=0; v<vecclone->traits.ncols; v++)
    {
        char *v1 = &((char *)(vecclone->cu_val))[v*vecclone->traits.nrowspadded*sizeofdt];
        char *v2 = &((char *)(vec2clone->cu_val))[v*vec2clone->traits.nrowspadded*sizeofdt];
        if (vec->traits.datatype & GHOST_DT_COMPLEX)
        {
            if (vec->traits.datatype & GHOST_DT_DOUBLE)
            {
                CUBLAS_CALL_RETURN(cublasZdotc(ghost_cublas_handle,vec->traits.nrows,
                            (const cuDoubleComplex *)v1,1,(const cuDoubleComplex *)v2,1,&((cuDoubleComplex *)res)[v]));
            } 
            else 
            {
                CUBLAS_CALL_RETURN(cublasCdotc(ghost_cublas_handle,vec->traits.nrows,
                            (const cuFloatComplex *)v1,1,(const cuFloatComplex *)v2,1,&((cuFloatComplex *)res)[v]));
            }
        }
        else
        {
            if (vec->traits.datatype & GHOST_DT_DOUBLE)
            {
                CUBLAS_CALL_RETURN(cublasDdot(ghost_cublas_handle,vec->traits.nrows,
                            (const double *)v1,1,(const double *)v2,1,&((double *)res)[v]));
            } 
            else 
            {
                CUBLAS_CALL_RETURN(cublasSdot(ghost_cublas_handle,vec->traits.nrows,
                            (const float *)v1,1,(const float *)v2,1,&((float *)res)[v]));
            }
        }
    }
    return GHOST_SUCCESS;
}

extern "C" ghost_error_t ghost_densemat_cm_cu_axpy(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *a)
{
    if (vec->traits.datatype != vec2->traits.datatype)
    {
        ERROR_LOG("Cannot AXPY vectors with different data types");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
    
    char colfield[vec->traits.ncolsorig];
    char rowfield[vec->traits.nrowsorig];

    char *cucolfield, *curowfield;
    GHOST_CALL_RETURN(ghost_cu_malloc((void **)&cucolfield,vec->traits.ncolsorig));
    GHOST_CALL_RETURN(ghost_cu_malloc((void **)&curowfield,vec->traits.nrowsorig));

    ghost_densemat_mask2charfield(vec->cumask,vec->traits.ncolsorig,colfield);
    ghost_densemat_mask2charfield(vec->mask,vec->traits.nrowsorig,rowfield);

    GHOST_CALL_RETURN(ghost_cu_upload(cucolfield,colfield,vec->traits.ncolsorig));
    GHOST_CALL_RETURN(ghost_cu_upload(curowfield,rowfield,vec->traits.nrowsorig));

    
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
    return GHOST_SUCCESS;
}

extern "C" ghost_error_t ghost_densemat_cm_cu_axpby(ghost_densemat_t *v1, ghost_densemat_t *v2, void *a, void *b)
{
    if (v1->traits.datatype != v2->traits.datatype)
    {
        ERROR_LOG("Cannot AXPY vectors with different data types");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
    char colfield[v1->traits.ncolsorig];
    char rowfield[v1->traits.nrowsorig];

    char *cucolfield, *curowfield;
    GHOST_CALL_RETURN(ghost_cu_malloc((void **)&cucolfield,v1->traits.ncolsorig));
    GHOST_CALL_RETURN(ghost_cu_malloc((void **)&curowfield,v1->traits.nrowsorig));

    ghost_densemat_mask2charfield(v1->cumask,v1->traits.ncolsorig,colfield);
    ghost_densemat_mask2charfield(v1->mask,v1->traits.nrowsorig,rowfield);

    GHOST_CALL_RETURN(ghost_cu_upload(cucolfield,colfield,v1->traits.ncolsorig));
    GHOST_CALL_RETURN(ghost_cu_upload(curowfield,rowfield,v1->traits.nrowsorig));
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

    return GHOST_SUCCESS;
}

extern "C" ghost_error_t ghost_densemat_cm_cu_scale(ghost_densemat_t *vec, void *a)
{
    char colfield[vec->traits.ncolsorig];
    char rowfield[vec->traits.nrowsorig];

    char *cucolfield, *curowfield;
    GHOST_CALL_RETURN(ghost_cu_malloc((void **)&cucolfield,vec->traits.ncolsorig));
    GHOST_CALL_RETURN(ghost_cu_malloc((void **)&curowfield,vec->traits.nrowsorig));

    ghost_densemat_mask2charfield(vec->cumask,vec->traits.ncolsorig,colfield);
    ghost_densemat_mask2charfield(vec->mask,vec->traits.nrowsorig,rowfield);

    GHOST_CALL_RETURN(ghost_cu_upload(cucolfield,colfield,vec->traits.ncolsorig));
    GHOST_CALL_RETURN(ghost_cu_upload(curowfield,rowfield,vec->traits.nrowsorig));

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

    return GHOST_SUCCESS;
}

extern "C" ghost_error_t ghost_densemat_cm_cu_vscale(ghost_densemat_t *vec, void *a)
{
    void *d_a;
    size_t sizeofdt;
    ghost_datatype_size(&sizeofdt,vec->traits.datatype);
    GHOST_CALL_RETURN(ghost_cu_malloc(&d_a,vec->traits.ncols*sizeofdt));
    ghost_cu_upload(d_a,a,vec->traits.ncols*sizeofdt);
    char colfield[vec->traits.ncolsorig];
    char rowfield[vec->traits.nrowsorig];

    char *cucolfield, *curowfield;
    GHOST_CALL_RETURN(ghost_cu_malloc((void **)&cucolfield,vec->traits.ncolsorig));
    GHOST_CALL_RETURN(ghost_cu_malloc((void **)&curowfield,vec->traits.nrowsorig));

    ghost_densemat_mask2charfield(vec->cumask,vec->traits.ncolsorig,colfield);
    ghost_densemat_mask2charfield(vec->mask,vec->traits.nrowsorig,rowfield);

    GHOST_CALL_RETURN(ghost_cu_upload(cucolfield,colfield,vec->traits.ncolsorig));
    GHOST_CALL_RETURN(ghost_cu_upload(curowfield,rowfield,vec->traits.nrowsorig));

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

    return GHOST_SUCCESS;
}

extern "C" ghost_error_t ghost_densemat_cm_cu_fromScalar(ghost_densemat_t *vec, void *a)
{
    char colfield[vec->traits.ncolsorig];
    char rowfield[vec->traits.nrowsorig];

    char *cucolfield, *curowfield;
    GHOST_CALL_RETURN(ghost_cu_malloc((void **)&cucolfield,vec->traits.ncolsorig));
    GHOST_CALL_RETURN(ghost_cu_malloc((void **)&curowfield,vec->traits.nrowsorig));

    ghost_densemat_mask2charfield(vec->cumask,vec->traits.ncolsorig,colfield);
    ghost_densemat_mask2charfield(vec->mask,vec->traits.nrowsorig,rowfield);

    GHOST_CALL_RETURN(ghost_cu_upload(cucolfield,colfield,vec->traits.ncolsorig));
    GHOST_CALL_RETURN(ghost_cu_upload(curowfield,rowfield,vec->traits.nrowsorig));
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

    return GHOST_SUCCESS;
}

extern "C" ghost_error_t ghost_densemat_cm_cu_fromRand(ghost_densemat_t *vec)
{
    long pid = getpid();
    double time;
    ghost_timing_wcmilli(&time);
    ghost_densemat_cm_malloc(vec);
    curandGenerator_t gen;
    CURAND_CALL_RETURN(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL_RETURN(curandSetPseudoRandomGeneratorSeed(gen,ghost_hash(int(time),clock(),pid)));

    ghost_idx_t v;
    for (v=0; v<vec->traits.ncols; v++)
    {
        if (vec->traits.datatype & GHOST_DT_COMPLEX)
        {
            if (vec->traits.datatype & GHOST_DT_DOUBLE)
            {
                CURAND_CALL_RETURN(curandGenerateUniformDouble(gen,
                            &((double *)(vec->cu_val))[v*vec->traits.nrowspadded],
                            vec->traits.nrows*2));
            } 
            else 
            {
                CURAND_CALL_RETURN(curandGenerateUniform(gen,
                            &((float *)(vec->cu_val))[v*vec->traits.nrowspadded],
                            vec->traits.nrows*2));
            }
        }
        else
        {
            if (vec->traits.datatype & GHOST_DT_DOUBLE)
            {
                CURAND_CALL_RETURN(curandGenerateUniformDouble(gen,
                            &((double *)(vec->cu_val))[v*vec->traits.nrowspadded],
                            vec->traits.nrows));
            } 
            else 
            {
                CURAND_CALL_RETURN(curandGenerateUniform(gen,
                            &((float *)(vec->cu_val))[v*vec->traits.nrowspadded],
                            vec->traits.nrows));
            }
        }
    }

    return GHOST_SUCCESS;
}
