#include <ghost_config.h>
#undef GHOST_HAVE_MPI
#include <ghost_types.h>
#include <ghost_sell.h>
#include <ghost_complex.h>
#include <ghost_util.h>
#include <ghost_constants.h>
#include <cuComplex.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define SELL_CUDA_NBLOCKS (int)ceil(SELL(mat)->cumat->nrowsPadded/(double)(SELL_CUDA_THREADSPERBLOCK/SELL(mat)->T)) 

extern __shared__ char shared[];
extern int ghost_cu_device;

#define CHOOSE_KERNEL(dt1,dt2) {\
    if ((SELL(mat)->T > 32) || (SELL(mat)->T == 0) || (SELL(mat)->T & (SELL(mat)->T-1)))\
        WARNING_LOG("Invalid T: %d (must be power of two <33",SELL(mat)->T);\
    if (SELL(mat)->chunkHeight == SELL(mat)->nrowsPadded) {\
        if (SELL(mat)->T > 1) {\
            INFO_LOG("ELLPACK-T kernel not available. Switching to SELL-T kernel although we have only one chunk. Performance may suffer.");\
            size_t reqSmem = ghost_sizeofDataType(lhs->traits->datatype)*SELL_CUDA_THREADSPERBLOCK;\
            struct cudaDeviceProp prop;\
            CU_safecall(cudaGetDeviceProperties(&prop,ghost_cu_device));\
            if (prop.sharedMemPerBlock < reqSmem) {\
                WARNING_LOG("Not enough shared memory available! CUDA kernel will not execute!");\
            }\
            dim3 block(SELL_CUDA_THREADSPERBLOCK/SELL(mat)->T,SELL(mat)->T);\
            SELLT_kernel_CU_tmpl\
                <dt1,dt2>\
                <<< SELL_CUDA_NBLOCKS,block,reqSmem >>>\
                ((dt2 *)lhs->CU_val[0],(dt2 *)rhs->CU_val[0],options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLenPadded,SELL(mat)->cumat->col,(dt1 *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen,SELL(mat)->chunkHeight,SELL(mat)->T);\
        } else {\
            SELL_kernel_CU_ELLPACK_tmpl\
                <dt1,dt2>\
                <<< SELL_CUDA_NBLOCKS,SELL_CUDA_THREADSPERBLOCK >>>\
                ((dt2 *)lhs->CU_val[0],(dt2 *)rhs->CU_val[0],options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(dt1 *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen,SELL(mat)->chunkHeight,SELL(mat)->T);\
        }\
    }else{\
        if (SELL(mat)->T > 1) {\
            size_t reqSmem = ghost_sizeofDataType(lhs->traits->datatype)*SELL_CUDA_THREADSPERBLOCK;\
            struct cudaDeviceProp prop;\
            CU_safecall(cudaGetDeviceProperties(&prop,ghost_cu_device));\
            if (prop.sharedMemPerBlock < reqSmem) {\
                WARNING_LOG("Not enough shared memory available! CUDA kernel will not execute!");\
            }\
            dim3 block(SELL_CUDA_THREADSPERBLOCK/SELL(mat)->T,SELL(mat)->T);\
            SELLT_kernel_CU_tmpl\
                <dt1,dt2>\
                <<< SELL_CUDA_NBLOCKS,block,reqSmem >>>\
                ((dt2 *)lhs->CU_val[0],(dt2 *)rhs->CU_val[0],options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLenPadded,SELL(mat)->cumat->col,(dt1 *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen,SELL(mat)->chunkHeight,SELL(mat)->T);\
        } else {\
            SELL_kernel_CU_tmpl\
                <dt1,dt2>\
                <<< SELL_CUDA_NBLOCKS,SELL_CUDA_THREADSPERBLOCK >>>\
                ((dt2 *)lhs->CU_val[0],(dt2 *)rhs->CU_val[0],options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(dt1 *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen,SELL(mat)->chunkHeight,SELL(mat)->T);\
        }\
    }\
}

template<typename T>
__device__ inline void zero(T &val)
{
    val = 0.;
}

template<>
__device__ inline void zero<cuFloatComplex>(cuFloatComplex &val)
{
    val = make_cuFloatComplex(0.,0.);
}

template<>
__device__ inline void zero<cuDoubleComplex>(cuDoubleComplex &val)
{
    val = make_cuDoubleComplex(0.,0.);
}

// val += val2*val3
template<typename T, typename T2>
__device__ inline T axpy(T val, T val2, T2 val3)
{
    return val+val2*val3;
}

template<>
__device__ inline cuFloatComplex axpy<cuFloatComplex,cuFloatComplex>(cuFloatComplex val, cuFloatComplex val2, cuFloatComplex val3)
{
    return cuCaddf(val,cuCmulf(val2,val3));
}

template<>
__device__ inline cuFloatComplex axpy<cuFloatComplex,double>(cuFloatComplex val, cuFloatComplex val2, double val3)
{
    return cuCaddf(val,cuCmulf(val2,make_cuFloatComplex((float)val3,0.f)));
}

template<>
__device__ inline cuFloatComplex axpy<cuFloatComplex,float>(cuFloatComplex val, cuFloatComplex val2, float val3)
{
    return cuCaddf(val,cuCmulf(val2,make_cuFloatComplex(val3,0.f)));
}

template<>
__device__ inline cuFloatComplex axpy<cuFloatComplex,cuDoubleComplex>(cuFloatComplex val, cuFloatComplex val2, cuDoubleComplex val3)
{
    return cuCaddf(val,cuCmulf(val2,make_cuFloatComplex((float)(cuCreal(val3)),(float)(cuCimag(val3)))));
}

template<>
__device__ inline cuDoubleComplex axpy<cuDoubleComplex,double>(cuDoubleComplex val, cuDoubleComplex val2, double val3)
{
    return cuCadd(val,cuCmul(val2,make_cuDoubleComplex(val3,0.)));
}

template<>
__device__ inline cuDoubleComplex axpy<cuDoubleComplex,float>(cuDoubleComplex val, cuDoubleComplex val2, float val3)
{
    return cuCadd(val,cuCmul(val2,make_cuDoubleComplex((double)val3,0.)));
}

template<>
__device__ inline cuDoubleComplex axpy<cuDoubleComplex,cuDoubleComplex>(cuDoubleComplex val, cuDoubleComplex val2, cuDoubleComplex val3)
{
    return cuCadd(val,cuCmul(val2,val3));
}

template<>
__device__ inline cuDoubleComplex axpy<cuDoubleComplex,cuFloatComplex>(cuDoubleComplex val, cuDoubleComplex val2, cuFloatComplex val3)
{
    return cuCadd(val,cuCmul(val2,make_cuDoubleComplex((double)(cuCrealf(val3)),(double)(cuCimagf(val3)))));
}

template<>
__device__ inline double axpy<double,cuFloatComplex>(double val, double val2, cuFloatComplex val3)
{
    return val+val2*(double)cuCrealf(val3);
}


template<>
__device__ inline double axpy<double,cuDoubleComplex>(double val, double val2, cuDoubleComplex val3)
{
    return val+val2*cuCreal(val3);
}

template<>
__device__ inline float axpy<float,cuFloatComplex>(float val, float val2, cuFloatComplex val3)
{
    return val+val2*cuCrealf(val3);
}


template<>
__device__ inline float axpy<float,cuDoubleComplex>(float val, float val2, cuDoubleComplex val3)
{
    return val+val2*(float)cuCreal(val3);
}

template<typename m_t, typename v_t>  
__global__ void SELL_kernel_CU_ELLPACK_tmpl(v_t *lhs, v_t *rhs, int options, int nrows, int nrowspadded, ghost_midx_t *rowlen, ghost_midx_t *col, m_t *val, ghost_mnnz_t *chunkstart, ghost_midx_t *chunklen, int C, int T)
{
    UNUSED(C);
    UNUSED(T);

    int i = threadIdx.x+blockIdx.x*blockDim.x;

    if (i<nrows) {
        int j;
        v_t tmp;
        zero<v_t>(tmp);

        for (j=0; j<rowlen[i]; j++) {
            tmp = axpy<v_t,m_t>(tmp, rhs[col[i + j*nrowspadded]], val[i + j*nrowspadded]);
        }
        if (options & GHOST_SPMVM_AXPY)
            lhs[i] = axpy<v_t,float>(lhs[i],tmp,1.f);
        else 
            lhs[i] = tmp;
    }
}

template<typename m_t, typename v_t>  
__global__ void SELL_kernel_CU_tmpl(v_t *lhs, v_t *rhs, int options, int nrows, int nrowspadded, ghost_midx_t *rowlen, ghost_midx_t *col, m_t *val, ghost_mnnz_t *chunkstart, ghost_midx_t *chunklen, int C, int T)
{
    UNUSED(T);
    int i = threadIdx.x+blockIdx.x*blockDim.x;

    if (i<nrows) {
        int cs, tid;
        if (C == blockDim.x) {
            cs = chunkstart[blockIdx.x];
            tid = threadIdx.x;
        } else {
            cs = chunkstart[i/C];
            tid = threadIdx.x%C;
        }
        int j;
        v_t tmp;
        zero<v_t>(tmp);

        for (j=0; j<rowlen[i]; j++) {
            tmp = axpy<v_t,m_t>(tmp, rhs[col[cs + tid + j*C]], val[cs + tid + j*C]);
        }
        if (options & GHOST_SPMVM_AXPY)
            lhs[i] = axpy<v_t,float>(lhs[i],tmp,1.f);
        else 
            lhs[i] = tmp;
    }
}

template<typename m_t, typename v_t>  
__global__ void SELLT_kernel_CU_tmpl(v_t *lhs, v_t *rhs, int options, ghost_midx_t nrows, ghost_midx_t nrowspadded, ghost_midx_t *rowlen, ghost_midx_t *col, m_t *val, ghost_mnnz_t *chunkstart, ghost_midx_t *chunklen, ghost_midx_t C, int T)
{
    int i = threadIdx.x+blockIdx.x*blockDim.x;

    if (i<nrows) {
        int tib = threadIdx.x*blockDim.y+threadIdx.y;
        int cs, tid; // chunk start, thread row in block
        int j;
        v_t tmp;
        v_t *smem = (v_t *)shared;
        
        if (C == blockDim.x) {
            cs = chunkstart[blockIdx.x];
            tid = threadIdx.x;
        } else {
            cs = chunkstart[i/C];
            tid = threadIdx.x%C;
        }
        zero<v_t>(tmp);


        for (j=0; j<rowlen[i]/T; j++) {
            tmp = axpy<v_t,m_t>(tmp, rhs[col[cs + tid + (threadIdx.y*rowlen[i]/T+j)*C]], val[cs + tid + (threadIdx.y*rowlen[i]/T+j)*C]);
        }
        smem[tib] = tmp;
        __syncthreads();
        
        if (T>2) {
            if (T>4) {
                if (T>8) {
                    if (T>16) {
                        if (threadIdx.y<16) {
                            smem[tib] = axpy<v_t,float>(smem[tib],smem[tib+16],1.f);
                            __syncthreads();
                        }
                    }
                    if (threadIdx.y<8) {
                        smem[tib] = axpy<v_t,float>(smem[tib],smem[tib+8],1.f);
                        __syncthreads();
                    }
                }
                if (threadIdx.y<4) {
                    smem[tib] = axpy<v_t,float>(smem[tib],smem[tib+4],1.f);
                    __syncthreads();    
                }
            }
            if (threadIdx.y<2) {
                smem[tib] = axpy<v_t,float>(smem[tib],smem[tib+2],1.f);
                __syncthreads();
            }
        }
        __syncthreads();
        
        if (threadIdx.y == 0) {
            if (options & GHOST_SPMVM_AXPY)
                lhs[i] = axpy<v_t,float>(lhs[i],axpy<v_t,float>(smem[tib],smem[tib+1],1.f),1.f);
            else 
                lhs[i] = axpy<v_t,float>(smem[tib],smem[tib+1],1.f);
        }
    }
}


extern "C" void dd_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{
    CHOOSE_KERNEL(double,double);
}

extern "C" void ds_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ 
    CHOOSE_KERNEL(double,float);
}

extern "C" void dc_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ 
    CHOOSE_KERNEL(double,cuFloatComplex);
}

extern "C" void dz_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ 
    CHOOSE_KERNEL(double,cuDoubleComplex);
}

extern "C" void sd_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ 
    CHOOSE_KERNEL(float,double);
}

extern "C" void ss_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ 
    CHOOSE_KERNEL(float,float);
}

extern "C" void sc_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ 
    CHOOSE_KERNEL(float,cuFloatComplex);
}

extern "C" void sz_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ 
    CHOOSE_KERNEL(float,cuDoubleComplex);
}

extern "C" void zd_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ 
    CHOOSE_KERNEL(cuDoubleComplex,double);
}

extern "C" void zs_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ 
    CHOOSE_KERNEL(cuDoubleComplex,float);
}

extern "C" void zc_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ 
    CHOOSE_KERNEL(cuDoubleComplex,cuFloatComplex);
}

extern "C" void zz_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ 
    CHOOSE_KERNEL(cuDoubleComplex,cuDoubleComplex);
}

extern "C" void cd_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ 
    CHOOSE_KERNEL(cuFloatComplex,double);
}

extern "C" void cs_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ 
    CHOOSE_KERNEL(cuFloatComplex,float);
}

extern "C" void cc_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ 
    CHOOSE_KERNEL(cuFloatComplex,cuFloatComplex);
}

extern "C" void cz_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ 
    CHOOSE_KERNEL(cuFloatComplex,cuDoubleComplex);
}
