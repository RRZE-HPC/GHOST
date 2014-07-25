#include "ghost/config.h"
#undef GHOST_HAVE_MPI
#undef GHOST_HAVE_INSTR_LIKWID
#include "ghost/types.h"
#include "ghost/sell.h"
#include "ghost/complex.h"
#include "ghost/instr.h"
#include "ghost/log.h"
#include "ghost/error.h"
#include "ghost/util.h"
#include "ghost/math.h"

#include <cuComplex.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <complex.h>

#include "ghost/cu_complex.h"
#include "ghost/cu_sell_kernel.h"

#define MAX_COLS_PER_BLOCK 32
#define SELL_CUDA_THREADSPERBLOCK 1024
#define SELL_CUDA_NBLOCKS (int)ceil(mat->nrowsPadded/ceil((double)(SELL_CUDA_THREADSPERBLOCK/((double)SELL(mat)->T*(double)(MIN(rhs->traits.ncols,MAX_COLS_PER_BLOCK))))))
//#define SELLT_STRIDE_ONE
#define LOCALDOT_ONTHEFLY


#define CALL(func,dt1,dt2,b1,b2,b3,b4,...){\
    func<dt1,dt2,b1,b2,b3,b4><<<__VA_ARGS__>>>((dt2 *)lhs->cu_val,*(int *)(lhs->stride),(dt2 *)rhs->cu_val,*(int *)rhs->stride,flags,mat->nrows,mat->nrowsPadded,rhs->traits.ncols/MAX_COLS_PER_BLOCK,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(dt1 *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen,SELL(mat)->chunkHeight,SELL(mat)->T,(dt2 *)cu_shift,(dt2)scale,(dt2)beta,(dt2 *)cu_localdot);\
}\

#define SWITCH_BOOLS(func,dt1,dt2,...)\
        if (flags & GHOST_SPMV_AXPBY || flags & GHOST_SPMV_AXPY) {\
            if (flags & GHOST_SPMV_SCALE) {\
                if (flags & (GHOST_SPMV_VSHIFT | GHOST_SPMV_SHIFT)) {\
                    if (flags & GHOST_SPMV_DOT) {\
                        CALL(func,dt1,dt2,true,true,true,true,__VA_ARGS__)\
                    } else {\
                        CALL(func,dt1,dt2,true,true,true,false,__VA_ARGS__)\
                    }\
                } else {\
                    if (flags & GHOST_SPMV_DOT) {\
                        CALL(func,dt1,dt2,true,true,false,true,__VA_ARGS__)\
                    } else {\
                        CALL(func,dt1,dt2,true,true,false,false,__VA_ARGS__)\
                    }\
                }\
            } else {\
                if (flags & (GHOST_SPMV_VSHIFT | GHOST_SPMV_SHIFT)) {\
                    if (flags & GHOST_SPMV_DOT) {\
                        CALL(func,dt1,dt2,true,false,true,true,__VA_ARGS__)\
                    } else {\
                        CALL(func,dt1,dt2,true,false,true,false,__VA_ARGS__)\
                    }\
                } else {\
                    if (flags & GHOST_SPMV_DOT) {\
                        CALL(func,dt1,dt2,true,false,false,true,__VA_ARGS__)\
                    } else {\
                        CALL(func,dt1,dt2,true,false,false,false,__VA_ARGS__)\
                    }\
                }\
            }\
        } else {\
            if (flags & GHOST_SPMV_SCALE) {\
                if (flags & (GHOST_SPMV_VSHIFT | GHOST_SPMV_SHIFT)) {\
                    if (flags & GHOST_SPMV_DOT) {\
                        CALL(func,dt1,dt2,false,true,true,true,__VA_ARGS__)\
                    } else {\
                        CALL(func,dt1,dt2,false,true,true,false,__VA_ARGS__)\
                    }\
                } else {\
                    if (flags & GHOST_SPMV_DOT) {\
                        CALL(func,dt1,dt2,false,true,false,true,__VA_ARGS__)\
                    } else {\
                        CALL(func,dt1,dt2,false,true,false,false,__VA_ARGS__)\
                    }\
                }\
            } else {\
                if (flags & (GHOST_SPMV_VSHIFT | GHOST_SPMV_SHIFT)) {\
                    if (flags & GHOST_SPMV_DOT) {\
                        CALL(func,dt1,dt2,false,false,true,true,__VA_ARGS__)\
                    } else {\
                        CALL(func,dt1,dt2,false,false,true,false,__VA_ARGS__)\
                    }\
                } else {\
                    if (flags & GHOST_SPMV_DOT) {\
                        CALL(func,dt1,dt2,false,false,false,true,__VA_ARGS__)\
                    } else {\
                        CALL(func,dt1,dt2,false,false,false,false,__VA_ARGS__)\
                    }\
                }\
            }\
        }\


#ifdef LOCALDOT_ONTHEFLY
#define PROCESS_LOCALDOT(dt2_host)\
        GHOST_INSTR_START(spmv_cuda_dot_reduction)\
        int block, col;\
        INFO_LOG("Experimental local dot product with final reduction over %d blocks!",grid.x);\
        dt2_host *localdot_blocks;\
        GHOST_CALL_RETURN(ghost_malloc((void **)&localdot_blocks,sizeof(dt2_host)*rhs->traits.ncols*3*grid.x));\
        GHOST_CALL_RETURN(ghost_cu_download(localdot_blocks,cu_localdot,sizeof(dt2_host)*rhs->traits.ncols*3*grid.x));\
        _Pragma("omp parallel for private(block)")\
        for (col=0; col<rhs->traits.ncols; col++) {\
            localdot[col                      ] = 0;\
            localdot[col + 1*rhs->traits.ncols] = 0;\
            localdot[col + 2*rhs->traits.ncols] = 0;\
            for (block=0; block<grid.x; block++) {\
                localdot[col                      ] += localdot_blocks[                             col*grid.x + block];\
                localdot[col + 1*rhs->traits.ncols] += localdot_blocks[1*grid.x*rhs->traits.ncols + col*grid.x + block];\
                localdot[col + 2*rhs->traits.ncols] += localdot_blocks[2*grid.x*rhs->traits.ncols + col*grid.x + block];\
            }\
        }\
        free(localdot_blocks);\
        GHOST_INSTR_STOP(spmv_cuda_dot_reduction)
#else
#define PROCESS_LOCALDOT(dt2_host)\
        GHOST_INSTR_START(spmv_cuda_dot)\
          INFO_LOG("Not doing the local dot product on-the-fly!");\
          memset(localdot,0,rhs->traits.ncols*3*sizeof(dt2_host));\
          lhs->dot(lhs,&localdot[0],lhs);\
          lhs->dot(lhs,&localdot[rhs->traits.ncols],rhs);\
          lhs->dot(rhs,&localdot[2*rhs->traits.ncols],rhs);\
          GHOST_INSTR_STOP(spmv_cuda_dot)
#endif

#define CHOOSE_KERNEL(dt1,dt2,dt2_host) {\
    ghost_error_t ret = GHOST_SUCCESS;\
    int cu_device;\
    GHOST_CALL_RETURN(ghost_cu_device(&cu_device));\
    dt2 *cu_localdot = NULL;\
    dt2 *cu_shift = NULL;\
    dt2_host *localdot = NULL;\
    dt2 *shift, scale, beta;\
    dim3 block, grid;\
    GHOST_SPMV_PARSE_ARGS(flags,argp,scale,beta,shift,localdot,dt2_host,dt2);\
    if (flags & GHOST_SPMV_AXPY) {\
        dt2_host hbeta = 1.;\
        beta = *((dt2 *)&hbeta);\
    }\
    size_t shiftsize = sizeof(dt2)*(flags & (GHOST_SPMV_VSHIFT|GHOST_SPMV_SHIFT)?rhs->traits.ncols:0);\
    GHOST_CALL_RETURN(ghost_cu_malloc((void **)&cu_shift,shiftsize));\
    if (flags & GHOST_SPMV_SHIFT) {\
        INFO_LOG("scatter shift %zu bytes",shiftsize);\
        ghost_lidx_t c;\
        for (c=0; c<rhs->traits.ncols; c++) {\
            GHOST_CALL_RETURN(ghost_cu_upload(&cu_shift[c],shift,sizeof(dt2)));\
        }\
    } else {\
        GHOST_CALL_RETURN(ghost_cu_upload(cu_shift,shift,shiftsize));\
    }\
    struct cudaDeviceProp prop;\
    CUDA_CALL_RETURN(cudaGetDeviceProperties(&prop,cu_device));\
    GHOST_INSTR_START(spmv_cuda)\
    if (rhs->traits.storage == GHOST_DENSEMAT_COLMAJOR) {\
        block.x = SELL_CUDA_THREADSPERBLOCK/MIN(MAX_COLS_PER_BLOCK,rhs->traits.ncols);\
        block.y = (MAX_COLS_PER_BLOCK,rhs->traits.ncols);\
        grid.x = (int)ceil(mat->nrowsPadded/(double)block.x);\
        grid.y = (int)(ceil(rhs->traits.ncols/(double)MAX_COLS_PER_BLOCK));\
        size_t reqSmem = 0;\
        if (flags & GHOST_SPMV_DOT) {\
            reqSmem = sizeof(dt2)*32*block.y*3;\
        }\
        if (prop.sharedMemPerBlock < reqSmem) {\
            WARNING_LOG("Not enough shared memory available! CUDA kernel will not execute!");\
        }\
        INFO_LOG("grid %dx%d block %dx%d shmem %zu",grid.x,grid.y,block.x,block.y,reqSmem);\
        GHOST_CALL_RETURN(ghost_cu_malloc((void **)&cu_localdot,sizeof(dt2)*rhs->traits.ncols*3*grid.x));\
        SWITCH_BOOLS(SELL_kernel_CU_tmpl,dt1,dt2,grid,block,reqSmem)\
    } else {\
        INFO_LOG("Experimental row-major CUDA SELL-SpMMV");\
        block.x = MIN(MAX_COLS_PER_BLOCK,rhs->traits.ncols);\
        block.y = SELL_CUDA_THREADSPERBLOCK/MIN(MAX_COLS_PER_BLOCK,rhs->traits.ncols);\
        grid.x = (int)ceil(mat->nrows/(double)block.y);\
        grid.y = (int)(ceil(rhs->traits.ncols/(double)MAX_COLS_PER_BLOCK));\
        size_t reqSmem = 0;\
        if (flags & GHOST_SPMV_DOT) {\
            reqSmem = sizeof(dt2)*block.x*block.y*3;\
        }\
        if (prop.sharedMemPerBlock < reqSmem) {\
            WARNING_LOG("Not enough shared memory available! CUDA kernel will not execute!");\
        }\
        INFO_LOG("grid %dx%d block %dx%d shmem %zu",grid.x,grid.y,block.x,block.y,reqSmem);\
        GHOST_CALL_RETURN(ghost_cu_malloc((void **)&cu_localdot,sizeof(dt2)*rhs->traits.ncols*3*grid.x));\
        SWITCH_BOOLS(SELL_kernel_CU_rm_tmpl,dt1,dt2,grid,block,reqSmem)\
    }\
    cudaDeviceSynchronize();\
    GHOST_INSTR_STOP(spmv_cuda)\
    if (flags & GHOST_SPMV_DOT) {\
        PROCESS_LOCALDOT(dt2_host)\
    }\
    GHOST_CALL_RETURN(ghost_cu_free(cu_localdot));\
    GHOST_CALL_RETURN(ghost_cu_free(cu_shift));\
    return ret;\
}

    template<typename m_t, typename v_t, bool do_axpby, bool do_scale, bool do_vshift, bool do_localdot>  
__global__ void SELL_kernel_CU_rm_tmpl(v_t *lhs, int lhs_lda, v_t *rhs, int rhs_lda, ghost_spmv_flags_t flags, int nrows, int nrowspadded, int ncols, ghost_lidx_t *rowlen, ghost_lidx_t *mcol, m_t *val, ghost_lidx_t *chunkstart, ghost_lidx_t *chunklen, int C, int T, v_t *shift, v_t alpha, v_t beta, v_t *localdot)
{
    UNUSED(T);
    int i = threadIdx.y+blockIdx.x*blockDim.y;
    int col = blockDim.x*blockIdx.y+threadIdx.x;

    if (i<nrows) {
        int cs, tid;
        if (C == blockDim.y) {
            cs = chunkstart[blockIdx.x];
            tid = threadIdx.y;
        } else {
            cs = chunkstart[i/C];
            tid = threadIdx.y%C;
        }
        int j;
        v_t tmp;

        zero<v_t>(tmp);

        for (j=0; j<rowlen[i]; j++) {
            tmp = axpy<v_t,m_t>(tmp, rhs[rhs_lda*mcol[cs + tid + j*C]+col], val[cs+tid+j*C]);
        }

        if (do_vshift) {
            tmp = axpy<v_t,v_t>(tmp,rhs[rhs_lda*i+col],scale2<v_t,float>(shift[col],-1.f));
        }
        if (do_scale) {
            tmp = scale<v_t>(alpha,tmp);
        }
        if (do_axpby) {
            lhs[lhs_lda*i+col] = axpy<v_t,v_t>(tmp,lhs[lhs_lda*i+col],beta);
        } else {
            lhs[lhs_lda*i+col] = tmp;
        }
    }
#ifdef LOCALDOT_ONTHEFLY 
    if (do_localdot) {
        v_t dot1, dot2, dot3;
        zero<v_t>(dot1);
        zero<v_t>(dot2);
        zero<v_t>(dot3);
        int sidx1 = 0*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
        int sidx2 = 1*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
        int sidx3 = 2*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
        int stride = blockDim.x;

        v_t *shmem = (v_t *) shared;
        
        if (i<nrows) {
            shmem[sidx1] = axpy<v_t>(dot1,lhs[lhs_lda*i+col],lhs[lhs_lda*i+col]);
            shmem[sidx2] = axpy<v_t>(dot1,rhs[rhs_lda*i+col],lhs[lhs_lda*i+col]);
            shmem[sidx3] = axpy<v_t>(dot1,rhs[rhs_lda*i+col],rhs[rhs_lda*i+col]);
        } else {
            zero<v_t>(shmem[sidx1]);
            zero<v_t>(shmem[sidx2]);
            zero<v_t>(shmem[sidx3]);
        }

        __syncthreads();
        
        if (threadIdx.y < 64) {
            if (blockDim.y >= 128) {
                shmem[sidx1] = axpy<v_t>(shmem[sidx1],shmem[sidx1+64*stride],1.f);
                shmem[sidx2] = axpy<v_t>(shmem[sidx2],shmem[sidx2+64*stride],1.f);
                shmem[sidx3] = axpy<v_t>(shmem[sidx3],shmem[sidx3+64*stride],1.f);
                __syncthreads();
            }
            if (threadIdx.y < 32) {
                if (blockDim.y >= 64) {
                    shmem[sidx1] = axpy<v_t>(shmem[sidx1],shmem[sidx1+32*stride],1.f);
                    shmem[sidx2] = axpy<v_t>(shmem[sidx2],shmem[sidx2+32*stride],1.f);
                    shmem[sidx3] = axpy<v_t>(shmem[sidx3],shmem[sidx3+32*stride],1.f);
                    __syncthreads();
                }
                if (threadIdx.y < 16) {
                    if (blockDim.y >= 32) {
                        shmem[sidx1] = axpy<v_t>(shmem[sidx1],shmem[sidx1+16*stride],1.f);
                        shmem[sidx2] = axpy<v_t>(shmem[sidx2],shmem[sidx2+16*stride],1.f);
                        shmem[sidx3] = axpy<v_t>(shmem[sidx3],shmem[sidx3+16*stride],1.f);
                        __syncthreads();
                    }
                    if (threadIdx.y < 8) {
                        if (blockDim.y >= 16) {
                            shmem[sidx1] = axpy<v_t>(shmem[sidx1],shmem[sidx1+8*stride],1.f);
                            shmem[sidx2] = axpy<v_t>(shmem[sidx2],shmem[sidx2+8*stride],1.f);
                            shmem[sidx3] = axpy<v_t>(shmem[sidx3],shmem[sidx3+8*stride],1.f);
                            __syncthreads();
                        }
                        if (threadIdx.y < 4) {
                            if (blockDim.y >= 8) {
                                shmem[sidx1] = axpy<v_t>(shmem[sidx1],shmem[sidx1+4*stride],1.f);
                                shmem[sidx2] = axpy<v_t>(shmem[sidx2],shmem[sidx2+4*stride],1.f);
                                shmem[sidx3] = axpy<v_t>(shmem[sidx3],shmem[sidx3+4*stride],1.f);
                                __syncthreads();
                            }
                            if (threadIdx.y < 2) {
                                if (blockDim.y >= 4) {
                                    shmem[sidx1] = axpy<v_t>(shmem[sidx1],shmem[sidx1+2*stride],1.f);
                                    shmem[sidx2] = axpy<v_t>(shmem[sidx2],shmem[sidx2+2*stride],1.f);
                                    shmem[sidx3] = axpy<v_t>(shmem[sidx3],shmem[sidx3+2*stride],1.f);
                                    __syncthreads();
                                }
                            }
                        }
                    }
                }
            }
        }
        if (threadIdx.y==0) {
            localdot[0*blockDim.x*gridDim.y*gridDim.x + col*gridDim.x + blockIdx.x] = axpy<v_t>(shmem[sidx1],shmem[sidx1+stride],1.f);
            localdot[1*blockDim.x*gridDim.y*gridDim.x + col*gridDim.x + blockIdx.x] = axpy<v_t>(shmem[sidx2],shmem[sidx2+stride],1.f);
            localdot[2*blockDim.x*gridDim.y*gridDim.x + col*gridDim.x + blockIdx.x] = axpy<v_t>(shmem[sidx3],shmem[sidx3+stride],1.f);
        }


#if 0
        if (i<nrows) {
            dot1 = axpy<v_t>(dot1,lhs[lhs_lda*i+col],lhs[lhs_lda*i+col]);
            dot2 = axpy<v_t>(dot2,rhs[rhs_lda*i+col],lhs[lhs_lda*i+col]);
            dot3 = axpy<v_t>(dot3,rhs[rhs_lda*i+col],rhs[rhs_lda*i+col]);
        } else {
            zero<v_t>(dot1);
            zero<v_t>(dot2);
            zero<v_t>(dot3);
        }

        dot1 = blockReduceSum(dot1);
        dot2 = blockReduceSum(dot2);
        dot3 = blockReduceSum(dot3);

        if (threadIdx.y==0) {
            localdot[0*ncols*blockDim.y*gridDim.x + col*gridDim.x + blockIdx.x] = dot1;
            localdot[1*ncols*blockDim.y*gridDim.x + col*gridDim.x + blockIdx.x] = dot2;
            localdot[2*ncols*blockDim.y*gridDim.x + col*gridDim.x + blockIdx.x] = dot3;
        }
#endif
    }
#endif

}

    template<typename m_t, typename v_t, bool do_axpby, bool do_scale, bool do_vshift, bool do_localdot>  
__global__ void SELL_kernel_CU_tmpl(v_t *lhs, int lhs_lda, v_t *rhs, int rhs_lda, ghost_spmv_flags_t flags, int nrows, int nrowspadded, int ncols, ghost_lidx_t *rowlen, ghost_lidx_t *mcol, m_t *val, ghost_lidx_t *chunkstart, ghost_lidx_t *chunklen, int C, int T, v_t *shift, v_t alpha, v_t beta, v_t *localdot)
{
    UNUSED(T);
    int i = threadIdx.x+blockIdx.x*blockDim.x;
    int col = blockDim.y*blockIdx.y+threadIdx.y;

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
            tmp = axpy<v_t,m_t>(tmp, rhs[rhs_lda*col+mcol[cs + tid + j*C]], val[cs+tid+j*C]);
        }

        if (do_vshift) {
            tmp = axpy<v_t,v_t>(tmp,rhs[rhs_lda*col+i],scale2<v_t,float>(shift[col*blockDim.y+threadIdx.y],-1.f));
        }
        if (do_scale) {
            tmp = scale<v_t>(alpha,tmp);
        }
        if (do_axpby) {
            lhs[lhs_lda*col+i] = axpy<v_t,float>(scale<v_t>(lhs[lhs_lda*col+i],beta),tmp,1.f);
        } else {
            lhs[lhs_lda*col+i] = tmp;
        }
    }
#ifdef LOCALDOT_ONTHEFLY 
    if (do_localdot) {
        v_t dot1, dot2, dot3;
        zero<v_t>(dot1);
        zero<v_t>(dot2);
        zero<v_t>(dot3);

        if (i<nrows) {
            dot1 = axpy<v_t>(dot1,lhs[lhs_lda*col+i],lhs[lhs_lda*col+i]);
            dot2 = axpy<v_t>(dot2,rhs[rhs_lda*col+i],lhs[lhs_lda*col+i]);
            dot3 = axpy<v_t>(dot3,rhs[rhs_lda*col+i],rhs[rhs_lda*col+i]);
        } else {
            zero<v_t>(dot1);
            zero<v_t>(dot2);
            zero<v_t>(dot3);
        }

        dot1 = blockReduceSum(dot1);
        dot2 = blockReduceSum(dot2);
        dot3 = blockReduceSum(dot3);

        if (threadIdx.x==0) {
            localdot[0*gridDim.y*blockDim.y*gridDim.x + col*gridDim.x + blockIdx.x] = dot1;
            localdot[1*gridDim.y*blockDim.y*gridDim.x + col*gridDim.x + blockIdx.x] = dot2;
            localdot[2*gridDim.y*blockDim.y*gridDim.x + col*gridDim.x + blockIdx.x] = dot3;
        }
    }
#endif
}

extern "C" ghost_error_t dd_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{
    CHOOSE_KERNEL(double,double,double);
}

extern "C" ghost_error_t ds_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(double,float,float);
}

extern "C" ghost_error_t dc_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(double,cuFloatComplex,complex float);
}

extern "C" ghost_error_t dz_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(double,cuDoubleComplex,complex double);
}

extern "C" ghost_error_t sd_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(float,double,double);
}

extern "C" ghost_error_t ss_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(float,float,float);
}

extern "C" ghost_error_t sc_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(float,cuFloatComplex,complex float);
}

extern "C" ghost_error_t sz_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(float,cuDoubleComplex,complex double);
}

extern "C" ghost_error_t zd_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(cuDoubleComplex,double,double);
}

extern "C" ghost_error_t zs_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(cuDoubleComplex,float,float);
}

extern "C" ghost_error_t zc_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(cuDoubleComplex,cuFloatComplex,complex float);
}

extern "C" ghost_error_t zz_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(cuDoubleComplex,cuDoubleComplex,complex double);
}

extern "C" ghost_error_t cd_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(cuFloatComplex,double,double);
}

extern "C" ghost_error_t cs_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(cuFloatComplex,float,float);
}

extern "C" ghost_error_t cc_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(cuFloatComplex,cuFloatComplex,complex float);
}

extern "C" ghost_error_t cz_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(cuFloatComplex,cuDoubleComplex,complex double);
}

