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
    func<dt1,dt2,b1,b2,b3,b4><<<__VA_ARGS__>>>((dt2 *)lhs->cu_val,*(int *)(lhs->stride),(dt2 *)rhs->cu_val,*(int *)rhs->stride,flags,mat->nrows,mat->nrowsPadded,rhs->traits.ncols/MAX_COLS_PER_BLOCK,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(dt1 *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen,SELL(mat)->chunkHeight,SELL(mat)->T,shift,(dt2)scale,(dt2)beta,(dt2 *)cu_localdot);\
}\

#define SWITCH_BOOLS(func,dt1,dt2,...)\
        if (flags & GHOST_SPMV_AXPBY || flags & GHOST_SPMV_AXPY) {\
            if (flags & GHOST_SPMV_SCALE) {\
                if (flags & GHOST_SPMV_VSHIFT) {\
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
                if (flags & GHOST_SPMV_VSHIFT) {\
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
                if (flags & GHOST_SPMV_VSHIFT) {\
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
                if (flags & GHOST_SPMV_VSHIFT) {\
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
        INFO_LOG("Experimental local dot product with final reduction over %d blocks!",SELL_CUDA_NBLOCKS);\
        dt2_host *localdot_blocks;\
        GHOST_CALL_RETURN(ghost_malloc((void **)&localdot_blocks,sizeof(dt2_host)*rhs->traits.ncols*3*SELL_CUDA_NBLOCKS));\
        GHOST_CALL_RETURN(ghost_cu_download(localdot_blocks,cu_localdot,sizeof(dt2_host)*rhs->traits.ncols*3*SELL_CUDA_NBLOCKS));\
        _Pragma("omp parallel for private(block)")\
        for (col=0; col<rhs->traits.ncols; col++) {\
            localdot[col                      ] = 0;\
            localdot[col + 1*rhs->traits.ncols] = 0;\
            localdot[col + 2*rhs->traits.ncols] = 0;\
            for (block=0; block<SELL_CUDA_NBLOCKS; block++) {\
                localdot[col                      ] += localdot_blocks[                                        col*SELL_CUDA_NBLOCKS + block];\
                localdot[col + 1*rhs->traits.ncols] += localdot_blocks[1*SELL_CUDA_NBLOCKS*rhs->traits.ncols + col*SELL_CUDA_NBLOCKS + block];\
                localdot[col + 2*rhs->traits.ncols] += localdot_blocks[2*SELL_CUDA_NBLOCKS*rhs->traits.ncols + col*SELL_CUDA_NBLOCKS + block];\
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
    GHOST_SPMV_PARSE_ARGS(flags,argp,scale,beta,shift,localdot,dt2_host,dt2);\
    if (flags & GHOST_SPMV_AXPY) {\
        dt2_host hbeta = 1.;\
        beta = *((dt2 *)&hbeta);\
    }\
    if (flags & GHOST_SPMV_SHIFT) {\
        ERROR_LOG("Currently not working!");\
    }\
    GHOST_CALL_RETURN(ghost_cu_malloc((void **)&cu_localdot,sizeof(dt2)*rhs->traits.ncols*3*SELL_CUDA_NBLOCKS));\
    size_t shiftsize = sizeof(dt2)*(flags & GHOST_SPMV_SHIFT?1:(flags & GHOST_SPMV_VSHIFT?rhs->traits.ncols:0));\
    GHOST_CALL_RETURN(ghost_cu_malloc((void **)&cu_shift,shiftsize));\
    GHOST_CALL_RETURN(ghost_cu_upload(cu_shift,shift,shiftsize));\
    struct cudaDeviceProp prop;\
    CUDA_CALL_RETURN(cudaGetDeviceProperties(&prop,cu_device));\
    GHOST_INSTR_START(spmv_cuda)\
    if (rhs->traits.ncols > 1) {\
        if (SELL(mat)->T > 1) {\
            WARNING_LOG("SELL-T kernel for multiple vectors nor implemented, falling back to SELL-1!");\
        }\
        int blockheight = PAD((int)ceil((double)SELL_CUDA_THREADSPERBLOCK/MIN(rhs->traits.ncols,MAX_COLS_PER_BLOCK)),SELL(mat)->chunkHeight);\
        if (blockheight*MIN(rhs->traits.ncols,MAX_COLS_PER_BLOCK) > 1024) {\
            WARNING_LOG("Too many threads! (FIXME)");\
        }\
        if (rhs->traits.ncols > MAX_COLS_PER_BLOCK) {\
            WARNING_LOG("Will have a loop over the vectors!");\
        }\
        if (rhs->traits.storage == GHOST_DENSEMAT_COLMAJOR) {\
            dim3 block(SELL_CUDA_THREADSPERBLOCK/MIN(MAX_COLS_PER_BLOCK,rhs->traits.ncols),MIN(MAX_COLS_PER_BLOCK,rhs->traits.ncols));\
            dim3 grid((int)ceil(mat->nrowsPadded/(double)block.x),(int)(ceil(rhs->traits.ncols/(double)MAX_COLS_PER_BLOCK)));\
            size_t reqSmem = 0;\
            if (flags & GHOST_SPMV_DOT) {\
                reqSmem = sizeof(dt2)*32*block.y;\
            }\
            if (prop.sharedMemPerBlock < reqSmem) {\
                WARNING_LOG("Not enough shared memory available! CUDA kernel will not execute!");\
            }\
            INFO_LOG("grid %dx%d block %dx%d shmem %zu",grid.x,grid.y,block.x,block.y,reqSmem);\
            SWITCH_BOOLS(SELL_kernel_CU_tmpl,dt1,dt2,grid,block,reqSmem)\
            /*SELL_kernel_CU_tmpl<dt1,dt2><<<(int)ceil(mat->nrowsPadded/(double)blockheight),block,reqSmem>>>((dt2 *)lhs->cu_val,lhs->traits.nrowspadded,(dt2 *)rhs->cu_val,rhs->traits.nrowspadded,flags,mat->nrows,mat->nrowsPadded,rhs->traits.ncols/block.y,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(dt1 *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen,SELL(mat)->chunkHeight,SELL(mat)->T,cu_shift,scale,beta,cu_localdot,flags&GHOST_SPMV_AXPY,flags&GHOST_SPMV_AXPBY,flags&GHOST_SPMV_SCALE,flags&GHOST_SPMV_SHIFT,flags&GHOST_SPMV_VSHIFT,flags&GHOST_SPMV_DOT);*/\
        } else {\
            INFO_LOG("Experimental row-major CUDA SELL-SpMMV");\
            dim3 block(MIN(MAX_COLS_PER_BLOCK,rhs->traits.ncols),SELL_CUDA_THREADSPERBLOCK/MIN(MAX_COLS_PER_BLOCK,rhs->traits.ncols));\
            dim3 grid((int)ceil(mat->nrowsPadded/(double)block.y),(int)(ceil(rhs->traits.ncols/(double)MAX_COLS_PER_BLOCK)));\
            INFO_LOG("grid %dx%d block %dx%d",grid.x,grid.y,block.x,block.y);\
            SWITCH_BOOLS(SELL_kernel_CU_rm_tmpl,dt1,dt2,grid,block)\
            /*SELL_kernel_CU_rm_tmpl<dt1,dt2><<<(int)ceil(mat->nrowsPadded/(double)32),newblock,reqSmem>>>((dt2 *)lhs->cu_val,lhs->traits.ncolspadded,(dt2 *)rhs->cu_val,rhs->traits.ncolspadded,flags,mat->nrows,mat->nrowsPadded,rhs->traits.ncols/newblock.x,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(dt1 *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen,SELL(mat)->chunkHeight,SELL(mat)->T,cu_shift,scale,beta,cu_localdot,flags&GHOST_SPMV_AXPY,flags&GHOST_SPMV_AXPBY,flags&GHOST_SPMV_SCALE,flags&GHOST_SPMV_SHIFT,flags&GHOST_SPMV_VSHIFT,flags&GHOST_SPMV_DOT);*/\
        }\
    } else {\
        int blockheight = PAD((int)ceil((double)SELL_CUDA_THREADSPERBLOCK/rhs->traits.ncols),SELL(mat)->chunkHeight);\
        if (blockheight*rhs->traits.ncols > 1024) {\
            WARNING_LOG("Too many threads! (FIXME)");\
        }\
        size_t reqSmem;\
        ghost_datatype_size(&reqSmem,lhs->traits.datatype);\
        reqSmem *= blockheight*rhs->traits.ncols;\
        if (prop.sharedMemPerBlock < reqSmem) {\
            WARNING_LOG("Not enough shared memory available! CUDA kernel will not execute!");\
        }\
        dim3 block(blockheight,rhs->traits.ncols);\
        SWITCH_BOOLS(SELL_kernel_CU_tmpl,dt1,dt2,SELL_CUDA_NBLOCKS,SELL_CUDA_THREADSPERBLOCK)\
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
        if (C == blockDim.x) {
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
            lhs[lhs_lda*i+col] = axpy<v_t,float>(lhs[lhs_lda*i+col],tmp,1.f);
        } else {
            lhs[lhs_lda*i+col] = tmp;
        }
    }
#ifdef LOCALDOT_ONTHEFLY 
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
            localdot[0*ncols*blockDim.y*gridDim.x + col*gridDim.x + blockIdx.x] = dot1;
            localdot[1*ncols*blockDim.y*gridDim.x + col*gridDim.x + blockIdx.x] = dot2;
            localdot[2*ncols*blockDim.y*gridDim.x + col*gridDim.x + blockIdx.x] = dot3;
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

