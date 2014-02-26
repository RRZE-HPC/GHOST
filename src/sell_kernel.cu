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
#include <cuComplex.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "ghost/cu_complex.h"

#define SELL_CUDA_NBLOCKS (int)ceil(mat->nrowsPadded/(double)(SELL_CUDA_THREADSPERBLOCK/SELL(mat)->T)) 
//#define SELLT_STRIDE_ONE

extern __shared__ char shared[];

#define CALL(func,dt1,dt2,b1,b2,b3,b4,b5,...){\
    dt2 shift, scale, beta;\
    GHOST_SPMV_PARSE_ARGS(flags,argp,scale,beta,shift,localdot,dt2);\
    func<dt1,dt2,b1,b2,b3,b4,b5><<<__VA_ARGS__>>>((dt2 *)lhs->cu_val,(dt2 *)rhs->cu_val,flags,mat->nrows,mat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(dt1 *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen,SELL(mat)->chunkHeight,SELL(mat)->T,shift,scale,beta,(dt2 *)cu_localdot);\
   }\

#define SWITCH_BOOLS(func,dt1,dt2,...)\
            if (flags & GHOST_SPMV_AXPY) {\
                if (flags & GHOST_SPMV_AXPBY) {\
                    if (flags & GHOST_SPMV_SCALE) {\
                        if (flags & GHOST_SPMV_SHIFT) {\
                            if (flags & GHOST_SPMV_DOT) {\
                                CALL(func,dt1,dt2,true,true,true,true,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,true,true,true,true,false,__VA_ARGS__)\
                            }\
                        } else {\
                            if (flags & GHOST_SPMV_DOT) {\
                                CALL(func,dt1,dt2,true,true,true,false,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,true,true,true,false,false,__VA_ARGS__)\
                            }\
                        }\
                    } else {\
                        if (flags & GHOST_SPMV_SHIFT) {\
                            if (flags & GHOST_SPMV_DOT) {\
                                CALL(func,dt1,dt2,true,true,false,true,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,true,true,false,true,false,__VA_ARGS__)\
                            }\
                        } else {\
                            if (flags & GHOST_SPMV_DOT) {\
                                CALL(func,dt1,dt2,true,true,false,false,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,true,true,false,false,false,__VA_ARGS__)\
                            }\
                        }\
                    }\
                } else {\
                    if (flags & GHOST_SPMV_SCALE) {\
                        if (flags & GHOST_SPMV_SHIFT) {\
                            if (flags & GHOST_SPMV_DOT) {\
                                CALL(func,dt1,dt2,true,false,true,true,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,true,false,true,true,false,__VA_ARGS__)\
                            }\
                        } else {\
                            if (flags & GHOST_SPMV_DOT) {\
                                CALL(func,dt1,dt2,true,false,true,false,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,true,false,true,false,false,__VA_ARGS__)\
                            }\
                        }\
                    } else {\
                        if (flags & GHOST_SPMV_SHIFT) {\
                            if (flags & GHOST_SPMV_DOT) {\
                                CALL(func,dt1,dt2,true,false,false,true,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,true,false,false,true,false,__VA_ARGS__)\
                            }\
                        } else {\
                            if (flags & GHOST_SPMV_DOT) {\
                                CALL(func,dt1,dt2,true,false,false,false,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,true,false,false,false,false,__VA_ARGS__)\
                            }\
                        }\
                    }\
                }\
            } else {\
                if (flags & GHOST_SPMV_AXPBY) {\
                    if (flags & GHOST_SPMV_SCALE) {\
                        if (flags & GHOST_SPMV_SHIFT) {\
                            if (flags & GHOST_SPMV_DOT) {\
                                CALL(func,dt1,dt2,false,true,true,true,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,false,true,true,true,false,__VA_ARGS__)\
                            }\
                        } else {\
                            if (flags & GHOST_SPMV_DOT) {\
                                CALL(func,dt1,dt2,false,true,true,false,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,false,true,true,false,false,__VA_ARGS__)\
                            }\
                        }\
                    } else {\
                        if (flags & GHOST_SPMV_SHIFT) {\
                            if (flags & GHOST_SPMV_DOT) {\
                                CALL(func,dt1,dt2,false,true,false,true,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,false,true,false,true,false,__VA_ARGS__)\
                            }\
                        } else {\
                            if (flags & GHOST_SPMV_DOT) {\
                                CALL(func,dt1,dt2,false,true,false,false,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,false,true,false,false,false,__VA_ARGS__)\
                            }\
                        }\
                    }\
                } else {\
                    if (flags & GHOST_SPMV_SCALE) {\
                        if (flags & GHOST_SPMV_SHIFT) {\
                            if (flags & GHOST_SPMV_DOT) {\
                                CALL(func,dt1,dt2,false,false,true,true,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,false,false,true,true,false,__VA_ARGS__)\
                            }\
                        } else {\
                            if (flags & GHOST_SPMV_DOT) {\
                                CALL(func,dt1,dt2,false,false,true,false,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,false,false,true,false,false,__VA_ARGS__)\
                            }\
                        }\
                    } else {\
                        if (flags & GHOST_SPMV_SHIFT) {\
                            if (flags & GHOST_SPMV_DOT) {\
                                CALL(func,dt1,dt2,false,false,false,true,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,false,false,false,true,false,__VA_ARGS__)\
                            }\
                        } else {\
                            if (flags & GHOST_SPMV_DOT) {\
                                CALL(func,dt1,dt2,false,false,false,false,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,false,false,false,false,false,__VA_ARGS__)\
                            }\
                        }\
                    }\
                }\
            }\

#define CHOOSE_KERNEL(dt1,dt2) {\
    ghost_error_t ret = GHOST_SUCCESS;\
    int ghost_cu_device;\
    GHOST_CALL_RETURN(ghost_cu_getDevice(&ghost_cu_device));\
    static int infoprinted=0;\
    void *cu_localdot = NULL;\
    dt2 *localdot = NULL;\
    if ((SELL(mat)->T > 128) || (SELL(mat)->T == 0) || (SELL(mat)->T & (SELL(mat)->T-1)))\
    WARNING_LOG("Invalid T: %d (must be power of two and T <= 128)",SELL(mat)->T);\
    GHOST_INSTR_START(CU_SELL_SpMVM)\
    if (SELL(mat)->chunkHeight == mat->nrowsPadded) {\
        if (SELL(mat)->T > 1) {\
            INFO_LOG("ELLPACK-T kernel not available. Switching to SELL-T kernel although we have only one chunk. Performance may suffer.");\
            size_t reqSmem;\
            ghost_sizeofDatatype(&reqSmem,lhs->traits->datatype);\
            reqSmem *= SELL_CUDA_THREADSPERBLOCK;\
            struct cudaDeviceProp prop;\
            CUDA_CALL_RETURN(cudaGetDeviceProperties(&prop,ghost_cu_device));\
            if (prop.sharedMemPerBlock < reqSmem) {\
                WARNING_LOG("Not enough shared memory available! CUDA kernel will not execute!");\
            }\
            dim3 block(SELL_CUDA_THREADSPERBLOCK/SELL(mat)->T,SELL(mat)->T);\
            SWITCH_BOOLS(SELLT_kernel_CU_tmpl,dt1,dt2,SELL_CUDA_NBLOCKS,block,reqSmem)\
        } else {\
            SWITCH_BOOLS(SELL_kernel_CU_ELLPACK_tmpl,dt1,dt2,SELL_CUDA_NBLOCKS,SELL_CUDA_THREADSPERBLOCK)\
        }\
    }else{\
        if (SELL(mat)->T > 1) {\
            size_t reqSmem;\
            ghost_sizeofDatatype(&reqSmem,lhs->traits->datatype);\
            reqSmem *= SELL_CUDA_THREADSPERBLOCK;\
            struct cudaDeviceProp prop;\
            CUDA_CALL_RETURN(cudaGetDeviceProperties(&prop,ghost_cu_device));\
            if (prop.sharedMemPerBlock < reqSmem) {\
                WARNING_LOG("Not enough shared memory available! CUDA kernel will not execute!");\
            }\
            dim3 block(SELL_CUDA_THREADSPERBLOCK/SELL(mat)->T,SELL(mat)->T);\
            SWITCH_BOOLS(SELLT_kernel_CU_tmpl,dt1,dt2,SELL_CUDA_NBLOCKS,block,reqSmem)\
        } else {\
            SWITCH_BOOLS(SELL_kernel_CU_tmpl,dt1,dt2,SELL_CUDA_NBLOCKS,SELL_CUDA_THREADSPERBLOCK)\
        }\
    }\
    cudaDeviceSynchronize();\
    GHOST_INSTR_STOP(CU_SELL_SpMVM)\
    GHOST_INSTR_START(CU_SpMVM_localdot)\
    if (flags & GHOST_SPMV_DOT) {\
        if (!infoprinted)\
            INFO_LOG("Not doing the local dot product on-the-fly!");\
        infoprinted=1;\
        lhs->dot(lhs,lhs,&localdot[0]);\
        lhs->dot(lhs,rhs,&localdot[1]);\
        lhs->dot(rhs,rhs,&localdot[2]);\
    }\
    GHOST_INSTR_STOP(CU_SpMVM_localdot)\
    return ret;\
}

    template<typename m_t, typename v_t, bool do_axpy, bool do_axpby, bool do_scale, bool do_shift, bool do_localdot>  
__global__ void SELL_kernel_CU_ELLPACK_tmpl(v_t *lhs, v_t *rhs, ghost_spmv_flags_t flags, int nrows, int nrowspadded, ghost_idx_t *rowlen, ghost_idx_t *col, m_t *val, ghost_nnz_t *chunkstart, ghost_idx_t *chunklen, int C, int T, v_t shift, v_t alpha, v_t beta, v_t *localdot)
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
        
        if (do_shift) {
            if (do_scale) {
                if (do_axpy) {
                    lhs[i] = axpy<v_t,float>(lhs[i],scale<v_t>(alpha,axpy<v_t,v_t>(tmp,rhs[i],scale2<v_t,float>(shift,-1.f))),1.f);
                } else if (do_axpby) {
                    lhs[i] = axpy<v_t,float>(scale<v_t>(lhs[i],beta),scale<v_t>(alpha,axpy<v_t,v_t>(tmp,rhs[i],scale2<v_t,float>(shift,-1.f))),1.f);
                } else {
                    lhs[i] = scale<v_t>(alpha,axpy<v_t,v_t>(tmp,rhs[i],scale2<v_t,float>(shift,-1.f)));
                }
            } else {
                if (do_axpy) {
                    lhs[i] = axpy<v_t,float>(lhs[i],axpy<v_t,v_t>(tmp,rhs[i],scale2<v_t,float>(shift,-1.f)),1.f);
                } else if (do_axpby) {
                    lhs[i] = axpy<v_t,float>(scale<v_t>(lhs[i],beta),axpy<v_t,v_t>(tmp,rhs[i],scale2<v_t,float>(shift,-1.f)),1.f);
                } else {
                    lhs[i] = axpy<v_t,v_t>(tmp,rhs[i],scale2<v_t,float>(shift,-1.f));
                }
            }
        } else {
            if (do_scale) {
                if (do_axpy) {
                    lhs[i] = axpy<v_t,float>(lhs[i],scale<v_t>(alpha,tmp),1.f);
                } else if (do_axpby) {
                    lhs[i] = axpy<v_t,float>(scale<v_t>(lhs[i],beta),scale<v_t>(alpha,tmp),1.f);
                } else {
                    lhs[i] = scale<v_t>(alpha,tmp);
                }
            } else {
                if (do_axpy) {
                    lhs[i] = axpy<v_t,float>(lhs[i],tmp,1.f);
                } else if (do_axpby) {
                    lhs[i] = axpy<v_t,float>(scale<v_t>(lhs[i],beta),tmp,1.f);
                } else {
                    lhs[i] = tmp;
                }
            }
            
        }
    }
}

    template<typename m_t, typename v_t, bool do_axpy, bool do_axpby, bool do_scale, bool do_shift, bool do_localdot>  
__global__ void SELL_kernel_CU_tmpl(v_t *lhs, v_t *rhs, ghost_spmv_flags_t flags, int nrows, int nrowspadded, ghost_idx_t *rowlen, ghost_idx_t *col, m_t *val, ghost_nnz_t *chunkstart, ghost_idx_t *chunklen, int C, int T, v_t shift, v_t alpha, v_t beta, v_t *localdot)
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

        if (do_shift) {
            if (do_scale) {
                if (do_axpy) {
                    lhs[i] = axpy<v_t,float>(lhs[i],scale<v_t>(alpha,axpy<v_t,v_t>(tmp,rhs[i],scale2<v_t,float>(shift,-1.f))),1.f);
                } else if (do_axpby) {
                    lhs[i] = axpy<v_t,float>(scale<v_t>(lhs[i],beta),scale<v_t>(alpha,axpy<v_t,v_t>(tmp,rhs[i],scale2<v_t,float>(shift,-1.f))),1.f);
                } else {
                    lhs[i] = scale<v_t>(alpha,axpy<v_t,v_t>(tmp,rhs[i],scale2<v_t,float>(shift,-1.f)));
                }
            } else {
                if (do_axpy) {
                    lhs[i] = axpy<v_t,float>(lhs[i],axpy<v_t,v_t>(tmp,rhs[i],scale2<v_t,float>(shift,-1.f)),1.f);
                } else if (do_axpby) {
                    lhs[i] = axpy<v_t,float>(scale<v_t>(lhs[i],beta),axpy<v_t,v_t>(tmp,rhs[i],scale2<v_t,float>(shift,-1.f)),1.f);
                } else {
                    lhs[i] = axpy<v_t,v_t>(tmp,rhs[i],scale2<v_t,float>(shift,-1.f));
                }
            }
        } else {
            if (do_scale) {
                if (do_axpy) {
                    lhs[i] = axpy<v_t,float>(lhs[i],scale<v_t>(alpha,tmp),1.f);
                } else if (do_axpby) {
                    lhs[i] = axpy<v_t,float>(scale<v_t>(lhs[i],beta),scale<v_t>(alpha,tmp),1.f);
                } else {
                    lhs[i] = scale<v_t>(alpha,tmp);
                }
            } else {
                if (do_axpy) {
                    lhs[i] = axpy<v_t,float>(lhs[i],tmp,1.f);
                } else if (do_axpby) {
                    lhs[i] = axpy<v_t,float>(scale<v_t>(lhs[i],beta),tmp,1.f);
                } else {
                    lhs[i] = tmp;
                }
            }
            
        }

    }
}

    template<typename m_t, typename v_t, bool do_axpy, bool do_axpby, bool do_scale, bool do_shift, bool do_localdot>  
__global__ void SELLT_kernel_CU_tmpl(v_t *lhs, v_t *rhs, ghost_spmv_flags_t flags, ghost_idx_t nrows, ghost_idx_t nrowspadded, ghost_idx_t *rowlen, ghost_idx_t *col, m_t *val, ghost_nnz_t *chunkstart, ghost_idx_t *chunklen, ghost_idx_t C, int T, v_t shift, v_t alpha, v_t beta, v_t *localdot)
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
#ifdef SELLT_STRIDE_ONE
            tmp = axpy<v_t,m_t>(tmp, rhs[col[cs + tid + (threadIdx.y*rowlen[i]/T+j)*C]], val[cs + tid + (threadIdx.y*rowlen[i]/T+j)*C]);
#else
            tmp = axpy<v_t,m_t>(tmp, rhs[col[cs + tid + (threadIdx.y+j*blockDim.y)*C]], val[cs + tid + (threadIdx.y+j*blockDim.y)*C]);
#endif
        }

        smem[tib] = tmp;
        __syncthreads();

        if (T>2) {
            if (T>4) {
                if (T>8) {
                    if (T>16) {
                        if (T>32) {
                            if (T>64) {
                                if (threadIdx.y<64) {
                                    smem[tib] = axpy<v_t,float>(smem[tib],smem[tib+64],1.f);
                                    __syncthreads();
                                }
                            }
                            if (threadIdx.y<32) {
                                smem[tib] = axpy<v_t,float>(smem[tib],smem[tib+32],1.f);
                                __syncthreads();
                            }
                        }
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

        if (threadIdx.y == 0) {
            if (do_shift) {
                if (do_scale) {
                    if (do_axpy) {
                        lhs[i] = axpy<v_t,float>(lhs[i],scale<v_t>(alpha,axpy<v_t,v_t>(tmp,rhs[i],scale2<v_t,float>(shift,-1.f))),1.f);
                    } else if (do_axpby) {
                        lhs[i] = axpy<v_t,float>(scale<v_t>(lhs[i],beta),scale<v_t>(alpha,axpy<v_t,v_t>(tmp,rhs[i],scale2<v_t,float>(shift,-1.f))),1.f);
                    } else {
                        lhs[i] = scale<v_t>(alpha,axpy<v_t,v_t>(tmp,rhs[i],scale2<v_t,float>(shift,-1.f)));
                    }
                } else {
                    if (do_axpy) {
                        lhs[i] = axpy<v_t,float>(lhs[i],axpy<v_t,v_t>(tmp,rhs[i],scale2<v_t,float>(shift,-1.f)),1.f);
                    } else if (do_axpby) {
                        lhs[i] = axpy<v_t,float>(scale<v_t>(lhs[i],beta),axpy<v_t,v_t>(tmp,rhs[i],scale2<v_t,float>(shift,-1.f)),1.f);
                    } else {
                        lhs[i] = axpy<v_t,v_t>(tmp,rhs[i],scale2<v_t,float>(shift,-1.f));
                    }
                }
            } else {
                if (do_scale) {
                    if (do_axpy) {
                        lhs[i] = axpy<v_t,float>(lhs[i],scale<v_t>(alpha,tmp),1.f);
                    } else if (do_axpby) {
                        lhs[i] = axpy<v_t,float>(scale<v_t>(lhs[i],beta),scale<v_t>(alpha,tmp),1.f);
                    } else {
                        lhs[i] = scale<v_t>(alpha,tmp);
                    }
                } else {
                    if (do_axpy) {
                        lhs[i] = axpy<v_t,float>(lhs[i],tmp,1.f);
                    } else if (do_axpby) {
                        lhs[i] = axpy<v_t,float>(scale<v_t>(lhs[i],beta),tmp,1.f);
                    } else {
                        lhs[i] = tmp;
                    }
                }
                
            }
        }
    }
}


extern "C" ghost_error_t dd_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{
    CHOOSE_KERNEL(double,double);
}

extern "C" ghost_error_t ds_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(double,float);
}

extern "C" ghost_error_t dc_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(double,cuFloatComplex);
}

extern "C" ghost_error_t dz_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(double,cuDoubleComplex);
}

extern "C" ghost_error_t sd_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(float,double);
}

extern "C" ghost_error_t ss_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(float,float);
}

extern "C" ghost_error_t sc_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(float,cuFloatComplex);
}

extern "C" ghost_error_t sz_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(float,cuDoubleComplex);
}

extern "C" ghost_error_t zd_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(cuDoubleComplex,double);
}

extern "C" ghost_error_t zs_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(cuDoubleComplex,float);
}

extern "C" ghost_error_t zc_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(cuDoubleComplex,cuFloatComplex);
}

extern "C" ghost_error_t zz_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(cuDoubleComplex,cuDoubleComplex);
}

extern "C" ghost_error_t cd_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(cuFloatComplex,double);
}

extern "C" ghost_error_t cs_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(cuFloatComplex,float);
}

extern "C" ghost_error_t cc_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(cuFloatComplex,cuFloatComplex);
}

extern "C" ghost_error_t cz_SELL_kernel_CU(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t flags, va_list argp)
{ 
    CHOOSE_KERNEL(cuFloatComplex,cuDoubleComplex);
}

