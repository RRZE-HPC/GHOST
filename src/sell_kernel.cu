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

#include <ghost_cu_complex_helper.h>

#define SELL_CUDA_NBLOCKS (int)ceil(SELL(mat)->cumat->nrowsPadded/(double)(SELL_CUDA_THREADSPERBLOCK/SELL(mat)->T)) 
//#define SELLT_STRIDE_ONE

extern __shared__ char shared[];
extern int ghost_cu_device;

#define CALL(func,dt1,dt2,b1,b2,b3,b4,b5,...) func<dt1,dt2,b1,b2,b3,b4,b5><<<__VA_ARGS__>>>((dt2 *)lhs->CU_val,(dt2 *)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(dt1 *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen,SELL(mat)->chunkHeight,SELL(mat)->T,*(dt2 *)mat->traits->shift,*(dt2 *)mat->traits->scale,*(dt2 *)mat->traits->beta,(dt2 *)cu_localdot);

#define SWITCH_BOOLS(func,dt1,dt2,...)\
            if (options & GHOST_SPMVM_AXPY) {\
                if (options & GHOST_SPMVM_AXPBY) {\
                    if (options & GHOST_SPMVM_APPLY_SCALE) {\
                        if (options & GHOST_SPMVM_APPLY_SHIFT) {\
                            if (options & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {\
                                CALL(func,dt1,dt2,true,true,true,true,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,true,true,true,true,false,__VA_ARGS__)\
                            }\
                        } else {\
                            if (options & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {\
                                CALL(func,dt1,dt2,true,true,true,false,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,true,true,true,false,false,__VA_ARGS__)\
                            }\
                        }\
                    } else {\
                        if (options & GHOST_SPMVM_APPLY_SHIFT) {\
                            if (options & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {\
                                CALL(func,dt1,dt2,true,true,false,true,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,true,true,false,true,false,__VA_ARGS__)\
                            }\
                        } else {\
                            if (options & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {\
                                CALL(func,dt1,dt2,true,true,false,false,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,true,true,false,false,false,__VA_ARGS__)\
                            }\
                        }\
                    }\
                } else {\
                    if (options & GHOST_SPMVM_APPLY_SCALE) {\
                        if (options & GHOST_SPMVM_APPLY_SHIFT) {\
                            if (options & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {\
                                CALL(func,dt1,dt2,true,false,true,true,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,true,false,true,true,false,__VA_ARGS__)\
                            }\
                        } else {\
                            if (options & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {\
                                CALL(func,dt1,dt2,true,false,true,false,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,true,false,true,false,false,__VA_ARGS__)\
                            }\
                        }\
                    } else {\
                        if (options & GHOST_SPMVM_APPLY_SHIFT) {\
                            if (options & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {\
                                CALL(func,dt1,dt2,true,false,false,true,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,true,false,false,true,false,__VA_ARGS__)\
                            }\
                        } else {\
                            if (options & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {\
                                CALL(func,dt1,dt2,true,false,false,false,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,true,false,false,false,false,__VA_ARGS__)\
                            }\
                        }\
                    }\
                }\
            } else {\
                if (options & GHOST_SPMVM_AXPBY) {\
                    if (options & GHOST_SPMVM_APPLY_SCALE) {\
                        if (options & GHOST_SPMVM_APPLY_SHIFT) {\
                            if (options & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {\
                                CALL(func,dt1,dt2,false,true,true,true,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,false,true,true,true,false,__VA_ARGS__)\
                            }\
                        } else {\
                            if (options & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {\
                                CALL(func,dt1,dt2,false,true,true,false,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,false,true,true,false,false,__VA_ARGS__)\
                            }\
                        }\
                    } else {\
                        if (options & GHOST_SPMVM_APPLY_SHIFT) {\
                            if (options & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {\
                                CALL(func,dt1,dt2,false,true,false,true,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,false,true,false,true,false,__VA_ARGS__)\
                            }\
                        } else {\
                            if (options & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {\
                                CALL(func,dt1,dt2,false,true,false,false,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,false,true,false,false,false,__VA_ARGS__)\
                            }\
                        }\
                    }\
                } else {\
                    if (options & GHOST_SPMVM_APPLY_SCALE) {\
                        if (options & GHOST_SPMVM_APPLY_SHIFT) {\
                            if (options & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {\
                                CALL(func,dt1,dt2,false,false,true,true,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,false,false,true,true,false,__VA_ARGS__)\
                            }\
                        } else {\
                            if (options & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {\
                                CALL(func,dt1,dt2,false,false,true,false,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,false,false,true,false,false,__VA_ARGS__)\
                            }\
                        }\
                    } else {\
                        if (options & GHOST_SPMVM_APPLY_SHIFT) {\
                            if (options & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {\
                                CALL(func,dt1,dt2,false,false,false,true,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,false,false,false,true,false,__VA_ARGS__)\
                            }\
                        } else {\
                            if (options & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {\
                                CALL(func,dt1,dt2,false,false,false,false,true,__VA_ARGS__)\
                            } else {\
                                CALL(func,dt1,dt2,false,false,false,false,false,__VA_ARGS__)\
                            }\
                        }\
                    }\
                }\
            }\

#define CHOOSE_KERNEL(dt1,dt2) {\
    static int infoprinted=0;\
    void *cu_localdot = NULL;\
    if ((SELL(mat)->T > 128) || (SELL(mat)->T == 0) || (SELL(mat)->T & (SELL(mat)->T-1)))\
    WARNING_LOG("Invalid T: %d (must be power of two and T <= 128)",SELL(mat)->T);\
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
            SWITCH_BOOLS(SELLT_kernel_CU_tmpl,dt1,dt2,SELL_CUDA_NBLOCKS,block,reqSmem)\
        } else {\
            SWITCH_BOOLS(SELL_kernel_CU_ELLPACK_tmpl,dt1,dt2,SELL_CUDA_NBLOCKS,SELL_CUDA_THREADSPERBLOCK)\
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
            SWITCH_BOOLS(SELLT_kernel_CU_tmpl,dt1,dt2,SELL_CUDA_NBLOCKS,block,reqSmem)\
        } else {\
            SWITCH_BOOLS(SELL_kernel_CU_tmpl,dt1,dt2,SELL_CUDA_NBLOCKS,SELL_CUDA_THREADSPERBLOCK)\
        }\
    }\
    if (options & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {\
        if (!infoprinted)\
            INFO_LOG("Not doing the local dot product on-the-fly!");\
        infoprinted=1;\
        lhs->dotProduct(lhs,lhs,lhs->traits->localdot);\
        lhs->dotProduct(lhs,rhs,(char *)lhs->traits->localdot+sizeof(dt2));\
        lhs->dotProduct(rhs,rhs,(char *)lhs->traits->localdot+2*sizeof(dt2));\
    }\
}

    template<typename m_t, typename v_t, bool do_axpy, bool do_axpby, bool do_scale, bool do_shift, bool do_localdot>  
__global__ void SELL_kernel_CU_ELLPACK_tmpl(v_t *lhs, v_t *rhs, int options, int nrows, int nrowspadded, ghost_midx_t *rowlen, ghost_midx_t *col, m_t *val, ghost_mnnz_t *chunkstart, ghost_midx_t *chunklen, int C, int T, v_t shift, v_t alpha, v_t beta, v_t *localdot)
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
                    lhs[i] = axpy<v_t,float>(lhs[i],scale<v_t>(alpha,axpy<v_t,v_t>(rhs[i],tmp,scale2<v_t,float>(shift,-1.f))),1.f);
                } else if (do_axpby) {
                    lhs[i] = axpy<v_t,float>(scale<v_t>(lhs[i],beta),scale<v_t>(alpha,axpy<v_t,v_t>(rhs[i],tmp,scale2<v_t,float>(shift,-1.f))),1.f);
                } else {
                    lhs[i] = scale<v_t>(alpha,axpy<v_t,v_t>(rhs[i],tmp,scale2<v_t,float>(shift,-1.f)));
                }
            } else {
                if (do_axpy) {
                    lhs[i] = axpy<v_t,float>(lhs[i],axpy<v_t,v_t>(rhs[i],tmp,scale2<v_t,float>(shift,-1.f)),1.f);
                } else if (do_axpby) {
                    lhs[i] = axpy<v_t,float>(scale<v_t>(lhs[i],beta),axpy<v_t,v_t>(rhs[i],tmp,scale2<v_t,float>(shift,-1.f)),1.f);
                } else {
                    lhs[i] = axpy<v_t,v_t>(rhs[i],tmp,scale2<v_t,float>(shift,-1.f));
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
__global__ void SELL_kernel_CU_tmpl(v_t *lhs, v_t *rhs, int options, int nrows, int nrowspadded, ghost_midx_t *rowlen, ghost_midx_t *col, m_t *val, ghost_mnnz_t *chunkstart, ghost_midx_t *chunklen, int C, int T, v_t shift, v_t alpha, v_t beta, v_t *localdot)
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
                    lhs[i] = axpy<v_t,float>(lhs[i],scale<v_t>(alpha,axpy<v_t,v_t>(rhs[i],tmp,scale2<v_t,float>(shift,-1.f))),1.f);
                } else if (do_axpby) {
                    lhs[i] = axpy<v_t,float>(scale<v_t>(lhs[i],beta),scale<v_t>(alpha,axpy<v_t,v_t>(rhs[i],tmp,scale2<v_t,float>(shift,-1.f))),1.f);
                } else {
                    lhs[i] = scale<v_t>(alpha,axpy<v_t,v_t>(rhs[i],tmp,scale2<v_t,float>(shift,-1.f)));
                }
            } else {
                if (do_axpy) {
                    lhs[i] = axpy<v_t,float>(lhs[i],axpy<v_t,v_t>(rhs[i],tmp,scale2<v_t,float>(shift,-1.f)),1.f);
                } else if (do_axpby) {
                    lhs[i] = axpy<v_t,float>(scale<v_t>(lhs[i],beta),axpy<v_t,v_t>(rhs[i],tmp,scale2<v_t,float>(shift,-1.f)),1.f);
                } else {
                    lhs[i] = axpy<v_t,v_t>(rhs[i],tmp,scale2<v_t,float>(shift,-1.f));
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
__global__ void SELLT_kernel_CU_tmpl(v_t *lhs, v_t *rhs, int options, ghost_midx_t nrows, ghost_midx_t nrowspadded, ghost_midx_t *rowlen, ghost_midx_t *col, m_t *val, ghost_mnnz_t *chunkstart, ghost_midx_t *chunklen, ghost_midx_t C, int T, v_t shift, v_t alpha, v_t beta, v_t *localdot)
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
                        lhs[i] = axpy<v_t,float>(lhs[i],scale<v_t>(alpha,axpy<v_t,v_t>(rhs[i],tmp,scale2<v_t,float>(shift,-1.f))),1.f);
                    } else if (do_axpby) {
                        lhs[i] = axpy<v_t,float>(scale<v_t>(lhs[i],beta),scale<v_t>(alpha,axpy<v_t,v_t>(rhs[i],tmp,scale2<v_t,float>(shift,-1.f))),1.f);
                    } else {
                        lhs[i] = scale<v_t>(alpha,axpy<v_t,v_t>(rhs[i],tmp,scale2<v_t,float>(shift,-1.f)));
                    }
                } else {
                    if (do_axpy) {
                        lhs[i] = axpy<v_t,float>(lhs[i],axpy<v_t,v_t>(rhs[i],tmp,scale2<v_t,float>(shift,-1.f)),1.f);
                    } else if (do_axpby) {
                        lhs[i] = axpy<v_t,float>(scale<v_t>(lhs[i],beta),axpy<v_t,v_t>(rhs[i],tmp,scale2<v_t,float>(shift,-1.f)),1.f);
                    } else {
                        lhs[i] = axpy<v_t,v_t>(rhs[i],tmp,scale2<v_t,float>(shift,-1.f));
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


extern "C" void dd_SELL_kernel_CU(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{
    CHOOSE_KERNEL(double,double);
/*
    SELL_kernel_CU_tmpl<double,double,options&GHOST_SPMVM_AXPY,GHOST_SPMVM_AXPBY,GHOST_SPMVM_APPLY_SCALE,GHOST_SPMVM_APPLY_SHIFT,GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT> 
        <<< SELL_CUDA_NBLOCKS,SELL_CUDA_THREADSPERBLOCK >>> ((double *)lhs->CU_val,(double *)rhs->CU_val,options,SELL(mat)->cumat->nrows,SELL(mat)->cumat->nrowsPadded,SELL(mat)->cumat->rowLen,SELL(mat)->cumat->col,(double *)SELL(mat)->cumat->val,SELL(mat)->cumat->chunkStart,SELL(mat)->cumat->chunkLen,SELL(mat)->chunkHeight,SELL(mat)->T,*(double *)mat->traits->shift,*(double *)mat->traits->scale,*(double *)mat->traits->beta,(double *)lhs->traits->localdot);*/
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

