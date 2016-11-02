#include "ghost/config.h"
#include "ghost/sparsemat.h"
#include "ghost/densemat.h"
#include "ghost/sell_spmv_cu_kernel.h"
#include "ghost/sell_spmv_cu_fallback.h"
#include <complex>
#include <cuComplex.h>

    template<typename m_t, typename v_t, typename v_t_b>  
__global__ void SELL_kernel_rm_fallback_tmpl(v_t * const __restrict__ lhs, const int lhs_lda, const v_t * const __restrict__ rhs, const int rhs_lda, const ghost_spmv_flags flags, const int nrows, const int nrowsinblock, const ghost_lidx * const __restrict__ rowlen, const ghost_lidx * const __restrict__ mcol, const m_t * const __restrict__ val, const ghost_lidx * const __restrict__ chunkstart, const v_t * const __restrict__ shift, const v_t alpha, const v_t beta, v_t * const __restrict__ localdot, int C, int ncols, v_t * const __restrict__ z, const int z_lda, const v_t delta, const v_t eta, bool do_axpby, bool do_scale, bool do_vshift, bool do_dot, bool do_chain_axpby)
{
    int ncolsinblock = blockDim.x/nrowsinblock;
    int row = blockIdx.x*nrowsinblock+threadIdx.x/ncolsinblock;
    int col = blockIdx.y*ncolsinblock+threadIdx.x%ncolsinblock;

    if (row<nrows && col<ncols) {
        int cs, ric; // chunkstart and row in chunk
        cs = chunkstart[row/C];
        ric = row%C;
        int j;
        v_t tmp;

        zero<v_t>(tmp);

        for (j=0; j<rowlen[row]; j++) {
            tmp = axpy<v_t,m_t>(tmp, rhs[rhs_lda*mcol[cs + ric + j*C]+col], val[cs+ric+j*C]);
        }

        if (do_vshift) {
            tmp = axpy<v_t,v_t>(tmp,rhs[rhs_lda*row+col],scale2<v_t,float>(shift[col],-1.f));
        }
        if (do_scale) {
            tmp = scale<v_t>(alpha,tmp);
        }
        if (do_axpby) {
            lhs[lhs_lda*row+col] = axpy<v_t,v_t>(tmp,lhs[lhs_lda*row+col],beta);
        } else {
            lhs[lhs_lda*row+col] = tmp;
        }
        if (do_chain_axpby) {
            z[z_lda*row+col] = axpby<v_t>(lhs[lhs_lda*row+col],z[z_lda*row+col],eta,delta);
        }
    }
#ifdef LOCALDOT_ONTHEFLY
    row = blockIdx.x*nrowsinblock+threadIdx.x%nrowsinblock;
    col = blockIdx.y*ncolsinblock+threadIdx.x/nrowsinblock;
    if (col < ncols) {
        if (do_dot) {
            v_t_b dot1, dot3;
            v_t dot2;

            __syncthreads();
            if (row<nrows) {
                dot1 = mulConjSame<v_t,v_t_b>(lhs[lhs_lda*row+col]);
                dot2 = mulConj<v_t>(rhs[rhs_lda*row+col],lhs[lhs_lda*row+col]);
                dot3 = mulConjSame<v_t,v_t_b>(rhs[rhs_lda*row+col]);
            } else {
                zero<v_t_b>(dot1);
                zero<v_t>(dot2);
                zero<v_t_b>(dot3);
            }

            if (nrowsinblock <= 32) {
                dot1 = ghost_partialWarpReduceSum(dot1,nrowsinblock,warpSize);
                dot2 = ghost_partialWarpReduceSum(dot2,nrowsinblock,warpSize);
                dot3 = ghost_partialWarpReduceSum(dot3,nrowsinblock,warpSize);
                if (threadIdx.x%nrowsinblock == 0) {
                    fromReal<v_t,v_t_b>(localdot[0*ncols*gridDim.x + gridDim.x*col + blockIdx.x],dot1);
                    localdot[1*ncols*gridDim.x + gridDim.x*col + blockIdx.x] = dot2;
                    fromReal<v_t,v_t_b>(localdot[2*ncols*gridDim.x + gridDim.x*col + blockIdx.x],dot3);
                }
            } else {
                dot1 = ghost_1dPartialBlockReduceSum(dot1,nrowsinblock/warpSize);
                __syncthreads();
                dot2 = ghost_1dPartialBlockReduceSum(dot2,nrowsinblock/warpSize);
                __syncthreads();
                dot3 = ghost_1dPartialBlockReduceSum(dot3,nrowsinblock/warpSize);
                __syncthreads();

                if ((threadIdx.x<blockDim.x/warpSize) && threadIdx.x%(nrowsinblock/warpSize) == 0) {
                    col=threadIdx.x/(blockDim.x/(warpSize*ncolsinblock));
                    fromReal<v_t,v_t_b>(localdot[0*ncols*gridDim.x + gridDim.x*col + blockIdx.x],dot1);
                    localdot[1*ncols*gridDim.x + gridDim.x*col + blockIdx.x] = dot2;
                    fromReal<v_t,v_t_b>(localdot[2*ncols*gridDim.x + gridDim.x*col + blockIdx.x],dot3);
                }
            }

        }
    }
#endif
}

    template<typename m_t, typename v_t, typename v_t_b>  
__global__ void SELL_kernel_cm_fallback_tmpl(v_t * const __restrict__ lhs, const int lhs_lda, const v_t * const __restrict__ rhs, const int rhs_lda, const ghost_spmv_flags flags, const int nrows, const ghost_lidx * const __restrict__ rowlen, const ghost_lidx * const __restrict__ mcol, const m_t * const __restrict__ val, const ghost_lidx * const __restrict__ chunkstart, const v_t * const __restrict__ shift, const v_t alpha, const v_t beta, v_t * const __restrict__ localdot, int C, int ncols, v_t * const __restrict__ z, const int z_lda, const v_t delta, const v_t eta, bool do_axpby, bool do_scale, bool do_vshift, bool do_dot, bool do_chain_axpby)
{
    int i = threadIdx.x+blockIdx.x*blockDim.x;
    int col = blockDim.y*blockIdx.y+threadIdx.y;

    if (i<nrows && col<ncols) {
        int cs, tid;
        if (C == blockDim.x) {
            cs = chunkstart[blockIdx.x];
            tid = threadIdx.x;
        } else {
            cs = chunkstart[i/C];
            tid = i%C;
        }
        int j;
        v_t tmp;

        zero<v_t>(tmp);

        for (j=0; j<rowlen[i]; j++) {
            tmp = axpy<v_t,m_t>(tmp, rhs[rhs_lda*col+mcol[cs + tid + j*C]], val[cs+tid+j*C]);
        }

        if (do_vshift) {
            tmp = axpy<v_t,v_t>(tmp,rhs[rhs_lda*col+i],scale2<v_t,float>(shift[col],-1.f));
        }
        if (do_scale) {
            tmp = scale<v_t>(alpha,tmp);
        }
        if (do_axpby) {
            lhs[lhs_lda*col+i] = axpy<v_t,float>(scale<v_t>(lhs[lhs_lda*col+i],beta),tmp,1.f);
        } else {
            lhs[lhs_lda*col+i] = tmp;
        }
        if (do_chain_axpby) {
            z[z_lda*col+i] = axpby<v_t>(lhs[lhs_lda*col+i],z[z_lda*col+i],eta,delta);
        }
    }
#ifdef LOCALDOT_ONTHEFLY 
    if (do_dot) {
        v_t_b dot1, dot3;
        v_t dot2;
        zero<v_t_b>(dot1);
        zero<v_t>(dot2);
        zero<v_t_b>(dot3);

        if (i<nrows) {
            dot1 = mulConjSame<v_t,v_t_b>(lhs[lhs_lda*col+i]);
            dot2 = mulConj<v_t>(rhs[rhs_lda*col+i],lhs[lhs_lda*col+i]);
            dot3 = mulConjSame<v_t,v_t_b>(rhs[rhs_lda*col+i]);
        }

        dot1 = ghost_blockReduceSum(dot1);
        __syncthreads();
        dot2 = ghost_blockReduceSum(dot2);
        __syncthreads();
        dot3 = ghost_blockReduceSum(dot3);

        if (threadIdx.x==0) {
            fromReal<v_t,v_t_b>(localdot[0*ncols*gridDim.x + gridDim.x*col + blockIdx.x],dot1);
            localdot[1*ncols*gridDim.x + gridDim.x*col + blockIdx.x] = dot2;
            fromReal<v_t,v_t_b>(localdot[2*ncols*gridDim.x + gridDim.x*col + blockIdx.x],dot3);
        }
    }

#endif
}


    template <typename m_dt, typename v_dt_host, typename v_dt_device, typename v_dt_base>
static ghost_error ghost_sellspmv_cu_tmpl_fallback(ghost_densemat *lhs, ghost_sparsemat *mat, ghost_densemat *rhs, ghost_spmv_opts opts)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH)
        cudaGetLastError(); /* Remove previous error */
    ghost_error ret = GHOST_SUCCESS;
    void *lhsval, *rhsval, *zval;
    ghost_lidx zstride;
    ghost_densemat *lhscompact, *rhscompact, *zcompact;
    if (lhs->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        PERFWARNING_LOG("Cloning (and compressing) lhs before operation");
        GHOST_CALL_RETURN(ghost_densemat_clone(&lhscompact,lhs,lhs->traits.ncols,0));
    } else {
        lhscompact = lhs;
    }
    if (rhs->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        PERFWARNING_LOG("Cloning (and compressing) rhs before operation");
        GHOST_CALL_RETURN(ghost_densemat_clone(&rhscompact,rhs,rhs->traits.ncols,0));
    } else {
        rhscompact = rhs;
    }
    lhsval = lhscompact->cu_val;
    rhsval = rhscompact->cu_val;
    int cu_device;
    GHOST_CALL_RETURN(ghost_cu_device(&cu_device));
    v_dt_device *cu_localdot = NULL;
    v_dt_device *cu_shift = NULL;
    v_dt_host *localdot = NULL;
    v_dt_device *shift, scale, beta, sdelta,seta;
    ghost_densemat *z = NULL;
    dim3 block, grid;
    GHOST_SPMV_PARSE_TRAITS(opts,scale,beta,shift,localdot,z,sdelta,seta,v_dt_host,v_dt_device);
    if (z && (z->traits.flags & GHOST_DENSEMAT_SCATTERED)) {
        PERFWARNING_LOG("Cloning (and compressing) z before operation");
        GHOST_CALL_RETURN(ghost_densemat_clone(&zcompact,z,z->traits.ncols,0));
    } else {
        zcompact = z;
    }
    if (z) {
        zstride = z->stride;
        zval = zcompact->cu_val;
    } else {
        zstride = 0;
    }
    if (opts.flags & GHOST_SPMV_AXPY) {
        v_dt_host hbeta = 1.;
        beta = *((v_dt_device *)&hbeta);
    }
    if (opts.flags & (GHOST_SPMV_SHIFT|GHOST_SPMV_VSHIFT)) {
        size_t shiftsize = sizeof(v_dt_device)*(opts.flags & (GHOST_SPMV_VSHIFT|GHOST_SPMV_SHIFT)?rhs->traits.ncols:0);
        GHOST_CALL_RETURN(ghost_cu_malloc((void **)&cu_shift,shiftsize));
        if (opts.flags & GHOST_SPMV_SHIFT) {
            ghost_lidx c;
            for (c=0; c<rhs->traits.ncols; c++) {
                GHOST_CALL_RETURN(ghost_cu_upload(&cu_shift[c],shift,sizeof(v_dt_device)));
            }
        } else {
            GHOST_CALL_RETURN(ghost_cu_upload(cu_shift,shift,shiftsize));
        }
    }

    ghost_cu_deviceprop prop;
    GHOST_CALL_RETURN(ghost_cu_deviceprop_get(&prop));


    GHOST_INSTR_START("spmv_cuda")
        if (rhs->traits.storage == GHOST_DENSEMAT_COLMAJOR || (rhs->stride == 1)) {
            block.x = PAD(CEILDIV(SELL_CUDA_THREADSPERBLOCK,MIN(MAX_COLS_PER_BLOCK_COLMAJOR,rhs->traits.ncols)),32);
            block.y = MIN(MAX_COLS_PER_BLOCK_COLMAJOR,rhs->traits.ncols);
            while(block.x*block.y > SELL_CUDA_THREADSPERBLOCK && block.x > 32) {
                block.x -= 32;
            }

            grid.x = CEILDIV(SPM_NROWSPAD(mat),block.x);
            grid.y = CEILDIV(rhs->traits.ncols,MAX_COLS_PER_BLOCK_COLMAJOR);
            if (opts.flags & GHOST_SPMV_DOT) {
                GHOST_CALL_RETURN(ghost_cu_malloc((void **)&cu_localdot,sizeof(v_dt_device)*rhs->traits.ncols*3*grid.x));
            }
            size_t smem = 0;
            if (opts.flags & GHOST_SPMV_DOT) {
                smem = sizeof(v_dt_device)*32*block.y;
            }
            if (prop.sharedMemPerBlock < smem) {
                WARNING_LOG("Not enough shared memory available! CUDA kernel will not execute!");
            }
            DEBUG_LOG(1,"grid %dx%d block %dx%d shmem %zu",grid.x,grid.y,block.x,block.y,smem);
            SELL_kernel_cm_fallback_tmpl<m_dt,v_dt_device,v_dt_base><<<grid,block,smem>>>(
                    (v_dt_device *)lhsval,(int)(lhs->stride),(v_dt_device *)rhsval,
                    (int)rhs->stride,opts.flags,SPM_NROWS(mat),mat->cu_rowLen,
                    mat->cu_col,(m_dt *)mat->cu_val,
                    mat->cu_chunkStart,(v_dt_device *)cu_shift,(v_dt_device)scale,
                    (v_dt_device)beta,(v_dt_device *)cu_localdot,mat->traits.C,
                    rhs->traits.ncols,(v_dt_device *)zval,zstride,sdelta,seta,opts.flags&(GHOST_SPMV_AXPBY|GHOST_SPMV_AXPY),
                    opts.flags&GHOST_SPMV_SCALE,
                    opts.flags&(GHOST_SPMV_VSHIFT|GHOST_SPMV_SHIFT),
                    opts.flags&GHOST_SPMV_DOT,
                    opts.flags&GHOST_SPMV_CHAIN_AXPBY);
        } else {
            if (rhs->traits.ncols > 8) { // 16 rows per block, 9...16 columns per block, 144...256 threads per block
                const int nrowsinblock = 16;
                grid.x = CEILDIV(SPM_NROWS(mat),nrowsinblock);
                grid.y = CEILDIV(rhs->traits.ncols,nrowsinblock);
                block.x = nrowsinblock*CEILDIV(rhs->traits.ncols,grid.y);
                if (opts.flags & GHOST_SPMV_DOT) {
                    GHOST_CALL_RETURN(ghost_cu_malloc((void **)&cu_localdot,sizeof(v_dt_device)*rhs->traits.ncols*3*grid.x));
                }
                DEBUG_LOG(1,"grid %dx%d block %dx%d nrowsinblock %d",grid.x,grid.y,block.x,block.y,nrowsinblock);
                SELL_kernel_rm_fallback_tmpl<m_dt,v_dt_device,v_dt_base><<<grid,block,0>>>(
                        (v_dt_device *)lhsval,(int)(lhs->stride),(v_dt_device *)rhsval,
                        (int)rhs->stride,opts.flags,SPM_NROWS(mat),nrowsinblock,mat->cu_rowLen,
                        mat->cu_col,(m_dt *)mat->cu_val,
                        mat->cu_chunkStart,(v_dt_device *)cu_shift,(v_dt_device)scale,
                        (v_dt_device)beta,(v_dt_device *)cu_localdot,mat->traits.C,
                        rhs->traits.ncols,(v_dt_device *)zval,zstride,sdelta,seta,opts.flags&(GHOST_SPMV_AXPBY|GHOST_SPMV_AXPY),
                        opts.flags&GHOST_SPMV_SCALE,
                        opts.flags&(GHOST_SPMV_VSHIFT|GHOST_SPMV_SHIFT),
                        opts.flags&GHOST_SPMV_DOT,
                        opts.flags&GHOST_SPMV_CHAIN_AXPBY);
            } else if (rhs->traits.ncols > 4) { // 32 rows per block, 5...8 columns per block, 160...256 threads per block
                const int nrowsinblock = 32;
                grid.x = CEILDIV(SPM_NROWS(mat),nrowsinblock);
                grid.y = 1;
                block.x = nrowsinblock*rhs->traits.ncols;
                if (opts.flags & GHOST_SPMV_DOT) {
                    GHOST_CALL_RETURN(ghost_cu_malloc((void **)&cu_localdot,sizeof(v_dt_device)*rhs->traits.ncols*3*grid.x));
                }
                DEBUG_LOG(1,"grid %dx%d block %dx%d nrowsinblock %d",grid.x,grid.y,block.x,block.y,nrowsinblock);
                SELL_kernel_rm_fallback_tmpl<m_dt,v_dt_device,v_dt_base><<<grid,block,0>>>(
                        (v_dt_device *)lhsval,(int)(lhs->stride),(v_dt_device *)rhsval,
                        (int)rhs->stride,opts.flags,SPM_NROWS(mat),nrowsinblock,mat->cu_rowLen,
                        mat->cu_col,(m_dt *)mat->cu_val,
                        mat->cu_chunkStart,(v_dt_device *)cu_shift,(v_dt_device)scale,
                        (v_dt_device)beta,(v_dt_device *)cu_localdot,mat->traits.C,
                        rhs->traits.ncols,(v_dt_device *)zval,zstride,sdelta,seta,opts.flags&(GHOST_SPMV_AXPBY|GHOST_SPMV_AXPY),
                        opts.flags&GHOST_SPMV_SCALE,
                        opts.flags&(GHOST_SPMV_VSHIFT|GHOST_SPMV_SHIFT),
                        opts.flags&GHOST_SPMV_DOT,
                        opts.flags&GHOST_SPMV_CHAIN_AXPBY);
            } else if (rhs->traits.ncols > 2) { // 64 rows per block, 3...4 columns per block, 192...256 threads per block
                const int nrowsinblock = 64;
                grid.x = CEILDIV(SPM_NROWS(mat),nrowsinblock);
                grid.y = 1;
                block.x = nrowsinblock*rhs->traits.ncols;
                if (opts.flags & GHOST_SPMV_DOT) {
                    GHOST_CALL_RETURN(ghost_cu_malloc((void **)&cu_localdot,sizeof(v_dt_device)*rhs->traits.ncols*3*grid.x));
                }
                int smem = (block.x/32)*sizeof(v_dt_device);
                DEBUG_LOG(1,"grid %dx%d block %dx%d nrowsinblock %d smem %d",grid.x,grid.y,block.x,block.y,nrowsinblock,smem);
                SELL_kernel_rm_fallback_tmpl<m_dt,v_dt_device,v_dt_base><<<grid,block,smem>>>(
                        (v_dt_device *)lhsval,(int)(lhs->stride),(v_dt_device *)rhsval,
                        (int)rhs->stride,opts.flags,SPM_NROWS(mat),nrowsinblock,mat->cu_rowLen,
                        mat->cu_col,(m_dt *)mat->cu_val,
                        mat->cu_chunkStart,(v_dt_device *)cu_shift,(v_dt_device)scale,
                        (v_dt_device)beta,(v_dt_device *)cu_localdot,mat->traits.C,
                        rhs->traits.ncols,(v_dt_device *)zval,zstride,sdelta,seta,opts.flags&(GHOST_SPMV_AXPBY|GHOST_SPMV_AXPY),
                        opts.flags&GHOST_SPMV_SCALE,
                        opts.flags&(GHOST_SPMV_VSHIFT|GHOST_SPMV_SHIFT),
                        opts.flags&GHOST_SPMV_DOT,
                        opts.flags&GHOST_SPMV_CHAIN_AXPBY);
            } else if (rhs->traits.ncols > 1) { // 64 rows per block, 3...4 columns per block, 192...256 threads per block
                const int nrowsinblock = 128;
                grid.x = CEILDIV(SPM_NROWS(mat),nrowsinblock);
                grid.y = 1;
                block.x = nrowsinblock*rhs->traits.ncols;
                if (opts.flags & GHOST_SPMV_DOT) {
                    GHOST_CALL_RETURN(ghost_cu_malloc((void **)&cu_localdot,sizeof(v_dt_device)*rhs->traits.ncols*3*grid.x));
                }
                int smem = (block.x/32)*sizeof(v_dt_device);
                DEBUG_LOG(1,"grid %dx%d block %dx%d nrowsinblock %d smem %d",grid.x,grid.y,block.x,block.y,nrowsinblock,smem);
                SELL_kernel_rm_fallback_tmpl<m_dt,v_dt_device,v_dt_base><<<grid,block,smem>>>(
                        (v_dt_device *)lhsval,(int)(lhs->stride),(v_dt_device *)rhsval,
                        (int)rhs->stride,opts.flags,SPM_NROWS(mat),nrowsinblock,mat->cu_rowLen,
                        mat->cu_col,(m_dt *)mat->cu_val,
                        mat->cu_chunkStart,(v_dt_device *)cu_shift,(v_dt_device)scale,
                        (v_dt_device)beta,(v_dt_device *)cu_localdot,mat->traits.C,
                        rhs->traits.ncols,(v_dt_device *)zval,zstride,sdelta,seta,opts.flags&(GHOST_SPMV_AXPBY|GHOST_SPMV_AXPY),
                        opts.flags&GHOST_SPMV_SCALE,
                        opts.flags&(GHOST_SPMV_VSHIFT|GHOST_SPMV_SHIFT),
                        opts.flags&GHOST_SPMV_DOT,
                        opts.flags&GHOST_SPMV_CHAIN_AXPBY);
            } else {
                const int nrowsinblock = 256;
                grid.x = CEILDIV(SPM_NROWS(mat),nrowsinblock);
                grid.y = 1;
                block.x = nrowsinblock*rhs->traits.ncols;
                if (opts.flags & GHOST_SPMV_DOT) {
                    GHOST_CALL_RETURN(ghost_cu_malloc((void **)&cu_localdot,sizeof(v_dt_device)*rhs->traits.ncols*3*grid.x));
                }
                int smem = (block.x/32)*sizeof(v_dt_device);
                DEBUG_LOG(1,"grid %dx%d block %dx%d nrowsinblock %d smem %d",grid.x,grid.y,block.x,block.y,nrowsinblock,smem);
                SELL_kernel_rm_fallback_tmpl<m_dt,v_dt_device,v_dt_base><<<grid,block,smem>>>(
                        (v_dt_device *)lhsval,(int)(lhs->stride),(v_dt_device *)rhsval,
                        (int)rhs->stride,opts.flags,SPM_NROWS(mat),nrowsinblock,mat->cu_rowLen,
                        mat->cu_col,(m_dt *)mat->cu_val,
                        mat->cu_chunkStart,(v_dt_device *)cu_shift,(v_dt_device)scale,
                        (v_dt_device)beta,(v_dt_device *)cu_localdot,mat->traits.C,
                        rhs->traits.ncols,(v_dt_device *)zval,zstride,sdelta,seta,opts.flags&(GHOST_SPMV_AXPBY|GHOST_SPMV_AXPY),
                        opts.flags&GHOST_SPMV_SCALE,
                        opts.flags&(GHOST_SPMV_VSHIFT|GHOST_SPMV_SHIFT),
                        opts.flags&GHOST_SPMV_DOT,
                        opts.flags&GHOST_SPMV_CHAIN_AXPBY);
            }
        }
    CUDA_CALL_RETURN(cudaGetLastError());

    if (lhscompact != lhs) {
        DEBUG_LOG(1,"Transform lhs back");
        GHOST_CALL_RETURN(ghost_densemat_init_densemat(lhs,lhscompact,0,0));
        ghost_densemat_destroy(lhscompact);
    }
    if (rhscompact != rhs) {
        DEBUG_LOG(1,"Transform rhs back");
        GHOST_CALL_RETURN(ghost_densemat_init_densemat(rhs,rhscompact,0,0));
        ghost_densemat_destroy(rhscompact);
    }
    cudaDeviceSynchronize();
    GHOST_INSTR_STOP("spmv_cuda")
        if (opts.flags & GHOST_SPMV_DOT) {
#ifdef LOCALDOT_ONTHEFLY
            GHOST_INSTR_START("spmv_cuda_dot_reduction")
                ghost_lidx col;
            for (col=0; col<rhs->traits.ncols; col++) {
                deviceReduce3<v_dt_device>(&cu_localdot[grid.x*col], &cu_localdot[col], rhs->traits.ncols*grid.x, grid.x);
            }
            if (opts.flags & GHOST_SPMV_DOT_YY) {
                GHOST_CALL_RETURN(ghost_cu_download(localdot,cu_localdot,rhs->traits.ncols*sizeof(v_dt_host)));
            }
            if (opts.flags & GHOST_SPMV_DOT_XY) {
                GHOST_CALL_RETURN(ghost_cu_download(&localdot[rhs->traits.ncols],&cu_localdot[rhs->traits.ncols*grid.x],rhs->traits.ncols*sizeof(v_dt_host)));
            }
            if (opts.flags & GHOST_SPMV_DOT_XX) {
                GHOST_CALL_RETURN(ghost_cu_download(&localdot[2*rhs->traits.ncols],&cu_localdot[2*rhs->traits.ncols*grid.x],rhs->traits.ncols*sizeof(v_dt_host)));
            }
            GHOST_INSTR_STOP("spmv_cuda_dot_reduction")
#else
                GHOST_INSTR_START("spmv_cuda_dot")
                PERFWARNING_LOG("Not doing the local dot product on-the-fly!");
            memset(localdot,0,rhs->traits.ncols*3*sizeof(v_dt_host));
            lhs->localdot_vanilla(lhs,&localdot[0],lhs);
            lhs->localdot_vanilla(lhs,&localdot[rhs->traits.ncols],rhs);
            rhs->localdot_vanilla(rhs,&localdot[2*rhs->traits.ncols],rhs);
            GHOST_INSTR_STOP("spmv_cuda_dot")
#endif
        }
    //if (traits.flags & GHOST_SPMV_CHAIN_AXPBY) {
    //    PERFWARNING_LOG("AXPBY will not be done on-the-fly!");
    //    z->axpby(z,lhs,&seta,&sdelta);
    // }
    if (opts.flags & GHOST_SPMV_DOT) {
        GHOST_CALL_RETURN(ghost_cu_free(cu_localdot));
    }
    if (opts.flags & (GHOST_SPMV_SHIFT|GHOST_SPMV_VSHIFT)) {
        GHOST_CALL_RETURN(ghost_cu_free(cu_shift));
    }
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);
    return ret;


}

ghost_error ghost_sellspmv_cu_fallback_selector(ghost_densemat *lhs, ghost_sparsemat *mat, ghost_densemat *rhs, ghost_spmv_opts opts)
{
    ghost_error ret;
    SELECT_TMPL_4DATATYPES(mat->traits.datatype,rhs->traits.datatype,std::complex,ret,ghost_sellspmv_cu_tmpl_fallback,lhs,mat,rhs,opts);
    return ret;
}
