#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/math.h"
#include "ghost/locality.h"
#include "ghost/blas_mangle.h"
#include "ghost/blas_util.h"

#include <strings.h>
#ifdef GHOST_HAVE_CUDA
#include <cublas_v2.h>
#endif

ghost_error_t ghost_gemm_valid(ghost_densemat_t *x, ghost_densemat_t *v, const char * transv, 
ghost_densemat_t *w, const char *transw, void *alpha, void *beta, int reduce,ghost_gemm_flags_t flags, int printerror) 
{
    if (v->traits.datatype != w->traits.datatype) {
        if (printerror) {
            ERROR_LOG("GEMM with mixed datatypes does not work!");
        }
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
    if ((v->traits.location != w->traits.location) || (v->traits.location != x->traits.location)) { 
        ERROR_LOG("The location of all densemats has to be uniform (host or device)!");
        return GHOST_ERR_INVALID_ARG;
    }
    

    ghost_lidx_t nrV,ncV,nrW,ncW,nrX,ncX;

    if (strncasecmp(transv,"N",1)) {
        nrV=v->traits.ncols; ncV=v->traits.nrows;
    } else {
        nrV=v->traits.nrows; ncV=v->traits.ncols;
    }
    if (strncasecmp(transw,"N",1)) {
        nrW=w->traits.ncols; ncW=w->traits.nrows;
    } else {
        nrW=w->traits.nrows; ncW=w->traits.ncols;
    }

    nrX=x->traits.nrows;
    ncX=x->traits.ncols;
    
    if (ncV!=nrW || nrV!=nrX || ncW!=ncX) {
        if (printerror) {
            ERROR_LOG("GEMM with incompatible vectors: %"PRLIDX"x%"PRLIDX" * %"PRLIDX"x%"PRLIDX" = %"PRLIDX"x%"PRLIDX,nrV,ncV,nrW,ncW,nrX,ncX);
        }
        return GHOST_ERR_INVALID_ARG;
    }

    return GHOST_SUCCESS;

}


static ghost_error_t ghost_gemm_blas(ghost_densemat_t *x_in, ghost_densemat_t *v_in, const char * transv_in, 
ghost_densemat_t *w_in, const char *transw_in, void *alpha, void *beta, int reduce,ghost_gemm_flags_t flags) 
{
    UNUSED(flags);

    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    ghost_error_t ret = GHOST_SUCCESS;
    char transv[1], transw[1];
    
    transv[0]=transv_in[0];
    transw[0]=transw_in[0];
    
    ghost_densemat_t *x = x_in;
    ghost_densemat_t *v = v_in;
    ghost_densemat_t *w = w_in;
    
    
    /*if ((v->traits.storage != w->traits.storage) || (x->traits.storage != w->traits.storage)){
        ERROR_LOG("Different storage layouts of input densemats!");
        return GHOST_ERR_INVALID_ARG;
    }*/

    if (v == x && !(flags & GHOST_GEMM_NOT_CLONE_ALIASED)) {
        WARNING_LOG("x equals v! v will be cloned.");
        ghost_densemat_t *vc;
        v->clone(v,&vc,v->traits.nrows,0,v->traits.ncols,0);
        v = vc;
    }

    if (w == x && !(flags & GHOST_GEMM_NOT_CLONE_ALIASED)) {
        WARNING_LOG("x equals w! w will be cloned.");
        ghost_densemat_t *wc;
        w->clone(w,&wc,w->traits.nrows,0,w->traits.ncols,0);
        w = wc;
    }
    
    if (v->context == NULL && w->context == NULL && x->context == NULL && reduce != GHOST_GEMM_NO_REDUCE) {
        INFO_LOG("Reduction should be done but none of the vectors has a context. Ommitting the reduction...");
        reduce = GHOST_GEMM_NO_REDUCE;
    }

    int nranks = 1;
    if (v->context) {
        GHOST_CALL_GOTO(ghost_nrank(&nranks, v->context->mpicomm),err,ret);
    }

    if ((reduce != GHOST_GEMM_NO_REDUCE) && (reduce >= nranks)) {
        WARNING_LOG("Reduction should be done to rank %d but only %d ranks are present. Reducing to 0...",
                reduce,nranks);
        reduce = 0;
    }

    ghost_lidx_t nrV,ncV,nrW,ncW,nrX,ncX;

    if (strncasecmp(transv_in,"N",1)) {
        nrV=v->traits.ncols; ncV=v->traits.nrows;
    } else {
        nrV=v->traits.nrows; ncV=v->traits.ncols;
    }
    if (strncasecmp(transw_in,"N",1)) {
        nrW=w->traits.ncols; ncW=w->traits.nrows;
    } else {
        nrW=w->traits.nrows; ncW=w->traits.ncols;
    }

    nrX=x->traits.nrows;
    ncX=x->traits.ncols;
    
    complex double zero = 0.+I*0.;

    ghost_blas_idx_t *m, *n, *k;
    m = (ghost_blas_idx_t *)&nrV;
    k = (ghost_blas_idx_t *)&ncV;
    n = (ghost_blas_idx_t *)&ncW;

    ghost_blas_idx_t *ldv = &v->stride;
    ghost_blas_idx_t *ldw = &w->stride;
    ghost_blas_idx_t *ldx = &x->stride;

    void *vdata = NULL;
    void *wdata = NULL;
    void *xdata = NULL;
    GHOST_CALL_GOTO(ghost_densemat_valptr(v,&vdata),err,ret);
    GHOST_CALL_GOTO(ghost_densemat_valptr(w,&wdata),err,ret);
    GHOST_CALL_GOTO(ghost_densemat_valptr(x,&xdata),err,ret);
    
    //note: if no reduction is requested, none of the input vecs may have
    // a context (or an MPI comm). If any reduction is requested, only v
    // needs context and comm, the others may be replicated. So we must
    // take the comm from v if and only if a reduction is requested.
    int myrank=0;

    if ((reduce != GHOST_GEMM_NO_REDUCE) && (v->context)) {
        GHOST_CALL_GOTO(ghost_rank(&myrank,v->context->mpicomm),err,ret);
    }

    void *mybeta;

      // careful, we should only access the comm of v, and only if 
      // a reduction operation is requested. The user may have all matrices
      // local as long as he does not request a reduction operation, or he
      // may have w and/or x local in a distributed context
    if (((reduce == GHOST_GEMM_ALL_REDUCE) && (myrank == 0))     ||
        ((reduce != GHOST_GEMM_NO_REDUCE) && (myrank == reduce)) ||
         (reduce == GHOST_GEMM_NO_REDUCE)                         ) 
    { // make sure that the initial value of x only gets added up once
        mybeta = beta;
    }
    else 
    {
        mybeta = &zero;
    }
    DEBUG_LOG(1,"Calling XGEMM with (%"PRBLASIDX"x%"PRBLASIDX") * (%"PRBLASIDX"x%"PRBLASIDX") = (%"PRBLASIDX"x%"PRBLASIDX")",*m,*k,*k,*n,*m,*n);
       /* char *xstr, *vstr, *wstr;
        char *xvstr, *vvstr, *wvstr;
        ghost_densemat_info_string(&xstr,x);
        ghost_densemat_info_string(&vstr,v);
        ghost_densemat_info_string(&wstr,w);
        x->string(x,&xvstr);
        v->string(v,&vvstr);
        w->string(w,&wvstr);

        printf("\nv\n%s\n%s\nw\n%s\n%s\nx\n%s\n%s\n\n",vstr,vvstr,wstr,wvstr,xstr,xvstr);
        WARNING_LOG("GEMM %dx%d * %dx%d = %dx%d",*m,*k,*k,*n,*m,*n);
        WARNING_LOG("%s %s",transv,transw);*/
    if ((v->traits.location == w->traits.location) && (v->traits.location ==  x->traits.location) && 
            (v->traits.location == GHOST_LOCATION_DEVICE)) {
#ifdef GHOST_HAVE_CUDA
        cublasHandle_t ghost_cublas_handle;
        ghost_blas_idx_t culdv,culdw,culdx;
        
        GHOST_CALL_GOTO(ghost_cu_cublas_handle(&ghost_cublas_handle),err,ret); 
        cublasOperation_t cutransv = !strncasecmp(transv,"T",1)?CUBLAS_OP_T:!strncasecmp(transv,"C",1)?CUBLAS_OP_C:CUBLAS_OP_N;
        cublasOperation_t cutransw = !strncasecmp(transw,"T",1)?CUBLAS_OP_T:!strncasecmp(transw,"C",1)?CUBLAS_OP_C:CUBLAS_OP_N;

        culdv = *ldv;
        culdw = *ldw;
        culdx = *ldx;

        void *xcuval;
        void *vcuval;
        void *wcuval;
        GHOST_CALL_GOTO(ghost_densemat_cu_valptr(x,&xcuval),err,ret);
        GHOST_CALL_GOTO(ghost_densemat_cu_valptr(v,&vcuval),err,ret);
        GHOST_CALL_GOTO(ghost_densemat_cu_valptr(w,&wcuval),err,ret);


        /*if (v->traits.storage == GHOST_DENSEMAT_ROWMAJOR && w->traits.storage == GHOST_DENSEMAT_ROWMAJOR && (cutransv == CUBLAS_OP_T || cutransv == CUBLAS_OP_C) && cutransw == CUBLAS_OP_N) {
            INFO_LOG("special case 1");
            cutransw = cutransv;
            cutransv = CUBLAS_OP_N;
        }
        if ((v->traits.storage != w->traits.storage) && cutransv == CUBLAS_OP_N && cutransw == CUBLAS_OP_N) {
            INFO_LOG("special case 2");
            //needMemTransposeX = 1;
            cutransv = CUBLAS_OP_T;
            culdx = x->traits.nrows;
        }*/

        // TSMM hack
        if (v->traits.storage == GHOST_DENSEMAT_ROWMAJOR && x->traits.storage == GHOST_DENSEMAT_ROWMAJOR &&
                !strncasecmp(transv_in,"N",1) && !strncasecmp(transw_in,"N",1)) {
            INFO_LOG("v and x are row-major and no transposing should be done. Tricking CUBLAS into thinking they are col-major!");
            
            void *tmp;
            tmp = vcuval;
            vcuval = wcuval;
            wcuval = tmp;

            tmp = m;
            m = n;
            n = tmp;
            
            ghost_blas_idx_t ldtmp = culdv;
            culdv = culdw;
            culdw = ldtmp;

            if (w->traits.storage == GHOST_DENSEMAT_COLMAJOR) {
                cutransv = CUBLAS_OP_T;
            }
        }
        
        // TSMTTSM hack
        if (v->traits.storage == GHOST_DENSEMAT_ROWMAJOR && w->traits.storage == GHOST_DENSEMAT_ROWMAJOR &&
                strncasecmp(transv_in,"N",1) && !strncasecmp(transw_in,"N",1)) {
            INFO_LOG("v and w are row-major and v is (conjugate) transposed. Tricking CUBLAS into thinking they are col-major!");

            /* 
             * Double-transpose W and interpret V as col-major as it is transposed
             *
             * [CM]   [CM ]   [CM ] 
             *
             * (X ) = (V^T) * (W^T)^T
             */

            cutransw = cutransv; 
            cutransv = CUBLAS_OP_N;
            
            if (x->traits.storage == GHOST_DENSEMAT_ROWMAJOR) {
                
                /* 
                 * Fake-transpose X and interchange V and W
                 *
                 * [CM ]   [CM ]    [CM ] 
                 *
                 * (X^T) = (W^T)^T * V^T
                 */
                void *tmp;
                tmp = vcuval;
                vcuval = wcuval;
                wcuval = tmp;

                tmp = m;
                m = n;
                n = tmp;
                
                ghost_blas_idx_t ldtmp = culdv;
                culdv = culdw;
                culdw = ldtmp;
            }
        }
            
/*
        if (v->traits.storage == GHOST_DENSEMAT_ROWMAJOR) {
            if (cutransv == CUBLAS_OP_T) {
                WARNING_LOG("v T->N");
                cutransv = CUBLAS_OP_N;
            } else if (cutransv == CUBLAS_OP_N) {
                //cutransv = CUBLAS_OP_T;
                //WARNING_LOG("v N->T");
            }
            culdv = v->traits.ncolspadded;//colspadded;
            WARNING_LOG("ldv %d->%d",*ldv,culdv);
        }
        if (w->traits.storage == GHOST_DENSEMAT_ROWMAJOR) {
            if (cutransw == CUBLAS_OP_T) {
                //cutransw = CUBLAS_OP_N;
                //WARNING_LOG("w T->N");
            } else if (cutransw == CUBLAS_OP_N) {
                cutransw = CUBLAS_OP_T;
                WARNING_LOG("w N->T");
            }
            culdw = w->traits.ncolspadded;
            WARNING_LOG("ldw %d->%d",*ldw,culdw);
        }
        if (x->traits.storage == GHOST_DENSEMAT_ROWMAJOR) {
            culdx = x->traits.nrows;//colspadded;
            WARNING_LOG("ldx %d->%d",*ldx,culdx);
        }
        WARNING_LOG("CUBLAS GEMM v %s:%d ::: w %s:%d ::: x %s:%d",ghost_densemat_storage_string(v),culdv,ghost_densemat_storage_string(w),culdw,ghost_densemat_storage_string(x),culdx);*/


        if (v->traits.datatype & GHOST_DT_COMPLEX) 
        {
            if (v->traits.datatype & GHOST_DT_DOUBLE) 
            {
                CUBLAS_CALL_GOTO(cublasZgemm(ghost_cublas_handle,cutransv,cutransw,*m,*n,*k,(cuDoubleComplex *)alpha,(cuDoubleComplex *)vcuval,culdv,(cuDoubleComplex *)wcuval,culdw,(cuDoubleComplex *)mybeta,(cuDoubleComplex *)xcuval,culdx),err,ret);
            } 
            else 
            {
                CUBLAS_CALL_GOTO(cublasCgemm(ghost_cublas_handle,cutransv,cutransw,*m,*n,*k,(cuFloatComplex *)alpha,(cuFloatComplex *)vcuval,culdv,(cuFloatComplex *)wcuval,culdw,(cuFloatComplex *)mybeta,(cuFloatComplex *)xcuval,culdx),err,ret);
            }
        } 
        else 
        {
            if (v->traits.datatype & GHOST_DT_DOUBLE) 
            {
                CUBLAS_CALL_GOTO(cublasDgemm(ghost_cublas_handle,cutransv,cutransw,*m,*n,*k,(double *)alpha,(double *)vcuval,culdv,(double *)wcuval,culdw,(double *)mybeta,(double *)xcuval,culdx),err,ret);
            } 
            else 
            {
                CUBLAS_CALL_GOTO(cublasSgemm(ghost_cublas_handle,cutransv,cutransw,*m,*n,*k,(float *)alpha,(float *)vcuval,culdv,(float *)wcuval,culdw,(float *)mybeta,(float *)xcuval,culdx),err,ret);
            }    
        }
#endif
    } else
    if ((v->traits.location == w->traits.location) && (v->traits.location ==  x->traits.location) && 
            (v->traits.location == GHOST_LOCATION_HOST)) {
        if (v->traits.datatype & GHOST_DT_COMPLEX) 
        {
            if (v->traits.datatype & GHOST_DT_DOUBLE) 
            {
                BLAS_CALL_GOTO(zgemm(v->traits.storage,transv,transw, m,n, k, alpha, vdata, ldv, wdata, ldw, mybeta, xdata, ldx),err,ret);
            } 
            else 
            {
                BLAS_CALL_GOTO(cgemm(v->traits.storage,transv,transw, m,n, k, alpha, vdata, ldv, wdata, ldw, mybeta, xdata, ldx),err,ret);
            }
        } 
        else 
        {
            if (v->traits.datatype & GHOST_DT_DOUBLE) 
            {
                BLAS_CALL_GOTO(dgemm(v->traits.storage,transv,transw, m,n, k, (double *)alpha, vdata, ldv, wdata, ldw, (double *)mybeta, xdata, ldx),err,ret);
            } 
            else 
            {
                BLAS_CALL_GOTO(sgemm(v->traits.storage,transv,transw, m,n, k, (float *)alpha, vdata, ldv, wdata, ldw, (float *)mybeta, xdata, ldx),err,ret);
            }    
        }
    }
        /*x->string(x,&xvstr);

        printf("\n\n%s\n\n",xvstr);*/

#ifdef GHOST_HAVE_MPI 
    ghost_lidx_t i;
    if (reduce != GHOST_GEMM_NO_REDUCE) {

#ifdef GHOST_HAVE_CUDA
        ghost_lidx_t lda = x->stride;
#endif
        ghost_lidx_t dima;
        ghost_lidx_t dimb;
        if (x->traits.storage == GHOST_DENSEMAT_ROWMAJOR) {
            dima = x->traits.nrows;
            dimb = x->traits.ncols;
        } else if (x->traits.storage == GHOST_DENSEMAT_COLMAJOR) {
            dima = x->traits.ncols;
            dimb = x->traits.nrows;
        } else {
            ERROR_LOG("Invalid vector storage");
            return GHOST_ERR_NOT_IMPLEMENTED;
        }


        for (i=0; i<dima; ++i) {
            int copied = 0;
            void *val = NULL;
            if (x->traits.location == GHOST_LOCATION_DEVICE) {
#ifdef GHOST_HAVE_CUDA
                size_t sizeofdt;
                ghost_datatype_size(&sizeofdt,x->traits.datatype);

                GHOST_CALL_GOTO(ghost_malloc((void **)&val,dimb*sizeofdt),err,ret);
                ghost_cu_download(val, &x->cu_val[i*lda*sizeofdt], dimb*sizeofdt);
                copied = 1;
#endif
            } else if (x->traits.location & GHOST_LOCATION_HOST) {
                val = x->val+i*x->stride*x->elSize;
            }
            ghost_mpi_op_t sumOp;
            ghost_mpi_datatype_t mpiDt;
            GHOST_CALL_GOTO(ghost_mpi_op_sum(&sumOp,x->traits.datatype),err,ret);
            GHOST_CALL_GOTO(ghost_mpi_datatype(&mpiDt,x->traits.datatype),err,ret);

            if (v->context) {
                if (reduce == GHOST_GEMM_ALL_REDUCE) {
                    MPI_CALL_GOTO(MPI_Allreduce(MPI_IN_PLACE,val,dimb,mpiDt,sumOp,v->context->mpicomm),err,ret);
                } else {
                    if (myrank == reduce) 
                    {
                        MPI_CALL_GOTO(MPI_Reduce(MPI_IN_PLACE,val,dimb,mpiDt,sumOp,reduce,v->context->mpicomm),err,ret);
                    } 
                    else 
                    {
                        MPI_CALL_GOTO(MPI_Reduce(val,NULL,dimb,mpiDt,sumOp,reduce,v->context->mpicomm),err,ret);
                    }
                }
                if (copied) {
#ifdef GHOST_HAVE_CUDA
                    size_t sizeofdt;
                    ghost_datatype_size(&sizeofdt,x->traits.datatype);
                    GHOST_CALL_GOTO(ghost_cu_upload(&x->cu_val[i*lda*sizeofdt],val,dimb*sizeofdt),err,ret);
                    free(val);
#endif
                }
            }
        }
    }
#else
    UNUSED(reduce);
#endif

#ifdef GHOST_HAVE_INSTR_TIMING
    ghost_gemm_perf_args_t gemm_perfargs;
    gemm_perfargs.xcols = x->traits.ncols;
    gemm_perfargs.vcols = v->traits.ncols;
    if (v->context) {
        gemm_perfargs.vrows = v->context->gnrows;
    } else {
        gemm_perfargs.vrows = v->traits.nrows;
    }
    gemm_perfargs.dt = x->traits.datatype;
    ghost_timing_set_perfFunc(__ghost_functag,ghost_gemm_perf_GFs,(void *)&gemm_perfargs,sizeof(gemm_perfargs),"GF/s");
#endif

    //char *str;
    //x->string(x,&str);
    //printf("$$$$$\n%s\n$$$$$",str);
    
    goto out;
err:

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL)
    return ret;

}

ghost_error_t ghost_gemm(ghost_densemat_t *x_in, ghost_densemat_t *v_in, const char * transv, 
ghost_densemat_t *w_in, const char *transw, void *alpha, void *beta, int reduce,ghost_gemm_flags_t flags) 
{
#ifdef GHOST_HAVE_LONGIDX_LOCAL
#ifndef GHOST_HAVE_MKL
    WARNING_LOG("Will cast 64-bit indices to 32 bit for non-MKL GEMM with LONGIDX");
#endif
#endif
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret = GHOST_SUCCESS;
    int donespecial = 0;
      
    ghost_densemat_t *x = NULL;
    ghost_densemat_t *v = v_in;
    ghost_densemat_t *w = w_in;
    
    // scattered vectors are copied together, if this occurs the user should rethink his or 
    // her data layout.
    if (x_in->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        INFO_LOG("The result vector x is scattered. It will be cloned and compressed before the computation and transformed back afterwards.");
        GHOST_CALL_GOTO(x_in->clone(x_in,&x,x_in->traits.nrows,0,x_in->traits.ncols,0),err,ret);
    } else {
        x = x_in;
    }
        

    if (v_in->traits.flags & GHOST_DENSEMAT_SCATTERED)
    {
        WARNING_LOG("The vector v is scattered. It will be cloned to a compressed "
                "vector before computation but not be changed itself.");
        ghost_densemat_t *vc;
        v_in->clone(v_in,&vc,v->traits.nrows,0,v->traits.ncols,0);
        v = vc;
    }
    if (w->traits.flags & GHOST_DENSEMAT_SCATTERED)
    {
        WARNING_LOG("The vector w is scattered. It will be cloned to a compressed "
                "vector before computation but not be changed itself.");
        ghost_densemat_t *wc;
        w_in->clone(w_in,&wc,w->traits.nrows,0,w->traits.ncols,0);
        w = wc;
    }
    
    if ((v->traits.location == GHOST_LOCATION_HOST) && !(flags & GHOST_GEMM_NOT_SPECIAL)) { 
        if (flags & GHOST_GEMM_KAHAN) {
            if (ghost_tsmttsm_kahan_valid(x,v,transv,w,transw,alpha,beta,reduce,0) == GHOST_SUCCESS) {
                INFO_LOG("Transparently call special implementation Kahan-TSMTTSM");
                GHOST_CALL_GOTO(ghost_tsmttsm_kahan(x,v,w,alpha,beta,reduce,transv[0] == 'C' || transv[0] == 'c'),err,ret);
                donespecial = 1;
            } else {
                WARNING_LOG("Will not do Kahan summation although requested!");
            }
        }

        if (ghost_tsmttsm_valid(x,v,transv,w,transw,alpha,beta,reduce,0) == GHOST_SUCCESS) {
            INFO_LOG("Transparently call special implementation TSMTTSM");
            GHOST_CALL_GOTO(ghost_tsmttsm(x,v,w,alpha,beta,reduce,transv[0] == 'C' || transv[0] == 'c'),err,ret);
            donespecial = 1;
        }
        if (ghost_tsmm_valid(x,v,transv,w,transw,alpha,beta,reduce,0) == GHOST_SUCCESS) {
            INFO_LOG("Transparently call special implementation TSMM");
            GHOST_CALL_GOTO(ghost_tsmm(x,v,w,alpha,beta),err,ret);
            donespecial = 1;
        }
        if (ghost_tsmm_inplace_valid(x,v,transv,w,transw,alpha,beta,reduce,0) == GHOST_SUCCESS) {
            INFO_LOG("Transparently call special implementation TSMM-inplace");
            GHOST_CALL_GOTO(ghost_tsmm_inplace(x,w,alpha,beta),err,ret);
            donespecial = 1;
        }
    }

    if (!donespecial) {

        if ((ret = ghost_gemm_valid(x,v,transv,w,transw,alpha,beta,reduce,flags,1)) != GHOST_SUCCESS) {
            return ret;
        }
        ret = ghost_gemm_blas(x,v,transv,w,transw,alpha,beta,reduce,flags);
    }
    
    if (x != x_in) {
        INFO_LOG("Transform x back");
        GHOST_CALL_GOTO(x_in->fromVec(x_in,x,0,0),err,ret);
        x->destroy(x);
    }
    
    goto out;
err:

out:
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH)
    return ret;
}

int ghost_gemm_perf_GFs(double *perf, double time, void *varg)
{
    ghost_gemm_perf_args_t arg = *(ghost_gemm_perf_args_t *)varg;
    
    *perf = (arg.vrows*arg.xcols*(1+arg.vcols*3))/1.e9/time;

    return 0;
}
