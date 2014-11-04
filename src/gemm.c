#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/math.h"
#include "ghost/locality.h"
#include "ghost/blas_mangle.h"

#include <strings.h>
#ifdef GHOST_HAVE_CUDA
#include <cublas_v2.h>
#endif


ghost_error_t ghost_gemm(ghost_densemat_t *x_in, ghost_densemat_t *v_in,  char * transv_in, 
ghost_densemat_t *w_in, char *transw_in, void *alpha, void *beta, int reduce) 
{
#ifdef GHOST_HAVE_LONGIDX_LOCAL
#ifndef GHOST_HAVE_MKL
    WARNING_LOG("Will cast 64-bit indices to 32 bit for non-MKL GEMM with LONGIDX");
#endif
#endif
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH)
       
    int deviceflags = (v_in->traits.flags & GHOST_DENSEMAT_DEVICE) + (w_in->traits.flags & GHOST_DENSEMAT_DEVICE) + (x_in->traits.flags & GHOST_DENSEMAT_DEVICE);

    if (deviceflags != 0 && deviceflags != (3*GHOST_DENSEMAT_DEVICE)) {
        ERROR_LOG("The storage of all densemats has to be uniform (host or device)!");
        return GHOST_ERR_INVALID_ARG;
    }

   
    if (!v_in->traits.flags & GHOST_DENSEMAT_DEVICE) { 
        ghost_tsmm_inplace_parameters_t tsmm_inplace_par = {.dt = x_in->traits.datatype, .blocksz = x_in->traits.ncols};
        ghost_tsmm_inplace_kernel_t tsmm_inplace_kernel = ghost_tsmm_inplace_kernel(tsmm_inplace_par,x_in,v_in,w_in,reduce);
        if (tsmm_inplace_kernel) {
            INFO_LOG("Doing in-place TSMM instead of GEMM!");
            return ghost_tsmm_inplace(x_in,w_in,alpha);
        }
        ghost_tsmm_parameters_t tsmm_par = {.dt = x_in->traits.datatype, .blocksz1 = x_in->traits.ncols, .blocksz2 = v_in->traits.ncols};
        ghost_tsmm_kernel_t tsmm_kernel = ghost_tsmm_kernel(tsmm_par,x_in,v_in,w_in,reduce);
        if (tsmm_kernel) {
            INFO_LOG("Doing TSMM instead of GEMM!");
            return ghost_tsmm(x_in,v_in,w_in,alpha,beta);
        }
        ghost_tsmttsm_parameters_t tsmttsm_par = {.dt = x_in->traits.datatype, .blocksz = x_in->traits.ncols};
        ghost_tsmttsm_kernel_t tsmttsm_kernel = ghost_tsmttsm_kernel(tsmttsm_par,x_in,v_in,w_in,reduce);
        if (tsmttsm_kernel) {
            INFO_LOG("Doing TSMTTSM instead of GEMM!");
            return ghost_tsmttsm(x_in,v_in,w_in,alpha,beta);
        }
    }

    char transv[1], transw[1];
    
    transv[0]=transv_in[0];
    transw[0]=transw_in[0];
    
    ghost_densemat_t *x;
    ghost_densemat_t *v = v_in;
    ghost_densemat_t *w = w_in;
    
    if (x_in->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        INFO_LOG("The result vector x is scattered. It will be cloned and compressed before the computation and transformed back afterwards.");
        GHOST_CALL_RETURN(x_in->clone(x_in,&x,x_in->traits.nrows,0,x_in->traits.ncols,0));
    } else {
        x = x_in;
    }
        
    // if the result should be transposed swap the transv and transw if possible.
    int needMemTransposeX=0; // compute X.' and transpose back afterwards
    int swapDimsX=0; /* formally compute X' because X has different storage layout, but no 
                            need to transpose back */

    if (!(x->traits.flags & GHOST_DENSEMAT_DEVICE)) {
        // we support the special cases V*W and V'*W with V row-major and W col-major or vice 
        // versa. If the result X has different storage layout than V, we use the 
        // memtranspose function afterwards if X is complex.
        if (v->traits.storage != w->traits.storage)
        {
            DEBUG_LOG(1,"gemm with different storage layout for V and W...");
            if (strncasecmp(transw,"N",1)) 
            {
                ERROR_LOG("GEMM with different storage layouts for V and W only implemented " 
                          "for transw='N'!");
                return GHOST_ERR_NOT_IMPLEMENTED;
            }
            // w has transposed mem-layout and "no transpose" is requested for w, cheat
            // cblas_xgemm into doing the right thing:
            transw[0]='T';
        }
        if (x->traits.storage != v->traits.storage)
        {
            DEBUG_LOG(1,"gemm with different storage layout for V and X...");
            if (strncasecmp(transv,"C",1) && strncasecmp(transv,"T",1))
            {
                // compute x=v'w and transpose afterwards
                DEBUG_LOG(1,"case a: post-memtranspose of X needed.");
                needMemTransposeX=1;
            }
            else
            {
                    DEBUG_LOG(1,"case b: fool gemm to compute transp(X)=W'V instead.");
                    // compute x' = w'*v instead of v'*w
                    v=w_in;
                    w=v_in;
                    swapDimsX=1;
            }
        }
    }

    // scattered vectors are copied together, if this occurs the user should rethink his or 
    // her data layout.
    if (v->traits.flags & GHOST_DENSEMAT_SCATTERED)
    {
        WARNING_LOG("The vector v is scattered. It will be cloned to a compressed "
                "vector before computation but not be changed itself.");
        ghost_densemat_t *vc;
        v->clone(v,&vc,w->traits.nrows,0,v->traits.ncols,0);
        v = vc;
    }
    if (w->traits.flags & GHOST_DENSEMAT_SCATTERED)
    {
        WARNING_LOG("The vector w is scattered. It will be cloned to a compressed "
                "vector before computation but not be changed itself.");
        ghost_densemat_t *wc;
        w->clone(w,&wc,w->traits.nrows,0,w->traits.ncols,0);
        w = wc;
    }
    
    if (v->traits.datatype != w->traits.datatype) {
        ERROR_LOG("GEMM with mixed datatypes does not work!");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }

    if (v->context == NULL && w->context == NULL && x->context == NULL && reduce != GHOST_GEMM_NO_REDUCE) {
        INFO_LOG("Reduction should be done but none of the vectors has a context. Ommitting the reduction...");
        reduce = GHOST_GEMM_NO_REDUCE;
    }

    int nranks;
    GHOST_CALL_RETURN(ghost_nrank(&nranks, v->context->mpicomm));

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

    if (swapDimsX)
    {
        nrX=x->traits.ncols;
        ncX=x->traits.nrows;
    }
    else
    {
        nrX=x->traits.nrows;
        ncX=x->traits.ncols;
    }
    if (ncV!=nrW || nrV!=nrX || ncW!=ncX) {
        ERROR_LOG("GEMM with incompatible vectors: %"PRLIDX"x%"PRLIDX" * %"PRLIDX"x%"PRLIDX" = %"PRLIDX"x%"PRLIDX,nrV,ncV,nrW,ncW,nrX,ncX);
       // return GHOST_ERR_INVALID_ARG;
    }
    if (v->traits.datatype != w->traits.datatype) {
        ERROR_LOG("GEMM with vectors of different datatype does not work");
        return GHOST_ERR_INVALID_ARG;
    }


    complex double zero = 0.+I*0.;

    ghost_blas_idx_t *m, *n, *k;
    m = (ghost_blas_idx_t *)&nrV;
    k = (ghost_blas_idx_t *)&ncV;
    n = (ghost_blas_idx_t *)&ncW;

    ghost_blas_idx_t *ldv = (ghost_blas_idx_t *)v->stride;
    ghost_blas_idx_t *ldw = (ghost_blas_idx_t *)w->stride;
    ghost_blas_idx_t *ldx = (ghost_blas_idx_t *)x->stride;

    void *vdata = NULL;
    void *wdata = NULL;
    void *xdata = NULL;
    GHOST_CALL_RETURN(ghost_densemat_valptr(v,&vdata));
    GHOST_CALL_RETURN(ghost_densemat_valptr(w,&wdata));
    GHOST_CALL_RETURN(ghost_densemat_valptr(x,&xdata));
    
    //note: if no reduction is requested, none of the input vecs may have
    // a context (or an MPI comm). If any reduction is requested, only v
    // needs context and comm, the others may be replicated. So we must
    // take the comm from v if and only if a reduction is requested.
    int myrank=0;

    if (reduce!=GHOST_GEMM_NO_REDUCE) {
        GHOST_CALL_RETURN(ghost_rank(&myrank,v->context->mpicomm));
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
    if (v->traits.flags & w->traits.flags & x->traits.flags & GHOST_DENSEMAT_DEVICE)
    {
#ifdef GHOST_HAVE_CUDA
        cublasHandle_t ghost_cublas_handle;
        ghost_blas_idx_t culdv,culdw,culdx;
        
        GHOST_CALL_RETURN(ghost_cu_cublas_handle(&ghost_cublas_handle)); 
        cublasOperation_t cutransv = !strncasecmp(transv,"T",1)?CUBLAS_OP_T:!strncasecmp(transv,"C",1)?CUBLAS_OP_C:CUBLAS_OP_N;
        cublasOperation_t cutransw = !strncasecmp(transw,"T",1)?CUBLAS_OP_T:!strncasecmp(transw,"C",1)?CUBLAS_OP_C:CUBLAS_OP_N;

        culdv = *ldv;
        culdw = *ldw;
        culdx = *ldx;
        if (v->traits.storage == GHOST_DENSEMAT_ROWMAJOR && w->traits.storage == GHOST_DENSEMAT_ROWMAJOR && (cutransv == CUBLAS_OP_T || cutransv == CUBLAS_OP_C) && cutransw == CUBLAS_OP_N) {
            INFO_LOG("special case 1");
            cutransw = cutransv;
            cutransv = CUBLAS_OP_N;
        }
        if ((v->traits.storage != w->traits.storage) && cutransv == CUBLAS_OP_N && cutransw == CUBLAS_OP_N) {
            INFO_LOG("special case 2");
            //needMemTransposeX = 1;
            cutransv = CUBLAS_OP_T;
            culdx = x->traits.nrows;
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



        void *xcuval;
        void *vcuval;
        void *wcuval;
        GHOST_CALL_RETURN(ghost_densemat_cu_valptr(x,&xcuval));
        GHOST_CALL_RETURN(ghost_densemat_cu_valptr(v,&vcuval));
        GHOST_CALL_RETURN(ghost_densemat_cu_valptr(w,&wcuval));

        if (v->traits.datatype & GHOST_DT_COMPLEX) 
        {
            if (v->traits.datatype & GHOST_DT_DOUBLE) 
            {
                CUBLAS_CALL_RETURN(cublasZgemm(ghost_cublas_handle,cutransv,cutransw,*m,*n,*k,(cuDoubleComplex *)alpha,(cuDoubleComplex *)vcuval,culdv,(cuDoubleComplex *)wcuval,culdw,(cuDoubleComplex *)mybeta,(cuDoubleComplex *)xcuval,culdx));
            } 
            else 
            {
                CUBLAS_CALL_RETURN(cublasCgemm(ghost_cublas_handle,cutransv,cutransw,*m,*n,*k,(cuFloatComplex *)alpha,(cuFloatComplex *)vcuval,culdv,(cuFloatComplex *)wcuval,culdw,(cuFloatComplex *)mybeta,(cuFloatComplex *)xcuval,culdx));
            }
        } 
        else 
        {
            if (v->traits.datatype & GHOST_DT_DOUBLE) 
            {
                CUBLAS_CALL_RETURN(cublasDgemm(ghost_cublas_handle,cutransv,cutransw,*m,*n,*k,(double *)alpha,(double *)vcuval,culdv,(double *)wcuval,culdw,(double *)mybeta,(double *)xcuval,culdx));
            } 
            else 
            {
                CUBLAS_CALL_RETURN(cublasSgemm(ghost_cublas_handle,cutransv,cutransw,*m,*n,*k,(float *)alpha,(float *)vcuval,culdv,(float *)wcuval,culdw,(float *)mybeta,(float *)xcuval,culdx));
            }    
        }
#endif
    } else
    if (v->traits.flags & w->traits.flags & x->traits.flags & GHOST_DENSEMAT_HOST)
    {
        if (v->traits.datatype & GHOST_DT_COMPLEX) 
        {
            if (v->traits.datatype & GHOST_DT_DOUBLE) 
            {
                zgemm(v->traits.storage,transv,transw, m,n, k, alpha, vdata, ldv, wdata, ldw, mybeta, xdata, ldx);
            } 
            else 
            {
                cgemm(v->traits.storage,transv,transw, m,n, k, alpha, vdata, ldv, wdata, ldw, mybeta, xdata, ldx);
            }
        } 
        else 
        {
            if (v->traits.datatype & GHOST_DT_DOUBLE) 
            {
                dgemm(v->traits.storage,transv,transw, m,n, k, (double *)alpha, vdata, ldv, wdata, ldw, (double *)mybeta, xdata, ldx);
            } 
            else 
            {
                sgemm(v->traits.storage,transv,transw, m,n, k, (double *)alpha, vdata, ldv, wdata, ldw, (double *)mybeta, xdata, ldx);
            }    
        }
    }
        /*x->string(x,&xvstr);

        printf("\n\n%s\n\n",xvstr);*/

#ifdef GHOST_HAVE_MPI 
    ghost_lidx_t i;
    if (reduce != GHOST_GEMM_NO_REDUCE) {

#ifdef GHOST_HAVE_CUDA
        ghost_lidx_t lda = *x->stride;
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


        for (i=0; i<dima; ++i) 
        {
            int copied = 0;
            void *val = NULL;
            if (x->traits.flags & GHOST_DENSEMAT_DEVICE)
            {
#ifdef GHOST_HAVE_CUDA
                size_t sizeofdt;
                ghost_datatype_size(&sizeofdt,x->traits.datatype);

                GHOST_CALL_RETURN(ghost_malloc((void **)&val,dimb*sizeofdt));
                ghost_cu_download(val, &x->cu_val[i*lda*sizeofdt], dimb*sizeofdt);
                copied = 1;
#endif
            }
            else if (x->traits.flags & GHOST_DENSEMAT_HOST)
            {
                val = x->val[i];
            }
            ghost_mpi_op_t sumOp;
            ghost_mpi_datatype_t mpiDt;
            GHOST_CALL_RETURN(ghost_mpi_op_sum(&sumOp,x->traits.datatype));
            GHOST_CALL_RETURN(ghost_mpi_datatype(&mpiDt,x->traits.datatype));

            if (reduce == GHOST_GEMM_ALL_REDUCE) 
            {
                MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE,val,dimb,mpiDt,sumOp,v->context->mpicomm));
            } 
            else 
            {
                if (myrank == reduce) 
                {
                    MPI_CALL_RETURN(MPI_Reduce(MPI_IN_PLACE,val,dimb,mpiDt,sumOp,reduce,v->context->mpicomm));
                } 
                else 
                {
                    MPI_CALL_RETURN(MPI_Reduce(val,NULL,dimb,mpiDt,sumOp,reduce,v->context->mpicomm));
                }
            }
            if (copied)
            {
#ifdef GHOST_HAVE_CUDA
                size_t sizeofdt;
                ghost_datatype_size(&sizeofdt,x->traits.datatype);
                GHOST_CALL_RETURN(ghost_cu_upload(&x->cu_val[i*lda*sizeofdt],val,dimb*sizeofdt));
                free(val);
#endif
            }
        }
    }
#else
    UNUSED(reduce);
#endif

    // in the case v^w for complex vectors we can't handle different storage layout of X
    // in xgemm directly so we need to explicitly transpose the result in memory
    if (needMemTransposeX!=0)
    {
        WARNING_LOG("gemm-result explicitly memtransposed, which presently means memory is"
        " reallocated!");
        GHOST_CALL_RETURN(x->memtranspose(x));
    }
    
    if (x != x_in) {
        INFO_LOG("Transform x back");
        GHOST_CALL_RETURN(x_in->fromVec(x_in,x,0,0));
        x->destroy(x);
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH)
    return GHOST_SUCCESS;
}

