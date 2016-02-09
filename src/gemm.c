#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/math.h"
#include "ghost/locality.h"
#include "ghost/blas_mangle.h"
#include "ghost/blas_util.h"
#include "ghost/tsmttsm.h"
#include "ghost/tsmm.h"

#include <strings.h>
#ifdef GHOST_HAVE_CUDA
#include <cublas_v2.h>
#endif

ghost_error ghost_gemm_valid(ghost_densemat *x, ghost_densemat *v, const char * transv, 
ghost_densemat *w, const char *transw, void *alpha, void *beta, int reduce,ghost_gemm_flags flags, int printerror) 
{
    if (v->traits.datatype != w->traits.datatype) {
        if (printerror) {
            ERROR_LOG("GEMM with mixed datatypes does not work!");
        }
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
    if (!((v->traits.location & w->traits.location) & x->traits.location)) { 
        ERROR_LOG("Invalid densemat locations: %s <- %s x %s",ghost_location_string(x->traits.location),ghost_location_string(v->traits.location),ghost_location_string(w->traits.location));
        return GHOST_ERR_INVALID_ARG;
    }
    

    ghost_lidx nrV,ncV,nrW,ncW,nrX,ncX;

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
    
    if ((x!=v) && (x!=w) && (ncV!=nrW || nrV!=nrX || ncW!=ncX)) {
        if (printerror) {
            ERROR_LOG("GEMM with incompatible vectors: %"PRLIDX"x%"PRLIDX" * %"PRLIDX"x%"PRLIDX" = %"PRLIDX"x%"PRLIDX,nrV,ncV,nrW,ncW,nrX,ncX);
        }
        return GHOST_ERR_INVALID_ARG;
    }

    UNUSED(alpha);
    UNUSED(beta);
    UNUSED(reduce);
    UNUSED(flags);
    return GHOST_SUCCESS;

}


static ghost_error ghost_gemm_blas(ghost_densemat *x_in, ghost_densemat *v_in, const char * transv_in, 
ghost_densemat *w_in, const char *transw_in, void *alpha, void *beta, int reduce,ghost_gemm_flags flags) 
{
    UNUSED(flags);

    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    ghost_error ret = GHOST_SUCCESS;
    char transv[1], transw[1];
    
    transv[0]=transv_in[0];
    transw[0]=transw_in[0];
    
    ghost_densemat *x = x_in;
    ghost_densemat *v = v_in;
    ghost_densemat *w = w_in;
    
    
    /*if ((v->traits.storage != w->traits.storage) || (x->traits.storage != w->traits.storage)){
        ERROR_LOG("Different storage layouts of input densemats!");
        return GHOST_ERR_INVALID_ARG;
    }*/

    if (v == x && !(flags & GHOST_GEMM_NOT_CLONE_ALIASED)) {
        WARNING_LOG("x equals v! v will be cloned.");
        ghost_densemat *vc;
        v->clone(v,&vc,v->traits.nrows,0,v->traits.ncols,0);
        v = vc;
    }

    if (w == x && !(flags & GHOST_GEMM_NOT_CLONE_ALIASED)) {
        WARNING_LOG("x equals w! w will be cloned.");
        ghost_densemat *wc;
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

    ghost_lidx nrV,ncV,ncW,nrVglob,ncVglob,ncWglob;

    if (strncasecmp(transv_in,"N",1)) {
        nrV = v->traits.ncols; 
        ncV = v->traits.nrows;
        if (v->context) {
            ncVglob = v->context->gnrows;
        } else {
            ncVglob = w->traits.nrows;
        }
        nrVglob = v->traits.ncols;
    } else {
        nrV = v->traits.nrows; 
        ncV = v->traits.ncols;
        if (v->context) {
            nrVglob = v->context->gnrows;
        } else {
            nrVglob = v->traits.nrows;
        }
        ncVglob = v->traits.ncols;
    }
    if (strncasecmp(transw_in,"N",1)) {
        ncW = w->traits.nrows;
        if (w->context) {
            ncWglob = w->context->gnrows;
        } else {
            ncWglob = w->traits.nrows;
        }
    } else {
        ncW = w->traits.ncols;
        ncWglob = w->traits.ncols;
    }

#ifdef GHOST_HAVE_INSTR_TIMING
    ghost_gemm_perf_args gemm_perfargs;
    gemm_perfargs.m = nrVglob;
    gemm_perfargs.k = ncVglob;
    gemm_perfargs.n = ncWglob;
    gemm_perfargs.dt = x->traits.datatype;
    gemm_perfargs.betaiszero = ghost_iszero(beta,v->traits.datatype);
    gemm_perfargs.alphaisone = ghost_isone(alpha,v->traits.datatype);
    ghost_timing_set_perfFunc(NULL,__ghost_functag,ghost_gemm_perf_GFs,(void *)&gemm_perfargs,sizeof(gemm_perfargs),"GF/s");
#else
    UNUSED(nrVglob);
    UNUSED(ncVglob);
    UNUSED(ncWglob);
#endif


    complex double zero = 0.+I*0.;

    ghost_blas_idx *m, *n, *k;
    m = (ghost_blas_idx *)&nrV;
    k = (ghost_blas_idx *)&ncV;
    n = (ghost_blas_idx *)&ncW;

    ghost_blas_idx *ldv = &v->stride;
    ghost_blas_idx *ldw = &w->stride;
    ghost_blas_idx *ldx = &x->stride;

    void *vdata = v->val;
    void *wdata = w->val;
    void *xdata = x->val;
    
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
    if (((v->traits.location & w->traits.location) & x->traits.location) & GHOST_LOCATION_DEVICE) {
#ifdef GHOST_HAVE_CUDA
        cublasHandle_t ghost_cublas_handle;
        ghost_blas_idx culdv,culdw,culdx;
        
        GHOST_CALL_GOTO(ghost_cu_cublas_handle(&ghost_cublas_handle),err,ret); 
        cublasOperation_t cutransv = !strncasecmp(transv,"T",1)?CUBLAS_OP_T:!strncasecmp(transv,"C",1)?CUBLAS_OP_C:CUBLAS_OP_N;
        cublasOperation_t cutransw = !strncasecmp(transw,"T",1)?CUBLAS_OP_T:!strncasecmp(transw,"C",1)?CUBLAS_OP_C:CUBLAS_OP_N;

        culdv = *ldv;
        culdw = *ldw;
        culdx = *ldx;

        void *xcuval = x->cu_val;
        void *vcuval = v->cu_val;
        void *wcuval = w->cu_val;


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
            
            ghost_blas_idx ldtmp = culdv;
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

            if (!strncasecmp(transv_in,"C",1) && v->traits.datatype & GHOST_DT_COMPLEX) {
                INFO_LOG("Conjugating input densemats. They have been cloned before, so this is safe.");
                v->conj(v);
                w->conj(w);
            }

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
                
                ghost_blas_idx ldtmp = culdv;
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
            (v->traits.location & GHOST_LOCATION_HOST)) {
        // BLAS cannot handle different storage layouts at once:
        if( x->traits.storage != v->traits.storage )
        {
          if( strncasecmp(transv,"N",1) == 0 )
            transv[0] = 'T';
          else if( strncasecmp(transv,"T",1) == 0 )
            transv[0] = 'N';
          else
          {
            // cannot do it
            ret = GHOST_ERR_NOT_IMPLEMENTED;
            goto err;
          }
        }
        if( x->traits.storage != w->traits.storage )
        {
          if( strncasecmp(transw,"N",1) == 0 )
            transw[0] = 'T';
          else if( strncasecmp(transw,"T",1) == 0 )
            transw[0] = 'N';
          else
          {
            // cannot do it
            ret = GHOST_ERR_NOT_IMPLEMENTED;
            goto err;
          }
        }
        if (v->traits.datatype & GHOST_DT_COMPLEX) 
        {
            if (v->traits.datatype & GHOST_DT_DOUBLE) 
            {
                BLAS_CALL_GOTO(zgemm(x->traits.storage,transv,transw, m,n, k, alpha, vdata, ldv, wdata, ldw, mybeta, xdata, ldx),err,ret);
            } 
            else 
            {
                BLAS_CALL_GOTO(cgemm(x->traits.storage,transv,transw, m,n, k, alpha, vdata, ldv, wdata, ldw, mybeta, xdata, ldx),err,ret);
            }
        } 
        else 
        {
            if (v->traits.datatype & GHOST_DT_DOUBLE) 
            {
                BLAS_CALL_GOTO(dgemm(x->traits.storage,transv,transw, m,n, k, (double *)alpha, vdata, ldv, wdata, ldw, (double *)mybeta, xdata, ldx),err,ret);
            } 
            else 
            {
                BLAS_CALL_GOTO(sgemm(x->traits.storage,transv,transw, m,n, k, (float *)alpha, vdata, ldv, wdata, ldw, (float *)mybeta, xdata, ldx),err,ret);
            }    
        }
    }

    if ((reduce != GHOST_GEMM_NO_REDUCE) && (v->context)) {
        x->reduce(x,v->context->mpicomm,reduce);
    }
    if (w != w_in) {
        INFO_LOG("Destroy clone of w");
        ghost_densemat_destroy(w);
    }
    if (v != v_in) {
        INFO_LOG("Destroy clone of v");
        ghost_densemat_destroy(v);
    }
    
    goto out;
err:

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL)
    return ret;

}

ghost_error ghost_gemm(ghost_densemat *x_in, ghost_densemat *v_in, const char * transv, 
ghost_densemat *w_in, const char *transw, void *alpha, void *beta, int reduce,ghost_gemm_flags flags) 
{
#ifdef GHOST_IDX64_LOCAL
#ifndef GHOST_HAVE_MKL
    WARNING_LOG("Will cast 64-bit indices to 32 bit for non-MKL GEMM with 64 bit indices");
#endif
#endif
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error ret = GHOST_SUCCESS;
    int donespecial = 0;
      
    ghost_densemat *x = NULL;
    ghost_densemat *v = v_in;
    ghost_densemat *w = w_in;
    
    // scattered vectors are copied together, if this occurs the user should rethink his or 
    // her data layout.
    if (x_in->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        INFO_LOG("The result vector x is scattered. It will be cloned and compressed before the computation and transformed back afterwards.");
        GHOST_CALL_GOTO(x_in->clone(x_in,&x,x_in->traits.nrows,0,x_in->traits.ncols,0),err,ret);
    } else {
        x = x_in;
    }
        
#ifdef GHOST_HAVE_CUDA
    if (((v->traits.location & w->traits.location) & x->traits.location) & GHOST_LOCATION_DEVICE) {
        if (v_in->traits.storage == GHOST_DENSEMAT_ROWMAJOR && w_in->traits.storage == GHOST_DENSEMAT_ROWMAJOR &&
                !strncasecmp(transv,"C",1) && !strncasecmp(transw,"N",1) && v->traits.datatype & GHOST_DT_COMPLEX) {
            WARNING_LOG("Cloning both input vectors because they need to be conjugated!");
            ghost_densemat *vc;
            v_in->clone(v_in,&vc,v->traits.nrows,0,v->traits.ncols,0);
            v = vc;
            ghost_densemat *wc;
            w_in->clone(w_in,&wc,w->traits.nrows,0,w->traits.ncols,0);
            w = wc;
        }
    }
#endif
    if (v->traits.flags & GHOST_DENSEMAT_SCATTERED)
    {
        WARNING_LOG("The vector v is scattered. It will be cloned to a compressed "
                "vector before computation but not be changed itself.");
        ghost_densemat *vc;
        v->clone(v,&vc,v->traits.nrows,0,v->traits.ncols,0);
        v = vc;
    }
    if (w->traits.flags & GHOST_DENSEMAT_SCATTERED)
    {
        WARNING_LOG("The vector w is scattered. It will be cloned to a compressed "
                "vector before computation but not be changed itself.");
        ghost_densemat *wc;
        w->clone(w,&wc,w->traits.nrows,0,w->traits.ncols,0);
        w = wc;
    }
    
    if (!(flags & GHOST_GEMM_NOT_SPECIAL)) { 
        /*if (flags & GHOST_GEMM_KAHAN) {
            if (ghost_tsmttsm_kahan_valid(x,v,transv,w,transw,alpha,beta,reduce,0) == GHOST_SUCCESS) {
                INFO_LOG("Transparently call special implementation Kahan-TSMTTSM");

                ret = ghost_tsmttsm_kahan(x,v,w,alpha,beta,reduce,transv[0] == 'C' || transv[0] == 'c');
                if( ret == GHOST_SUCCESS )
                  donespecial = 1;
                else if( ret != GHOST_ERR_INVALID_ARG )
                  goto err;
            }
            if( !donespecial ) {
                WARNING_LOG("Will not do Kahan summation although requested!");
            }
        }*/

        if (ghost_tsmttsm_valid(x,v,transv,w,transw,alpha,beta,reduce,flags,0) == GHOST_SUCCESS) {
            INFO_LOG("Transparently call special implementation TSMTTSM");
            ret = ghost_tsmttsm(x,v,w,alpha,beta,reduce,transv[0] == 'C' || transv[0] == 'c',flags);
            if( ret == GHOST_SUCCESS )
              donespecial = 1;
            else if( ret != GHOST_ERR_INVALID_ARG )
              goto err;
        }
        if (ghost_tsmm_valid(x,v,transv,w,transw,alpha,beta,reduce,0) == GHOST_SUCCESS) {
            INFO_LOG("Transparently call special implementation TSMM");
            ret = ghost_tsmm(x,v,w,alpha,beta);
            if( ret == GHOST_SUCCESS )
              donespecial = 1;
            else if( ret != GHOST_ERR_INVALID_ARG )
              goto err;
        }
        if (ghost_tsmm_inplace_valid(x,v,transv,w,transw,alpha,beta,reduce,0) == GHOST_SUCCESS) {
            INFO_LOG("Transparently call special implementation TSMM-inplace");
            ret = ghost_tsmm_inplace(x,w,alpha,beta);
            if( ret == GHOST_SUCCESS )
              donespecial = 1;
            else if( ret != GHOST_ERR_INVALID_ARG )
              goto err;
        }
    }

    if (!donespecial) {
        if ((ret = ghost_gemm_valid(x,v,transv,w,transw,alpha,beta,reduce,flags,1)) == GHOST_SUCCESS) {
            ret = ghost_gemm_blas(x,v,transv,w,transw,alpha,beta,reduce,flags);
        }
    }
    
    if (x != x_in) {
        INFO_LOG("Transform x back");
        GHOST_CALL_GOTO(x_in->fromVec(x_in,x,0,0),err,ret);
        ghost_densemat_destroy(x);
    }
    
    goto out;
err:

out:
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH)
    return ret;
}

int ghost_gemm_perf_GFs(double *perf, double time, void *varg)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    ghost_gemm_perf_args arg = *(ghost_gemm_perf_args *)varg;
    int maddflops = 2;
    int mulflops = 1;
    double totalflops = 0;

    if (arg.dt & GHOST_DT_COMPLEX) {
        maddflops = 8;
        mulflops = 6;
    }
    totalflops = (double)maddflops*arg.n*arg.m*arg.k;
  
    WARNING_LOG("The performance computation has to be checked for correctness!"); 
    if (!arg.alphaisone) {
        totalflops += mulflops*arg.m*arg.n; 
    }
    if (!arg.betaiszero) {
        totalflops += maddflops*arg.m*arg.n; 
    }

    *perf = totalflops/1.e9/time;

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return 0;
}

int ghost_gemm_perf_GBs(double *perf, double time, void *varg)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    size_t size;
    ghost_gemm_perf_args arg = *(ghost_gemm_perf_args *)varg;
    
    ghost_datatype_size(&size,arg.dt);

    if (arg.betaiszero) {
        *perf = size*(arg.m*arg.n+arg.m*arg.k+arg.n*arg.k)/1.e9/time;
    } else {
        *perf = size*(2*arg.m*arg.n+arg.m*arg.k+arg.n*arg.k)/1.e9/time;
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return 0;
}
