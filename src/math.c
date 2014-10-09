#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/math.h"
#include "ghost/densemat.h"
#include "ghost/locality.h"
#include "ghost/blas_mangle.h"
#include "ghost/instr.h"
#include "ghost/log.h"
#include "ghost/spmv_solvers.h"
#include <strings.h>
#include <math.h>
#include <complex.h>
#ifdef GHOST_HAVE_CUDA
#include <cublas_v2.h>
#endif

static ghost_mpi_op_t GHOST_MPI_OP_SUM_C = MPI_OP_NULL;
static ghost_mpi_op_t GHOST_MPI_OP_SUM_Z = MPI_OP_NULL;

static void ghost_spmv_selectMode(ghost_context_t * context, ghost_spmv_flags_t *flags);

ghost_error_t ghost_dot(void *res, ghost_densemat_t *vec, ghost_densemat_t *vec2)
{
    GHOST_FUNC_ENTRY(GHOST_FUNCTYPE_MATH)
    vec->dot(vec,res,vec2);
#ifdef GHOST_HAVE_MPI
    if (vec->context) {
        GHOST_INSTR_START("reduce")
        ghost_mpi_op_t sumOp;
        ghost_mpi_datatype_t mpiDt;
        ghost_mpi_op_sum(&sumOp,vec->traits.datatype);
        ghost_mpi_datatype(&mpiDt,vec->traits.datatype);
        int v;
        if (!(vec->traits.flags & GHOST_DENSEMAT_GLOBAL)) {
            for (v=0; v<MIN(vec->traits.ncols,vec2->traits.ncols); v++) {
                MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE, (char *)res+vec->elSize*v, 1, mpiDt, sumOp, vec->context->mpicomm));
            }
        }
        GHOST_INSTR_STOP("reduce")
    }
#endif

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH)
    return GHOST_SUCCESS;

}

ghost_error_t ghost_normalize(ghost_densemat_t *vec)
{
    ghost_lidx_t ncols = vec->traits.ncols;
    ghost_lidx_t c;

    GHOST_FUNC_ENTRY(GHOST_FUNCTYPE_MATH);
    if (vec->traits.datatype & GHOST_DT_FLOAT) {
        if (vec->traits.datatype & GHOST_DT_COMPLEX) {
            complex float res[ncols];
            GHOST_CALL_RETURN(ghost_dot(res,vec,vec));
            for (c=0; c<ncols; c++) {
                res[c] = 1.f/csqrtf(res[c]);
            }
            GHOST_CALL_RETURN(vec->vscale(vec,res));
        } else {
            float res[ncols];
            GHOST_CALL_RETURN(ghost_dot(res,vec,vec));
            for (c=0; c<ncols; c++) {
                res[c] = 1.f/sqrtf(res[c]);
            }
            GHOST_CALL_RETURN(vec->vscale(vec,res));
        }
    } else {
        if (vec->traits.datatype & GHOST_DT_COMPLEX) {
            complex double res[ncols];
            GHOST_CALL_RETURN(ghost_dot(res,vec,vec));
            for (c=0; c<ncols; c++) {
                res[c] = 1./csqrt(res[c]);
            }
            GHOST_CALL_RETURN(vec->vscale(vec,res));
        } else {
            double res[ncols];
            GHOST_CALL_RETURN(ghost_dot(res,vec,vec));
            for (c=0; c<ncols; c++) {
                res[c] = 1./sqrt(res[c]);
            }
            GHOST_CALL_RETURN(vec->vscale(vec,res));
        }
    }
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return GHOST_SUCCESS;
}

static ghost_error_t ghost_vspmv(ghost_densemat_t *res, ghost_sparsemat_t *mat, ghost_densemat_t *invec, ghost_spmv_flags_t *flags, va_list argp)
{
    va_list argp_backup;
    va_copy(argp_backup,argp);
    DEBUG_LOG(1,"Performing SpMV");
    ghost_spmvsolver_t solver = NULL;
    ghost_spmv_selectMode(mat->context,flags);
    if (*flags & GHOST_SPMV_MODE_VECTOR) {
        solver = &ghost_spmv_vectormode;
    } else if (*flags & GHOST_SPMV_MODE_OVERLAP) {
        solver = &ghost_spmv_goodfaith;
    } else if (*flags & GHOST_SPMV_MODE_TASK) {
        solver = &ghost_spmv_taskmode; 
    } else if (*flags & GHOST_SPMV_MODE_NOMPI) {
        solver = &ghost_spmv_nompi; 
    }

    if (!solver) {
        ERROR_LOG("The SpMV solver as specified in options cannot be found.");
        return GHOST_ERR_INVALID_ARG;
    }

    solver(res,mat,invec,*flags,argp);

#ifdef GHOST_HAVE_MPI

    if (!(*flags & GHOST_SPMV_NOT_REDUCE) && (*flags & GHOST_SPMV_DOT)) {
        GHOST_INSTR_START("dot_reduce");
        void *dot = NULL;
        if (*flags & GHOST_SPMV_SCALE) {
            dot = va_arg(argp_backup,void *);
        }
        if (*flags & GHOST_SPMV_AXPBY) {
            dot = va_arg(argp_backup,void *);
        }
        if ((*flags & GHOST_SPMV_SHIFT) || (*flags & GHOST_SPMV_VSHIFT)) {
            dot = va_arg(argp_backup,void *);
        }
        if (*flags & GHOST_SPMV_DOT) {
            dot = va_arg(argp_backup,void *);
        }
        ghost_mpi_op_t op;
        ghost_mpi_datatype_t dt;
        ghost_mpi_op_sum(&op,res->traits.datatype);
        ghost_mpi_datatype(&dt,res->traits.datatype);

        MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE, dot, 3*invec->traits.ncols, dt, op, mat->context->mpicomm));
        GHOST_INSTR_STOP("dot_reduce");
    }
#endif


    return GHOST_SUCCESS;


}
ghost_error_t ghost_spmv(ghost_densemat_t *res, ghost_sparsemat_t *mat, ghost_densemat_t *invec, ghost_spmv_flags_t *flags, ...) 
{
    GHOST_FUNC_ENTRY(GHOST_FUNCTYPE_MATH);

    ghost_error_t ret = GHOST_SUCCESS;

    if (*flags & GHOST_SPMV_DOT) {
        *flags |= GHOST_SPMV_DOT_YY|GHOST_SPMV_DOT_XY|GHOST_SPMV_DOT_XX;
    }
    va_list argp;
    va_start(argp, flags);
    ret = ghost_vspmv(res,mat,invec,flags,argp);
    va_end(argp);
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
}

ghost_error_t ghost_gemm(ghost_densemat_t *x, ghost_densemat_t *v_in,  char * transv_in, 
ghost_densemat_t *w_in, char *transw_in, void *alpha, void *beta, int reduce) 
{
#ifdef GHOST_HAVE_LONGIDX_LOCAL
#ifndef GHOST_HAVE_MKL
    WARNING_LOG("Will cast 64-bit indices to 32 bit for non-MKL GEMM with LONGIDX");
#endif
#endif
    GHOST_FUNC_ENTRY(GHOST_FUNCTYPE_MATH)

    
    ghost_tsmm_inplace_parameters_t tsmm_inplace_par = {.dt = x->traits.datatype, .blocksz = x->traits.ncols};
    ghost_tsmm_inplace_kernel_t tsmm_inplace_kernel = ghost_tsmm_inplace_kernel(tsmm_inplace_par,x,v_in,w_in,reduce);
    if (tsmm_inplace_kernel) {
        INFO_LOG("Doing in-place TSMM instead of GEMM!");
        return ghost_tsmm_inplace(x,w_in,alpha);
    }
    ghost_tsmm_parameters_t tsmm_par = {.dt = x->traits.datatype, .blocksz1 = x->traits.ncols, .blocksz2 = v_in->traits.ncols};
    ghost_tsmm_kernel_t tsmm_kernel = ghost_tsmm_kernel(tsmm_par,x,v_in,w_in,reduce);
    if (tsmm_kernel) {
        INFO_LOG("Doing TSMM instead of GEMM!");
        return ghost_tsmm(x,v_in,w_in,alpha,beta);
    }
    ghost_tsmttsm_parameters_t tsmttsm_par = {.dt = x->traits.datatype, .blocksz = x->traits.ncols};
    ghost_tsmttsm_kernel_t tsmttsm_kernel = ghost_tsmttsm_kernel(tsmttsm_par,x,v_in,w_in,reduce);
    if (tsmttsm_kernel) {
        INFO_LOG("Doing TSMTTSM instead of GEMM!");
        return ghost_tsmttsm(x,v_in,w_in,alpha,beta);
    }

    char transv[1], transw[1];
    
    transv[0]=transv_in[0];
    transw[0]=transw_in[0];
    
    ghost_densemat_t *v = v_in;
    ghost_densemat_t *w = w_in;

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
    // if the result should be transposed swap the transv and transw if possible.
    int needMemTransposeX=0; // compute X.' and transpose back afterwards
    int swapDimsX=0; /* formally compute X' because X has different storage layout, but no 
                        need to transpose back */
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
    if (x->traits.flags & GHOST_DENSEMAT_SCATTERED)
    {
        ERROR_LOG("The result vector x is scattered.");
        return GHOST_ERR_NOT_IMPLEMENTED;
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
    if (v->traits.flags & w->traits.flags & x->traits.flags & GHOST_DENSEMAT_DEVICE)
    {
#ifdef GHOST_HAVE_CUDA
        cublasHandle_t ghost_cublas_handle;
        GHOST_CALL_RETURN(ghost_cu_cublas_handle(&ghost_cublas_handle)); 
        cublasOperation_t cutransv = !strncasecmp(transv,"T",1)?CUBLAS_OP_T:!strncasecmp(transv,"C",1)?CUBLAS_OP_C:CUBLAS_OP_N;
        cublasOperation_t cutransw = !strncasecmp(transw,"T",1)?CUBLAS_OP_T:!strncasecmp(transw,"C",1)?CUBLAS_OP_C:CUBLAS_OP_N;

        if (v->traits.datatype & GHOST_DT_COMPLEX) 
        {
            if (v->traits.datatype & GHOST_DT_DOUBLE) 
            {
                CUBLAS_CALL_RETURN(cublasZgemm(ghost_cublas_handle,cutransv,cutransw,*m,*n,*k,(cuDoubleComplex *)alpha,(cuDoubleComplex *)v->cu_val,*ldv,(cuDoubleComplex *)w->cu_val,*ldw,(cuDoubleComplex *)mybeta,(cuDoubleComplex *)x->cu_val,*ldx));
            } 
            else 
            {
                CUBLAS_CALL_RETURN(cublasCgemm(ghost_cublas_handle,cutransv,cutransw,*m,*n,*k,(cuFloatComplex *)alpha,(cuFloatComplex *)v->cu_val,*ldv,(cuFloatComplex *)w->cu_val,*ldw,(cuFloatComplex *)mybeta,(cuFloatComplex *)x->cu_val,*ldx));
            }
        } 
        else 
        {
            if (v->traits.datatype & GHOST_DT_DOUBLE) 
            {
                CUBLAS_CALL_RETURN(cublasDgemm(ghost_cublas_handle,cutransv,cutransw,*m,*n,*k,(double *)alpha,(double *)v->cu_val,*ldv,(double *)w->cu_val,*ldw,(double *)mybeta,(double *)x->cu_val,*ldx));
            } 
            else 
            {
                CUBLAS_CALL_RETURN(cublasSgemm(ghost_cublas_handle,cutransv,cutransw,*m,*n,*k,(float *)alpha,(float *)v->cu_val,*ldv,(float *)w->cu_val,*ldw,(float *)mybeta,(float *)x->cu_val,*ldx));
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

#ifdef GHOST_HAVE_MPI 
    ghost_lidx_t i;
    if (reduce == GHOST_GEMM_NO_REDUCE) {
        return GHOST_SUCCESS;
    } 
    else 
    {

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

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH)
    return GHOST_SUCCESS;
}

#ifdef GHOST_HAVE_MPI

static void ghost_mpi_add_c(ghost_mpi_c *invec, ghost_mpi_c *inoutvec, int *len)
{
    int i;
    ghost_mpi_c c;

    for (i=0; i<*len; i++, invec++, inoutvec++){
        c.x = invec->x + inoutvec->x;
        c.y = invec->y + inoutvec->y;
        *inoutvec = c;
    }
}

static void ghost_mpi_add_z(ghost_mpi_z *invec, ghost_mpi_z *inoutvec, int *len)
{
    int i;
    ghost_mpi_z c;

    for (i=0; i<*len; i++, invec++, inoutvec++){
        c.x = invec->x + inoutvec->x;
        c.y = invec->y + inoutvec->y;
        *inoutvec = c;
    }
}

#endif

static void ghost_spmv_selectMode(ghost_context_t * context, ghost_spmv_flags_t *flags)
{
    if (!(*flags & GHOST_SPMV_MODES_ALL)) { // no mode specified
#ifdef GHOST_HAVE_MPI
        if (context->flags & GHOST_CONTEXT_REDUNDANT) {
            *flags |= GHOST_SPMV_MODE_NOMPI;
        } else {
            *flags |= GHOST_SPMV_MODE_OVERLAP;
        }
#else
        UNUSED(context);
        *flags |= GHOST_SPMV_MODE_NOMPI;
#endif
        DEBUG_LOG(1,"No spMVM mode has been specified, selecting a sensible default, namely %s",ghost_spmv_mode_string(*flags));
    } else {
#ifndef GHOST_HAVE_MPI
        if ((*flags & GHOST_SPMV_MODES_MPI)) {
            WARNING_LOG("Forcing non-MPI SpMV!");
            *flags &= ~(GHOST_SPMV_MODES_MPI);
            *flags |= GHOST_SPMV_MODE_NOMPI;
        }
#endif
    }
}

ghost_error_t ghost_mpi_op_sum(ghost_mpi_op_t * op, int datatype)
{
    if (!op) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }
#ifdef GHOST_HAVE_MPI
    if (datatype & GHOST_DT_FLOAT) {
        if (datatype & GHOST_DT_COMPLEX) {
            *op = GHOST_MPI_OP_SUM_C;
        } else {
            *op = MPI_SUM;
        }
    } else {
        if (datatype & GHOST_DT_COMPLEX) {
            *op = GHOST_MPI_OP_SUM_Z;
        } else {
            *op = MPI_SUM;
        }
    }
#else
    UNUSED(datatype);
    *op = MPI_OP_NULL;
#endif

    return GHOST_SUCCESS;

}

ghost_error_t ghost_mpi_operations_create()
{
#ifdef GHOST_HAVE_MPI
    MPI_CALL_RETURN(MPI_Op_create((MPI_User_function *)&ghost_mpi_add_c,1,&GHOST_MPI_OP_SUM_C));
    MPI_CALL_RETURN(MPI_Op_create((MPI_User_function *)&ghost_mpi_add_z,1,&GHOST_MPI_OP_SUM_Z));
#else
    UNUSED(GHOST_MPI_OP_SUM_C);
    UNUSED(GHOST_MPI_OP_SUM_Z);
#endif

    return GHOST_SUCCESS;
}

ghost_error_t ghost_mpi_operations_destroy()
{
#ifdef GHOST_HAVE_MPI
    MPI_CALL_RETURN(MPI_Op_free(&GHOST_MPI_OP_SUM_C));
    MPI_CALL_RETURN(MPI_Op_free(&GHOST_MPI_OP_SUM_Z));
#endif

    return GHOST_SUCCESS;
}

ghost_error_t ghost_spmv_nflops(int *nFlops, ghost_datatype_t m_t, ghost_datatype_t v_t)
{
    if (!ghost_datatype_valid(m_t) || !ghost_datatype_valid(v_t)) {
        ERROR_LOG("Invalid data type");
        return GHOST_ERR_INVALID_ARG;
    }
    if (!nFlops) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }

    *nFlops = 2;

    if (m_t & GHOST_DT_COMPLEX) {
        if (v_t & GHOST_DT_COMPLEX) {
            *nFlops = 8;
        }
    } else {
        if (v_t & GHOST_DT_COMPLEX) {
            *nFlops = 4;
        }
    }

    return GHOST_SUCCESS;
}

char * ghost_spmv_mode_string(ghost_spmv_flags_t flags) 
{
    if (flags & GHOST_SPMV_MODE_NOMPI) {
        return "Non-MPI";
    }
    if (flags & GHOST_SPMV_MODE_VECTOR) {
        return "Vector mode";
    }
    if (flags & GHOST_SPMV_MODE_OVERLAP) {
        return "Naïve overlap mode";
    }
    if (flags & GHOST_SPMV_MODE_TASK) {
        return "Task mode";
    }
    return "Invalid";

}

int ghost_spmv_perf(double *perf, double time, void *varg)
{
    ghost_spmv_perf_args_t *arg = (ghost_spmv_perf_args_t *)varg;

    ghost_lidx_t ncol = arg->rhs->traits.ncols;
    double flops;
    ghost_gidx_t nnz;
    ghost_gidx_t nrow;
    int spmvFlopsPerMatrixEntry = 0;
    int flopsPerAdd = 1;
    int flopsPerMul = 1;
    int flopsPerMulSame = 1;

    if (arg->rhs->traits.datatype & GHOST_DT_COMPLEX) {
        flopsPerAdd = 2;
        flopsPerMul = 6;
        flopsPerMulSame = 3;
    }


    ghost_sparsemat_nnz(&nnz,arg->mat);
    ghost_sparsemat_nrows(&nrow,arg->mat);
    ghost_spmv_nflops(&spmvFlopsPerMatrixEntry,arg->mat->traits->datatype,arg->rhs->traits.datatype);

    flops = (double)spmvFlopsPerMatrixEntry*(double)nnz*(double)ncol;

    if (arg->flags & GHOST_SPMV_AXPY) {
        flops += flopsPerAdd*nrow*ncol;
    }
    if (arg->flags & GHOST_SPMV_AXPBY) {
        flops += (flopsPerMul+flopsPerAdd)*nrow*ncol;
    }
    if (arg->flags & GHOST_SPMV_SHIFT) {
        flops += (flopsPerMul+flopsPerAdd)*nrow*ncol;
    }
    if (arg->flags & GHOST_SPMV_VSHIFT) {
        flops += (flopsPerMul+flopsPerAdd)*nrow*ncol;
    }
    if (arg->flags & GHOST_SPMV_SCALE) {
        flops += flopsPerMul*nrow*ncol;
    }
    if (arg->flags & GHOST_SPMV_DOT_YY) {
        flops += (flopsPerMulSame+1)*nrow*ncol;
    }
    if (arg->flags & GHOST_SPMV_DOT_XY) {
        flops += (flopsPerMul+flopsPerAdd)*nrow*ncol;
    }
    if (arg->flags & GHOST_SPMV_DOT_XX) {
        flops += (flopsPerMulSame+1)*nrow*ncol;
    }

    *perf = flops/1.e9/time;

    return 0;

}

int ghost_axpy_perf(double *perf, double time, void *varg)
{
    ghost_axpy_perf_args_t *arg = (ghost_axpy_perf_args_t *)varg;

    ghost_lidx_t ncol = arg->vec1->traits.ncols;
    ghost_lidx_t nrow = arg->vec1->traits.nrows;

    size_t size;
    ghost_datatype_size(&size,arg->vec1->traits.datatype);

    *perf = size*3*ncol*nrow/1.e9/time;

    return 0;
}

int ghost_dot_perf(double *perf, double time, void *varg)
{
    ghost_dot_perf_args_t *arg = (ghost_dot_perf_args_t *)varg;

    ghost_lidx_t ncol = arg->vec1->traits.ncols;
    ghost_lidx_t nrow = arg->vec1->traits.nrows;

    size_t size;
    ghost_datatype_size(&size,arg->vec1->traits.datatype);

    *perf = size*ncol*nrow/1.e9/time;

    if (arg->vec1 != arg->vec2) {
        *perf *= 2;
    }

    return 0;
}

int ghost_scale_perf(double *perf, double time, void *varg)
{
    ghost_scale_perf_args_t *arg = (ghost_scale_perf_args_t *)varg;

    ghost_lidx_t ncol = arg->vec->traits.ncols;
    ghost_lidx_t nrow = arg->vec->traits.nrows;

    size_t size;
    ghost_datatype_size(&size,arg->vec->traits.datatype);


    *perf = size*2*ncol*nrow/1.e9/time;

    return 0;
}
