#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/crs.h"
#include "ghost/log.h"
#include "ghost/cu_util.h"
#include "ghost/cu_complex.h"
#include "ghost/util.h"

#include <complex.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

typedef cusparseStatus_t (*cusparse_crs_spmv_kernel_t) (cusparseHandle_t handle, cusparseOperation_t transA, 
        int m, int n, int nnz, const void           *alpha, 
        const cusparseMatDescr_t descrA, 
        const void           *csrValA, 
        const void *csrRowPtrA, const void *csrColIndA,
        const void           *x, const void           *beta, 
        void           *y);

typedef cusparseStatus_t (*cusparse_crs_spmmv_cm_kernel_t) (cusparseHandle_t handle, cusparseOperation_t transA, 
        int m, int n, int k, int nnz, const void           *alpha, 
        const cusparseMatDescr_t descrA, 
        const void           *csrValA, 
        const void *csrRowPtrA, const void *csrColIndA,
        const void           *x, int ldx, const void           *beta, 
        void           *y, int ldy);

typedef cusparseStatus_t (*cusparse_crs_spmmv_rm_kernel_t) (cusparseHandle_t handle, cusparseOperation_t transA,
        cusparseOperation_t transB,
        int m, int n, int k, int nnz, const void           *alpha, 
        const cusparseMatDescr_t descrA, 
        const void           *csrValA, 
        const void *csrRowPtrA, const void *csrColIndA,
        const void           *x, int ldx, const void           *beta, 
        void           *y, int ldy);

    template<typename dt1, typename dt2>
static ghost_error_t ghost_cu_crsspmv_tmpl(ghost_sparsemat_t *mat, ghost_densemat_t * lhs, ghost_densemat_t *origlhs, ghost_densemat_t * rhs, ghost_spmv_flags_t options, va_list argp, cusparse_crs_spmv_kernel_t crskernel)
{
    cusparseHandle_t cusparse_handle;
    cusparseMatDescr_t descr;

    cusparseCreateMatDescr(&descr);
    GHOST_CALL_RETURN(ghost_cu_cusparse_handle(&cusparse_handle));

    dt2 *localdot = NULL;
    dt1 *shift = NULL, scale, beta, sdelta, seta;
    ghost_densemat_t *z = NULL;

    one<dt1>(scale);

    GHOST_SPMV_PARSE_ARGS(options,argp,scale,beta,shift,localdot,z,sdelta,seta,dt2,dt1);

    if (options & GHOST_SPMV_AXPY) {
        one<dt1>(beta);
    } else if (!(options & GHOST_SPMV_AXPBY)) {
        zero<dt1>(beta);
    }
    
    CUSPARSE_CALL_RETURN(crskernel(cusparse_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,mat->nrows,mat->ncols,mat->nnz,&scale,descr,(dt1 *)CR(mat)->cumat->val, CR(mat)->cumat->rpt, CR(mat)->cumat->col, (dt1 *)rhs->cu_val, &beta, (dt1 *)lhs->cu_val));
    
    if (options & (GHOST_SPMV_SHIFT|GHOST_SPMV_VSHIFT)) {
        PERFWARNING_LOG("Shift will not be applied on-the-fly!");
        dt2 minusshift[rhs->traits.ncols];
        ghost_lidx_t col;
        if (options & GHOST_SPMV_SHIFT) {
            for (col=0; col<rhs->traits.ncols; col++) {
                minusshift[col] = -1.*(*(dt2 *)&scale)*(*(dt2 *)shift);
            }
        } else {
            for (col=0; col<rhs->traits.ncols; col++) {
                minusshift[col] = -1.*(*(dt2 *)&scale)*(((dt2 *)shift)[col]);
            }
        }
        lhs->vaxpy(lhs,rhs,minusshift);
    }

    if (options & GHOST_SPMV_DOT_ANY) {
        PERFWARNING_LOG("Dot product computation will be not be done on-the-fly!");
        memset(localdot,0,lhs->traits.ncols*3*sizeof(dt1));
        if (options & GHOST_SPMV_DOT_YY) {
            lhs->dot(lhs,&localdot[0],lhs);
        }
        if (options & GHOST_SPMV_DOT_XY) {
            lhs->dot(lhs,&localdot[lhs->traits.ncols],rhs);
        }
        if (options & GHOST_SPMV_DOT_XX) {
            rhs->dot(rhs,&localdot[2*lhs->traits.ncols],rhs);
        }
            
    }
    if (options & GHOST_SPMV_CHAIN_AXPBY) {
        PERFWARNING_LOG("AXPBY will not be done on-the-fly!");
        z->axpby(z,lhs,&seta,&sdelta);
    }
    
    if (origlhs != lhs) {
        PERFWARNING_LOG("Scatter the result back");
        GHOST_CALL_RETURN(origlhs->fromVec(origlhs,lhs,0,0));
    }
    
    return GHOST_SUCCESS;
}
    
    template<typename dt1, typename dt2>
static ghost_error_t ghost_cu_crsspmmv_cm_tmpl(ghost_sparsemat_t *mat, ghost_densemat_t * lhs, ghost_densemat_t *origlhs, ghost_densemat_t * rhs, ghost_spmv_flags_t options, va_list argp, cusparse_crs_spmmv_cm_kernel_t crskernel)
{
    cusparseHandle_t cusparse_handle;
    cusparseMatDescr_t descr;

    cusparseCreateMatDescr(&descr);
    GHOST_CALL_RETURN(ghost_cu_cusparse_handle(&cusparse_handle));

    dt2 *localdot = NULL;
    dt1 *shift = NULL, scale, beta, sdelta, seta;
    ghost_densemat_t *z = NULL;

    one<dt1>(scale);

    GHOST_SPMV_PARSE_ARGS(options,argp,scale,beta,shift,localdot,z,sdelta,seta,dt2,dt1);

    if (options & GHOST_SPMV_AXPY) {
        one<dt1>(beta);
    } else {
        zero<dt1>(beta);
    }
    
    crskernel(cusparse_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,mat->nrows,rhs->traits.ncols,mat->ncols,mat->nnz,&scale,descr,(dt1 *)CR(mat)->cumat->val, CR(mat)->cumat->rpt, CR(mat)->cumat->col, (dt1 *)rhs->cu_val, rhs->stride, &beta, (dt1 *)lhs->cu_val, lhs->stride);
    
    if (options & (GHOST_SPMV_SHIFT|GHOST_SPMV_VSHIFT)) {
        PERFWARNING_LOG("Shift will not be applied on-the-fly!");
        dt2 minusshift[rhs->traits.ncols];
        ghost_lidx_t col;
        if (options & GHOST_SPMV_SHIFT) {
            for (col=0; col<rhs->traits.ncols; col++) {
                minusshift[col] = -1.*(*(dt2 *)&scale)*(*(dt2 *)shift);
            }
        } else {
            for (col=0; col<rhs->traits.ncols; col++) {
                minusshift[col] = -1.*(*(dt2 *)&scale)*(((dt2 *)shift)[col]);
            }
        }
        lhs->vaxpy(lhs,rhs,minusshift);
    }

    if (options & GHOST_SPMV_DOT_ANY) {
        PERFWARNING_LOG("Dot product computation will be not be done on-the-fly!");
        memset(localdot,0,lhs->traits.ncols*3*sizeof(dt1));
        if (options & GHOST_SPMV_DOT_YY) {
            lhs->dot(lhs,&localdot[0],lhs);
        }
        if (options & GHOST_SPMV_DOT_XY) {
            lhs->dot(lhs,&localdot[lhs->traits.ncols],rhs);
        }
        if (options & GHOST_SPMV_DOT_XX) {
            rhs->dot(rhs,&localdot[2*lhs->traits.ncols],rhs);
        }
            
    }
    if (options & GHOST_SPMV_CHAIN_AXPBY) {
        PERFWARNING_LOG("AXPBY will not be done on-the-fly!");
        z->axpby(z,lhs,&seta,&sdelta);
    }
    
    
    if (origlhs != lhs) {
        PERFWARNING_LOG("Scatter the result back");
        GHOST_CALL_RETURN(origlhs->fromVec(origlhs,lhs,0,0));
    }

    return GHOST_SUCCESS;
}

    template<typename dt1, typename dt2>
static ghost_error_t ghost_cu_crsspmmv_rm_tmpl(ghost_sparsemat_t *mat, ghost_densemat_t * lhs, ghost_densemat_t *origlhs, ghost_densemat_t * rhs, ghost_spmv_flags_t options, va_list argp, cusparse_crs_spmmv_rm_kernel_t crskernel)
{
    cusparseHandle_t cusparse_handle;
    cusparseMatDescr_t descr;

    cusparseCreateMatDescr(&descr);
    GHOST_CALL_RETURN(ghost_cu_cusparse_handle(&cusparse_handle));

    dt2 *localdot = NULL;
    dt1 *shift = NULL, scale, beta, sdelta, seta;
    ghost_densemat_t *z = NULL;

    one<dt1>(scale);

    GHOST_SPMV_PARSE_ARGS(options,argp,scale,beta,shift,localdot,z,sdelta,seta,dt2,dt1);

    if (options & GHOST_SPMV_AXPY) {
        one<dt1>(beta);
    } else if (!(options & GHOST_SPMV_AXPBY)) {
        zero<dt1>(beta);
    }
    
    CUSPARSE_CALL_RETURN(crskernel(cusparse_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_TRANSPOSE,mat->nrows,rhs->traits.ncols,mat->ncols,mat->nnz,&scale,descr,(dt1 *)CR(mat)->cumat->val, CR(mat)->cumat->rpt, CR(mat)->cumat->col, (dt1 *)rhs->cu_val, rhs->stride, &beta, (dt1 *)lhs->cu_val,lhs->stride));
    
    // This needs to be done prior to further computation because lhs is col-major and rhs is row-major.
    if (origlhs != lhs) {
        PERFWARNING_LOG("Transpose the result back");
        GHOST_CALL_RETURN(origlhs->fromVec(origlhs,lhs,0,0));
    }

    if (options & (GHOST_SPMV_SHIFT|GHOST_SPMV_VSHIFT)) {
        PERFWARNING_LOG("Shift will not be applied on-the-fly!");
        dt2 minusshift[rhs->traits.ncols];
        ghost_lidx_t col;
        if (options & GHOST_SPMV_SHIFT) {
            for (col=0; col<rhs->traits.ncols; col++) {
                minusshift[col] = -1.*(*(dt2 *)&scale)*(*(dt2 *)shift);
            }
        } else {
            for (col=0; col<rhs->traits.ncols; col++) {
                minusshift[col] = -1.*(*(dt2 *)&scale)*(((dt2 *)shift)[col]);
            }
        }
        lhs->vaxpy(lhs,rhs,minusshift);
    }
    
    if (options & GHOST_SPMV_DOT_ANY) {
        PERFWARNING_LOG("Dot product computation will be not be done on-the-fly!");
        memset(localdot,0,lhs->traits.ncols*3*sizeof(dt1));
        if (options & GHOST_SPMV_DOT_YY) {
            origlhs->dot(origlhs,&localdot[0],origlhs);
        }
        if (options & GHOST_SPMV_DOT_XY) {
            origlhs->dot(origlhs,&localdot[rhs->traits.ncols],rhs);
        }
        if (options & GHOST_SPMV_DOT_XX) {
            rhs->dot(rhs,&localdot[2*lhs->traits.ncols],rhs);
        }
            
    }
    if (options & GHOST_SPMV_CHAIN_AXPBY) {
        PERFWARNING_LOG("AXPBY will not be done on-the-fly!");
        z->axpby(z,lhs,&seta,&sdelta);
    }
    
    
    return GHOST_SUCCESS;
}

ghost_error_t ghost_cu_crs_spmv_selector(ghost_sparsemat_t *mat, ghost_densemat_t * lhs_in, ghost_densemat_t * rhs_in, ghost_spmv_flags_t options, va_list argp)
{
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_densemat_t *lhs = lhs_in, *rhs = rhs_in;
    ghost_densemat_traits_t lhstraits = lhs_in->traits;
    ghost_densemat_traits_t rhstraits = rhs_in->traits;
    
    if (mat->traits->datatype != lhs_in->traits.datatype) {
        ERROR_LOG("Mixed data types not implemented!");
        ret = GHOST_ERR_NOT_IMPLEMENTED;
        goto err;
    }
    if ((lhs_in->traits.flags & GHOST_DENSEMAT_SCATTERED) || (lhs_in->traits.storage == GHOST_DENSEMAT_ROWMAJOR)) {
        if (lhs_in->traits.flags & GHOST_DENSEMAT_SCATTERED) {
            PERFWARNING_LOG("Cloning and compressing lhs before operation because it is scattered");
        }
        if (lhs_in->traits.storage == GHOST_DENSEMAT_ROWMAJOR) {
            PERFWARNING_LOG("Cloning and transposing lhs before operation because it is row-major");
            lhstraits.storage = GHOST_DENSEMAT_COLMAJOR;
        }
        GHOST_CALL_GOTO(ghost_densemat_create(&lhs,lhs_in->context,lhstraits),err,ret);
        GHOST_CALL_GOTO(lhs->fromVec(lhs,lhs_in,0,0),err,ret);
    } else {
        lhs = lhs_in;
    }
    if (rhs_in->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        PERFWARNING_LOG("Cloning and compressing rhs before operation because it is scattered");
        GHOST_CALL_GOTO(ghost_densemat_create(&rhs,rhs_in->context,rhstraits),err,ret);
        GHOST_CALL_GOTO(rhs->fromVec(rhs,rhs_in,0,0),err,ret);
    } else {
        rhs = rhs_in;
    }


    INFO_LOG("Calling cuSparse CRS SpMV %d",options);

    if (lhs_in->traits.ncols == 1) {
        if (mat->traits->datatype & GHOST_DT_DOUBLE) {
            if (mat->traits->datatype & GHOST_DT_REAL) {
                GHOST_CALL_GOTO((ghost_cu_crsspmv_tmpl<double,double>(mat,lhs,lhs_in,rhs,options,argp,(cusparse_crs_spmv_kernel_t)cusparseDcsrmv)),err,ret);
            } else {
                GHOST_CALL_GOTO((ghost_cu_crsspmv_tmpl<cuDoubleComplex,complex double>(mat,lhs,lhs_in,rhs,options,argp,(cusparse_crs_spmv_kernel_t)cusparseZcsrmv)),err,ret);
            }
        } else {
            if (mat->traits->datatype & GHOST_DT_REAL) {
                GHOST_CALL_GOTO((ghost_cu_crsspmv_tmpl<float,float>(mat,lhs,lhs_in,rhs,options,argp,(cusparse_crs_spmv_kernel_t)cusparseScsrmv)),err,ret);
            } else {
                GHOST_CALL_GOTO((ghost_cu_crsspmv_tmpl<cuFloatComplex,complex float>(mat,lhs,lhs_in,rhs,options,argp,(cusparse_crs_spmv_kernel_t)cusparseCcsrmv)),err,ret);
            }
        }
    } else if (rhs_in->traits.storage == GHOST_DENSEMAT_COLMAJOR) {
        INFO_LOG("Calling col-major cuSparse CRS SpMMV");
        if (mat->traits->datatype & GHOST_DT_DOUBLE) {
            if (mat->traits->datatype & GHOST_DT_REAL) {
                GHOST_CALL_GOTO((ghost_cu_crsspmmv_cm_tmpl<double,double>(mat,lhs,lhs_in,rhs,options,argp,(cusparse_crs_spmmv_cm_kernel_t)cusparseDcsrmm)),err,ret);
            } else {
                GHOST_CALL_GOTO((ghost_cu_crsspmmv_cm_tmpl<cuDoubleComplex,complex double>(mat,lhs,lhs_in,rhs,options,argp,(cusparse_crs_spmmv_cm_kernel_t)cusparseZcsrmm)),err,ret);
            }
        } else {
            if (mat->traits->datatype & GHOST_DT_REAL) {
                GHOST_CALL_GOTO((ghost_cu_crsspmmv_cm_tmpl<float,float>(mat,lhs,lhs_in,rhs,options,argp,(cusparse_crs_spmmv_cm_kernel_t)cusparseScsrmm)),err,ret);
            } else {
                GHOST_CALL_GOTO((ghost_cu_crsspmmv_cm_tmpl<cuFloatComplex,complex float>(mat,lhs,lhs_in,rhs,options,argp,(cusparse_crs_spmmv_cm_kernel_t)cusparseCcsrmm)),err,ret);
            }
        }
    } else {
        INFO_LOG("Calling row-major cuSparse CRS SpMMV");
        if (mat->traits->datatype & GHOST_DT_DOUBLE) {
            if (mat->traits->datatype & GHOST_DT_REAL) {
                GHOST_CALL_GOTO((ghost_cu_crsspmmv_rm_tmpl<double,double>(mat,lhs,lhs_in,rhs,options,argp,(cusparse_crs_spmmv_rm_kernel_t)cusparseDcsrmm2)),err,ret);
            } else {
                GHOST_CALL_GOTO((ghost_cu_crsspmmv_rm_tmpl<cuDoubleComplex,complex double>(mat,lhs,lhs_in,rhs,options,argp,(cusparse_crs_spmmv_rm_kernel_t)cusparseZcsrmm2)),err,ret);
            }
        } else {
            if (mat->traits->datatype & GHOST_DT_REAL) {
                GHOST_CALL_GOTO((ghost_cu_crsspmmv_rm_tmpl<float,float>(mat,lhs,lhs_in,rhs,options,argp,(cusparse_crs_spmmv_rm_kernel_t)cusparseScsrmm2)),err,ret);
            } else {
                GHOST_CALL_GOTO((ghost_cu_crsspmmv_rm_tmpl<cuFloatComplex,complex float>(mat,lhs,lhs_in,rhs,options,argp,(cusparse_crs_spmmv_rm_kernel_t)cusparseCcsrmm2)),err,ret);
            }
        }
    }
    
    goto out;
err:
    ERROR_LOG("Error in CRS SpMV!");
out:
    if (lhs != lhs_in) {
        lhs->destroy(lhs);
    }
    if (rhs != rhs_in) {
        rhs->destroy(rhs);
    }

    return ret;


    /*cusparseHandle_t cusparse_handle;
      cusparseMatDescr_t descr;

      cusparseCreateMatDescr(&descr);
      GHOST_CALL_RETURN(ghost_cu_cusparse_handle(&cusparse_handle));

      double one = 1.;

      cusparseDcsrmv(cusparse_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,mat->nrows,mat->ncols,mat->nnz,&one,descr,(const double *)CR(mat)->cumat->val, CR(mat)->cumat->rpt, CR(mat)->cumat->col, (const double *)rhs->cu_val, &one, (double *)lhs->cu_val);

    //ERROR_LOG("CUDA CRS spMV not implemented");
    //return GHOST_ERR_NOT_IMPLEMENTED;

    INFO_LOG("ready"); 
    return GHOST_SUCCESS;*/

}
