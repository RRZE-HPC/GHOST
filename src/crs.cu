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
static ghost_error_t ghost_cu_crsspmv_tmpl(ghost_sparsemat_t *mat, ghost_densemat_t * lhs, ghost_densemat_t * rhs, ghost_spmv_flags_t options, va_list argp, cusparse_crs_spmv_kernel_t crskernel)
{
    cusparseHandle_t cusparse_handle;
    cusparseMatDescr_t descr;

    cusparseCreateMatDescr(&descr);
    GHOST_CALL_RETURN(ghost_cu_cusparse_handle(&cusparse_handle));

    dt2 *localdot = NULL;
    dt1 *shift = NULL, scale, beta;

    one<dt1>(scale);

    GHOST_SPMV_PARSE_ARGS(options,argp,scale,beta,shift,localdot,dt2,dt1);

    if (options & GHOST_SPMV_AXPY) {
        one<dt1>(beta);
    } else {
        zero<dt1>(beta);
    }
    
    if (localdot || shift) {
        WARNING_LOG("Localdot or shift are not NULL, something went wrong!");
    } 

    CUSPARSE_CALL_RETURN(crskernel(cusparse_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,mat->nrows,mat->ncols,mat->nnz,&scale,descr,(dt1 *)CR(mat)->cumat->val, CR(mat)->cumat->rpt, CR(mat)->cumat->col, (dt1 *)rhs->cu_val, &beta, (dt1 *)lhs->cu_val));

    return GHOST_SUCCESS;
}
    
    template<typename dt1, typename dt2>
static ghost_error_t ghost_cu_crsspmmv_cm_tmpl(ghost_sparsemat_t *mat, ghost_densemat_t * lhs, ghost_densemat_t * rhs, ghost_spmv_flags_t options, va_list argp, cusparse_crs_spmmv_cm_kernel_t crskernel)
{
    cusparseHandle_t cusparse_handle;
    cusparseMatDescr_t descr;

    cusparseCreateMatDescr(&descr);
    GHOST_CALL_RETURN(ghost_cu_cusparse_handle(&cusparse_handle));

    dt2 *localdot = NULL;
    dt1 *shift = NULL, scale, beta;

    one<dt1>(scale);

    GHOST_SPMV_PARSE_ARGS(options,argp,scale,beta,shift,localdot,dt2,dt1);

    if (options & GHOST_SPMV_AXPY) {
        one<dt1>(beta);
    } else {
        zero<dt1>(beta);
    }
    
    if (localdot || shift) {
        WARNING_LOG("Localdot or shift are not NULL, something went wrong!");
    } 

    crskernel(cusparse_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,mat->nrows,rhs->traits.ncols,mat->ncols,mat->nnz,&scale,descr,(dt1 *)CR(mat)->cumat->val, CR(mat)->cumat->rpt, CR(mat)->cumat->col, (dt1 *)rhs->cu_val, rhs->stride, &beta, (dt1 *)lhs->cu_val, lhs->stride);

    return GHOST_SUCCESS;
}

    template<typename dt1, typename dt2>
static ghost_error_t ghost_cu_crsspmmv_rm_tmpl(ghost_sparsemat_t *mat, ghost_densemat_t * lhs, ghost_densemat_t * rhs, ghost_spmv_flags_t options, va_list argp, cusparse_crs_spmmv_rm_kernel_t crskernel)
{
    cusparseHandle_t cusparse_handle;
    cusparseMatDescr_t descr;

    cusparseCreateMatDescr(&descr);
    GHOST_CALL_RETURN(ghost_cu_cusparse_handle(&cusparse_handle));

    dt2 *localdot = NULL;
    dt1 *shift = NULL, scale, beta;

    one<dt1>(scale);

    GHOST_SPMV_PARSE_ARGS(options,argp,scale,beta,shift,localdot,dt2,dt1);

    if (options & GHOST_SPMV_AXPY) {
        one<dt1>(beta);
    } else {
        zero<dt1>(beta);
    }
    
    if (localdot || shift) {
        WARNING_LOG("Localdot or shift are not NULL, something went wrong!");
    } 

    CUSPARSE_CALL_RETURN(crskernel(cusparse_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_TRANSPOSE,mat->nrows,rhs->traits.ncols,mat->ncols,mat->nnz,&scale,descr,(dt1 *)CR(mat)->cumat->val, CR(mat)->cumat->rpt, CR(mat)->cumat->col, (dt1 *)rhs->cu_val, rhs->stride, &beta, (dt1 *)lhs->cu_val,lhs->stride));

    return GHOST_SUCCESS;
}

ghost_error_t ghost_cu_crs_spmv_selector(ghost_sparsemat_t *mat, ghost_densemat_t * lhs, ghost_densemat_t * rhs, ghost_spmv_flags_t options, va_list argp)
{
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_densemat_t *lhscompact, *rhscompact;
    
    if (mat->traits->datatype != lhs->traits.datatype) {
        ERROR_LOG("Mixed data types not implemented!");
        ret = GHOST_ERR_NOT_IMPLEMENTED;
        goto err;
    }
    if (options & GHOST_SPMV_DOT_ANY) {
        ERROR_LOG("Localdot not implemented!");
        ret = GHOST_ERR_NOT_IMPLEMENTED;
        goto err;
    }
    if (options & GHOST_SPMV_SHIFT) {
        ERROR_LOG("Shift not implemented!");
        ret = GHOST_ERR_NOT_IMPLEMENTED;
        goto err;
    }
    
    if (lhs->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        INFO_LOG("Cloning (and compressing) lhs before operation");
        GHOST_CALL_GOTO(lhs->clone(lhs,&lhscompact,lhs->traits.nrows,0,lhs->traits.ncols,0),err,ret);
    } else {
        lhscompact = lhs;
    }
    if (rhs->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        INFO_LOG("Cloning (and compressing) v2 before operation");
        GHOST_CALL_GOTO(rhs->clone(rhs,&rhscompact,rhs->traits.nrows,0,rhs->traits.ncols,0),err,ret);
    } else {
        rhscompact = rhs;
    }


    INFO_LOG("Calling cuSparse CRS SpMV %d",options);

    if (lhs->traits.ncols == 1) {
        if (mat->traits->datatype & GHOST_DT_DOUBLE) {
            if (mat->traits->datatype & GHOST_DT_REAL) {
                GHOST_CALL_GOTO((ghost_cu_crsspmv_tmpl<double,double>(mat,lhscompact,rhscompact,options,argp,(cusparse_crs_spmv_kernel_t)cusparseDcsrmv)),err,ret);
            } else {
                GHOST_CALL_GOTO((ghost_cu_crsspmv_tmpl<cuDoubleComplex,complex double>(mat,lhscompact,rhscompact,options,argp,(cusparse_crs_spmv_kernel_t)cusparseZcsrmv)),err,ret);
            }
        } else {
            if (mat->traits->datatype & GHOST_DT_REAL) {
                GHOST_CALL_GOTO((ghost_cu_crsspmv_tmpl<float,float>(mat,lhscompact,rhscompact,options,argp,(cusparse_crs_spmv_kernel_t)cusparseScsrmv)),err,ret);
            } else {
                GHOST_CALL_GOTO((ghost_cu_crsspmv_tmpl<cuFloatComplex,complex float>(mat,lhscompact,rhscompact,options,argp,(cusparse_crs_spmv_kernel_t)cusparseCcsrmv)),err,ret);
            }
        }
    } else if (rhs->traits.storage == GHOST_DENSEMAT_COLMAJOR) {
        INFO_LOG("Calling col-major cuSparse CRS SpMMV");
        if (mat->traits->datatype & GHOST_DT_DOUBLE) {
            if (mat->traits->datatype & GHOST_DT_REAL) {
                GHOST_CALL_GOTO((ghost_cu_crsspmmv_cm_tmpl<double,double>(mat,lhscompact,rhscompact,options,argp,(cusparse_crs_spmmv_cm_kernel_t)cusparseDcsrmm)),err,ret);
            } else {
                GHOST_CALL_GOTO((ghost_cu_crsspmmv_cm_tmpl<cuDoubleComplex,complex double>(mat,lhscompact,rhscompact,options,argp,(cusparse_crs_spmmv_cm_kernel_t)cusparseZcsrmm)),err,ret);
            }
        } else {
            if (mat->traits->datatype & GHOST_DT_REAL) {
                GHOST_CALL_GOTO((ghost_cu_crsspmmv_cm_tmpl<float,float>(mat,lhscompact,rhscompact,options,argp,(cusparse_crs_spmmv_cm_kernel_t)cusparseScsrmm)),err,ret);
            } else {
                GHOST_CALL_GOTO((ghost_cu_crsspmmv_cm_tmpl<cuFloatComplex,complex float>(mat,lhscompact,rhscompact,options,argp,(cusparse_crs_spmmv_cm_kernel_t)cusparseCcsrmm)),err,ret);
            }
        }
    } else {
        if (lhs->traits.storage == GHOST_DENSEMAT_ROWMAJOR) {
            PERFWARNING_LOG("Need to memtranspose the result densemat!");
            ghost_densemat_t *lhscompactnew;
            ghost_densemat_traits_t transtraits = lhscompact->traits;
            transtraits.storage = GHOST_DENSEMAT_COLMAJOR;
            GHOST_CALL_GOTO(ghost_densemat_create(&lhscompactnew,lhscompact->context,transtraits),err,ret);
            GHOST_CALL_GOTO(lhscompactnew->fromVec(lhscompactnew,lhscompact,0,0),err,ret);
            if (lhscompact != lhs) {
                lhscompact->destroy(lhscompact);
            }
            lhscompact = lhscompactnew;
        }
        INFO_LOG("Calling row-major cuSparse CRS SpMMV");
        if (mat->traits->datatype & GHOST_DT_DOUBLE) {
            if (mat->traits->datatype & GHOST_DT_REAL) {
                GHOST_CALL_GOTO((ghost_cu_crsspmmv_rm_tmpl<double,double>(mat,lhscompact,rhscompact,options,argp,(cusparse_crs_spmmv_rm_kernel_t)cusparseDcsrmm2)),err,ret);
            } else {
                GHOST_CALL_GOTO((ghost_cu_crsspmmv_rm_tmpl<cuDoubleComplex,complex double>(mat,lhscompact,rhscompact,options,argp,(cusparse_crs_spmmv_rm_kernel_t)cusparseZcsrmm2)),err,ret);
            }
        } else {
            if (mat->traits->datatype & GHOST_DT_REAL) {
                GHOST_CALL_GOTO((ghost_cu_crsspmmv_rm_tmpl<float,float>(mat,lhscompact,rhscompact,options,argp,(cusparse_crs_spmmv_rm_kernel_t)cusparseScsrmm2)),err,ret);
            } else {
                GHOST_CALL_GOTO((ghost_cu_crsspmmv_rm_tmpl<cuFloatComplex,complex float>(mat,lhscompact,rhscompact,options,argp,(cusparse_crs_spmmv_rm_kernel_t)cusparseCcsrmm2)),err,ret);
            }
        }
    }

    goto out;
err:
    ERROR_LOG("Error in CRS SpMV!");
out:
    if (lhscompact != lhs) {
        GHOST_CALL_RETURN(lhs->fromVec(lhs,lhscompact,0,0));
        lhscompact->destroy(lhscompact);
    }
    if (rhscompact != rhs) {
        rhscompact->destroy(rhscompact);
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
