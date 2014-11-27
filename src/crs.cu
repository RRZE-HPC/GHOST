#include "ghost/config.h"
#undef GHOST_HAVE_MPI
#include "ghost/types.h"
#include "ghost/crs.h"
#include "ghost/log.h"
#include "ghost/cu_util.h"
#include "ghost/cu_complex.h"
#include "ghost/util.h"

#include <complex.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

typedef cusparseStatus_t (*crskernel_t) (cusparseHandle_t handle, cusparseOperation_t transA, 
        int m, int n, int nnz, const void           *alpha, 
        const cusparseMatDescr_t descrA, 
        const void           *csrValA, 
        const void *csrRowPtrA, const void *csrColIndA,
        const void           *x, const void           *beta, 
        void           *y);

    template<typename dt1, typename dt2>
static ghost_error_t ghost_cu_crsspmv_tmpl(ghost_sparsemat_t *mat, ghost_densemat_t * lhs, ghost_densemat_t * rhs, ghost_spmv_flags_t options, va_list argp, crskernel_t crskernel)
{
    if (options & GHOST_SPMV_DOT_ANY) {
        ERROR_LOG("Localdot not implemented!");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
    if (options & GHOST_SPMV_SHIFT) {
        ERROR_LOG("Shift not implemented!");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }

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
    
    ghost_densemat_t *lhscompact, *rhscompact;
    void *lhsval, *rhsval;
    
    if (lhs->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        INFO_LOG("Cloning (and compressing) lhs before operation");
        GHOST_CALL_RETURN(lhs->clone(lhs,&lhscompact,lhs->traits.nrows,0,lhs->traits.ncols,0));
    } else {
        lhscompact = lhs;
    }
    if (rhs->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        INFO_LOG("Cloning (and compressing) v2 before operation");
        GHOST_CALL_RETURN(rhs->clone(rhs,&rhscompact,rhs->traits.nrows,0,rhs->traits.ncols,0));
    } else {
        rhscompact = rhs;
    }
    GHOST_CALL_RETURN(ghost_densemat_cu_valptr(lhscompact,&lhsval));
    GHOST_CALL_RETURN(ghost_densemat_cu_valptr(rhscompact,&rhsval));


    if (localdot || shift) {
        WARNING_LOG("Localdot or shift are not NULL, something went wrong!");
    } 

    crskernel(cusparse_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,mat->nrows,mat->ncols,mat->nnz,&scale,descr,(dt1 *)CR(mat)->cumat->val, CR(mat)->cumat->rpt, CR(mat)->cumat->col, (dt1 *)rhsval, &beta, (dt1 *)lhsval);

    if (lhscompact != lhs) {
        GHOST_CALL_RETURN(lhs->fromVec(lhs,lhscompact,0,0));
        lhscompact->destroy(lhscompact);
    }
    if (rhscompact != rhs) {
        rhscompact->destroy(rhscompact);
    }

    return GHOST_SUCCESS;

}

ghost_error_t ghost_cu_crs_spmv_selector(ghost_sparsemat_t *mat, ghost_densemat_t * lhs, ghost_densemat_t * rhs, ghost_spmv_flags_t options, va_list argp)
{
    if (mat->traits->datatype != lhs->traits.datatype) {
        ERROR_LOG("Mixed data types not implemented!");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }

    INFO_LOG("Calling cuSparse CRS SpMV");

    if (mat->traits->datatype & GHOST_DT_DOUBLE) {
        if (mat->traits->datatype & GHOST_DT_REAL) {
            return ghost_cu_crsspmv_tmpl<double,double>(mat,lhs,rhs,options,argp,(crskernel_t)cusparseDcsrmv);
        } else {
            return ghost_cu_crsspmv_tmpl<cuDoubleComplex,complex double>(mat,lhs,rhs,options,argp,(crskernel_t)cusparseZcsrmv);
        }
    } else {
        if (mat->traits->datatype & GHOST_DT_REAL) {
            return ghost_cu_crsspmv_tmpl<float,float>(mat,lhs,rhs,options,argp,(crskernel_t)cusparseScsrmv);
        } else {
            return ghost_cu_crsspmv_tmpl<cuFloatComplex,complex float>(mat,lhs,rhs,options,argp,(crskernel_t)cusparseCcsrmv);
        }
    }


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
