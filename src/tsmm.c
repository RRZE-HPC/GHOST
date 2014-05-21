#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/math.h"
#include "ghost/instr.h"

#GHOST_FUNC_BEGIN#BLOCKSZ=4,8
ghost_error_t ghost_tsmm_BLOCKSZ(ghost_densemat_t *x, ghost_densemat_t *v, ghost_densemat_t *w, void *alpha)
{
    GHOST_INSTR_START(tsmm)
    ghost_error_t ret = GHOST_SUCCESS;


    if (!v->context) {
        ERROR_LOG("v needs to be distributed");
        ret = GHOST_ERR_INVALID_ARG;
        goto err;
    }
    if (!x->context) {
        ERROR_LOG("x needs to be distributed");
        ret = GHOST_ERR_INVALID_ARG;
        goto err;
    }
    if (w->traits.storage != GHOST_DENSEMAT_COLMAJOR) {
        ERROR_LOG("w needs to be present in col-major storage");
        ret = GHOST_ERR_INVALID_ARG;
        goto err;
    }
    if (x->traits.storage != GHOST_DENSEMAT_ROWMAJOR) {
        ERROR_LOG("x needs to be present in row-major storage");
        ret = GHOST_ERR_INVALID_ARG;
        goto err;
    }
    if (v->traits.storage != GHOST_DENSEMAT_ROWMAJOR) {
        ERROR_LOG("v needs to be present in row-major storage");
        ret = GHOST_ERR_INVALID_ARG;
        goto err;
    }
    if (v->traits.datatype != (GHOST_DT_DOUBLE|GHOST_DT_REAL)) {
        ERROR_LOG("Currently only double data supported");
        ret = GHOST_ERR_NOT_IMPLEMENTED;
        goto err;
    }
    if (v->traits.datatype != w->traits.datatype || v->traits.datatype != x->traits.datatype) {
        ERROR_LOG("Mixed datatypes not supported");
        ret = GHOST_ERR_INVALID_ARG;
        goto err;
    }

    ghost_idx_t k = w->traits.ncols;

    if (k != BLOCKSZ) {
        ERROR_LOG("Invalid dimensions");
        ret = GHOST_ERR_INVALID_ARG;
        goto err;
    }

    ghost_idx_t n = v->traits.nrows;
    ghost_idx_t m = v->traits.ncols;
    
    INFO_LOG("In TSMM with %"PRIDX"x%"PRIDX" <- %"PRIDX"x%"PRIDX" * %"PRIDX"x%"PRIDX,n,k,n,m,m,k);

    double * restrict vval, * restrict wval, * restrict xval;
    ghost_idx_t ldv, ldw, ldx;

    ldv = *v->stride;
    ldw = *w->stride;
    ldx = *x->stride;

    ghost_densemat_valptr(v,(void **)&vval);
    ghost_densemat_valptr(w,(void **)&wval);
    ghost_densemat_valptr(x,(void **)&xval);

    double dalpha = *(double *)alpha;
    ghost_idx_t i,j,s;
    
    if (BLOCKSZ == 4) {
#pragma omp parallel for private(j,s) schedule(runtime)
        for (i=0; i<n; i++) {
            for (s=0; s<BLOCKSZ; s++) {
#pragma simd
#pragma vector aligned
#pragma vector always
#pragma ivdep
                for (j=0; j<m; j++) {
                    xval[i*ldx+s] += dalpha*vval[i*ldv+j]*wval[s*ldw+j];
                }
            }
        }
    } else {
#pragma omp parallel for private(j,s) schedule(runtime)
        for (i=0; i<n; i++) {
            for (j=0; j<m; j++) {
#pragma simd
#pragma vector aligned
#pragma vector always
#pragma ivdep
                for (s=0; s<BLOCKSZ; s++) {
                    xval[i*ldx+s] += dalpha*vval[i*ldv+j]*wval[s*ldw+j];
                }
            }
        }
    }    


    goto out;
err:

out:
    GHOST_INSTR_STOP(tsmm)
    return ret;
}
#GHOST_FUNC_END

