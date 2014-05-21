#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/math.h"
#include "ghost/instr.h"
#include "ghost/locality.h"

#GHOST_FUNC_BEGIN#BLOCKSZ=4,8
ghost_error_t ghost_tsmttsm_BLOCKSZ(ghost_densemat_t *x, ghost_densemat_t *v, ghost_densemat_t *w, void *alpha, void *beta)
{
    GHOST_INSTR_START(tsmttsm)
    ghost_error_t ret = GHOST_SUCCESS;
    if (!v->context) {
        ERROR_LOG("v needs to be distributed");
        ret = GHOST_ERR_INVALID_ARG;
        goto err;
    }
    if (!w->context) {
        ERROR_LOG("w needs to be distributed");
        ret = GHOST_ERR_INVALID_ARG;
        goto err;
    }
    if (w->traits.storage != GHOST_DENSEMAT_ROWMAJOR) {
        ERROR_LOG("w needs to be present in row-major storage");
        ret = GHOST_ERR_INVALID_ARG;
        goto err;
    }
    if (v->traits.storage != GHOST_DENSEMAT_ROWMAJOR) {
        ERROR_LOG("v needs to be present in row-major storage");
        ret = GHOST_ERR_INVALID_ARG;
        goto err;
    }
    /*if (x->traits.storage != GHOST_DENSEMAT_COLMAJOR) {
        ERROR_LOG("x needs to be present in col-major storage");
        ret = GHOST_ERR_INVALID_ARG;
        goto err;
    }*/
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
    
    int myrank=0;

    GHOST_CALL_GOTO(ghost_rank(&myrank,v->context->mpicomm),err,ret);

    ghost_idx_t n = v->traits.nrows;
    ghost_idx_t m = v->traits.ncols;
    
    double * restrict vval, * restrict wval, * restrict xval;
    ghost_idx_t ldv, ldw, ldx;

    ldv = *v->stride;
    ldw = *w->stride;
    ldx = *x->stride;

    ghost_densemat_valptr(v,(void **)&vval);
    ghost_densemat_valptr(w,(void **)&wval);
    ghost_densemat_valptr(x,(void **)&xval);
    
    double dalpha = *(double *)alpha;
    double dbeta = *(double *)beta;
    
    // make sure that the initial x only gets added up once
    if (myrank) {
        dbeta = 0.;
    }

    ghost_idx_t i,j;
    
    if (x->traits.storage == GHOST_DENSEMAT_COLMAJOR) {
        for (j=0; j<m; j++) {
#GHOST_UNROLL#double tmp@ = dbeta*xval[@*ldx+j];#BLOCKSZ
#pragma omp parallel for schedule(runtime) reduction(+:\
        #GHOST_UNROLL#tmp@,\#BLOCKSZ-1
        tmpBLOCKSZ-1)
            for (i=0; i<n; i++) {
                #GHOST_UNROLL#tmp@ += dalpha*vval[i*ldv+j]*wval[i*ldw+@];#BLOCKSZ
            }
            #GHOST_UNROLL#xval[@*ldx+j] = tmp@;#BLOCKSZ
        }
    } else {
        for (j=0; j<m; j++) {
#GHOST_UNROLL#double tmp@ = dbeta*xval[j*ldx+@];#BLOCKSZ
#pragma omp parallel for schedule(runtime) reduction(+:\
        #GHOST_UNROLL#tmp@,\#BLOCKSZ-1
        tmpBLOCKSZ-1)
            for (i=0; i<n; i++) {
                #GHOST_UNROLL#tmp@ += dalpha*vval[i*ldv+j]*wval[i*ldw+@];#BLOCKSZ
            }
            #GHOST_UNROLL#xval[j*ldx+@] = tmp@;#BLOCKSZ
        }
    }
   
#ifdef GHOST_HAVE_MPI
    MPI_CALL_GOTO(MPI_Allreduce(MPI_IN_PLACE,xval,ldx*BLOCKSZ,MPI_DOUBLE,MPI_SUM,v->context->mpicomm),err,ret);
#endif
    
    goto out;
err:

out:
    GHOST_INSTR_STOP(tsmttsm)
    return ret;
}
#GHOST_FUNC_END
