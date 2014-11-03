#include "ghost/config.h"
#include "ghost/svqb.h"
#include "ghost/util.h"
#include "ghost/tsmttsm.h"
#include "ghost/tsmm.h"
#include <complex>

#ifdef GHOST_HAVE_LAPACK
#ifdef GHOST_HAVE_MKL
#include <mkl_lapacke.h>
#else 
#include <lapacke.h>
#endif

template<typename T, typename T_b>
static lapack_int call_eig_function(int matrix_order, char jobz, char uplo, lapack_int n, T *a, lapack_int lda, T_b *w)
{
    ERROR_LOG("This should not be called!");
    return -999;
}

template<>
static lapack_int call_eig_function<double,double>(int matrix_order, char jobz, char uplo, lapack_int n, double *a, lapack_int lda, double *w)
{
    return LAPACKE_dsyevd(matrix_order, jobz, uplo, n, a, lda, w);
}

template<>
static lapack_int call_eig_function<float,float>(int matrix_order, char jobz, char uplo, lapack_int n, float *a, lapack_int lda, float *w)
{
    return LAPACKE_ssyevd(matrix_order, jobz, uplo, n, a, lda, w);
}

template<>
static lapack_int call_eig_function<std::complex<float>,float>(int matrix_order, char jobz, char uplo, lapack_int n, std::complex<float> *a, lapack_int lda, float *w)
{
    return LAPACKE_cheevd(matrix_order, jobz, uplo, n, (lapack_complex_float *)a, lda, w);
}

template<>
static lapack_int call_eig_function<std::complex<double>,double>(int matrix_order, char jobz, char uplo, lapack_int n, std::complex<double> *a, lapack_int lda, double *w)
{
    return LAPACKE_zheevd(matrix_order, jobz, uplo, n, (lapack_complex_double *)a, lda, w);
}

    template <typename T, typename T_b>
static ghost_error_t ghost_svqb_tmpl (ghost_densemat_t * v_ot , ghost_densemat_t * v)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret = GHOST_SUCCESS;
    T one = 1.0;
    T zero = 0.0;
    ghost_idx_t i,j;
    ghost_idx_t n = v->traits.ncols;
    ghost_datatype_t DT = v->traits.datatype;
    ghost_densemat_t *x;
    ghost_idx_t ldx;
    T_b *eigs, *D;
    ghost_densemat_traits_t xtraits = GHOST_DENSEMAT_TRAITS_INITIALIZER;
    
    GHOST_CALL_GOTO(ghost_malloc((void **)&eigs, n*sizeof(double)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&D, n*sizeof(double)),err,ret);
    
    xtraits.flags |= GHOST_DENSEMAT_NO_HALO; 
    xtraits.ncols = n;
    xtraits.nrows = n;
    xtraits.storage = GHOST_DENSEMAT_COLMAJOR;
    xtraits.datatype = DT;
    GHOST_CALL_GOTO(ghost_densemat_create(&x,NULL,xtraits),err,ret);
    GHOST_CALL_GOTO(x->fromScalar(x,&zero),err,ret);
    ldx = *x->stride;

    T *  xval;
    GHOST_CALL_GOTO(ghost_densemat_valptr(x,(void **)&xval),err,ret);
    

    GHOST_CALL_GOTO(ghost_tsmttsm( x, v, v,&one,&zero),err,ret);
    
    for (i=0;i<n;i++) {
        D[i] = std::real((T)1./std::sqrt(xval[i*ldx+i]));
    }
    
    for (i=0;i<n;i++) {
        for( j=0;j<n;j++) { 
            xval[i*ldx+j] *= D[i]*D[j];
        }
    }
    
    if (call_eig_function<T,T_b>( LAPACK_COL_MAJOR, 'V' , 'U', n, xval, ldx, eigs)) {
        ERROR_LOG("LAPACK eigenvalue function failed!");
        ret = GHOST_ERR_LAPACK;
        goto err;
    }

    for( i=0;i<n;i++) {
                eigs[i] = 1./sqrt(eigs[i]);
                for( j=0;j<n;j++)  xval[i*ldx+j] *= D[j]*eigs[i];
     }
    GHOST_CALL_GOTO(ghost_tsmm( v_ot, v, x, &one, &zero),err,ret);
   
    goto out;
err:

out: 
    x->destroy(x);
    free(eigs);
    free(D);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
}

ghost_error_t ghost_svqb(ghost_densemat_t * v_ot , ghost_densemat_t * v)
{
    if (v->traits.datatype & GHOST_DT_COMPLEX) {
        if (v->traits.datatype & GHOST_DT_DOUBLE) {
            return ghost_svqb_tmpl<std::complex<double>, double>(v_ot, v);
        } else {
            return ghost_svqb_tmpl<std::complex<float>, float>(v_ot, v);
        }
    } else {
        if (v->traits.datatype & GHOST_DT_DOUBLE) {
            return ghost_svqb_tmpl<double, double>(v_ot, v);
        } else {
            return ghost_svqb_tmpl<float, float>(v_ot, v);
        }
    }
}

#else
ghost_error_t ghost_svqb(ghost_densemat_t * v_ot , ghost_densemat_t * v)
{
    UNUSED(v_ot);
    UNUSED(v);
    ERROR_LOG("LAPACKE not found!");
    return GHOST_ERR_NOT_IMPLEMENTED;
}
#endif
