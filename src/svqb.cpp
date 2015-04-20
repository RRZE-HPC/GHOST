#include "ghost/config.h"
#include "ghost/svqb.h"
#include "ghost/util.h"
#include "ghost/tsmttsm.h"
#include "ghost/tsmm.h"
#include <complex>
#include <cstdlib>

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
lapack_int call_eig_function<double,double>(int matrix_order, char jobz, char uplo, lapack_int n, double *a, lapack_int lda, double *w)
{
    return LAPACKE_dsyevd(matrix_order, jobz, uplo, n, a, lda, w);
}

template<>
lapack_int call_eig_function<float,float>(int matrix_order, char jobz, char uplo, lapack_int n, float *a, lapack_int lda, float *w)
{
    return LAPACKE_ssyevd(matrix_order, jobz, uplo, n, a, lda, w);
}

template<>
lapack_int call_eig_function<std::complex<float>,float>(int matrix_order, char jobz, char uplo, lapack_int n, std::complex<float> *a, lapack_int lda, float *w)
{
    return LAPACKE_cheevd(matrix_order, jobz, uplo, n, (lapack_complex_float *)a, lda, w);
}

template<>
lapack_int call_eig_function<std::complex<double>,double>(int matrix_order, char jobz, char uplo, lapack_int n, std::complex<double> *a, lapack_int lda, double *w)
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
    ghost_lidx_t n_set_rand = 0;
    ghost_lidx_t *set_rand;
    ghost_idx_t i,j;
    ghost_idx_t n = v->traits.ncols;
    ghost_datatype_t DT = v->traits.datatype;
    ghost_densemat_t *x;
    ghost_idx_t ldx;
    T_b *eigs, *D;
    ghost_densemat_traits_t xtraits = GHOST_DENSEMAT_TRAITS_INITIALIZER;
    
    GHOST_CALL_GOTO(ghost_malloc((void **)&eigs, n*sizeof(T_b)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&D, n*sizeof(T_b)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&set_rand, n*sizeof(ghost_lidx_t)),err,ret);
    
    xtraits.flags |= GHOST_DENSEMAT_NO_HALO; 
    xtraits.ncols = n;
    xtraits.nrows = n;
    xtraits.storage = GHOST_DENSEMAT_COLMAJOR;
    xtraits.datatype = DT;
    xtraits.location = GHOST_LOCATION_HOST|GHOST_LOCATION_DEVICE;
    GHOST_CALL_GOTO(ghost_densemat_create(&x,NULL,xtraits),err,ret);
    GHOST_CALL_GOTO(x->fromScalar(x,&zero),err,ret);
    ldx = x->stride;

    T *  xval;
    GHOST_CALL_GOTO(ghost_densemat_valptr(x,(void **)&xval),err,ret);
    

    GHOST_CALL_GOTO(ghost_tsmttsm_kahan( x, v, v,&one,&zero,GHOST_GEMM_ALL_REDUCE,1),err,ret);
    x->download(x);
   
    for (i=0;i<n;i++) {
        D[i] = (T_b)1./std::sqrt(std::real(xval[i*ldx+i]));
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
    
    for ( i=0;i<n;i++){  
        if( eigs[i] <  0. ){
           eigs[i] = -eigs[i];
        }
        if(eigs[i] <  (T_b)1.e-14*eigs[n-1] ){  // TODO make it for single precision, too
           eigs[i] += (T_b)1.e-14*eigs[n-1];
           set_rand[n_set_rand] = i;
           n_set_rand++;
        }

      eigs[i] = (T_b)1./std::sqrt(eigs[i]);
    }
    for ( i=0;i<n;i++) {
         for( j=0;j<n;j++) {
            xval[i*ldx+j] *= D[j]*eigs[i];
         }
    }
    
    x->upload(x);
    
    GHOST_CALL_GOTO(ghost_tsmm( v_ot, v, x, &one, &zero),err,ret);
   
   if( n_set_rand > 0 ){
      ghost_densemat_t * vec_view2rand;
      v_ot->viewScatteredCols(v_ot, &vec_view2rand, n_set_rand, set_rand);
      vec_view2rand->fromRand( vec_view2rand );
      vec_view2rand->destroy(vec_view2rand);
     }  
   
   
    goto out;
err:

out: 
    x->destroy(x);
    free(eigs);
    free(D);
    free(set_rand);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
}

    template <typename T, typename T_b>
static ghost_error_t ghost_blockortho_tmpl (ghost_densemat_t * w , ghost_densemat_t * v)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret = GHOST_SUCCESS;
    T one = 1.0;
    T zero = 0.0;
    T minusone = -1.;
    ghost_idx_t m = v->traits.ncols;
    ghost_idx_t n = w->traits.ncols;
    ghost_datatype_t DT = v->traits.datatype;
    ghost_densemat_t *x;
    //ghost_idx_t ldx;
    ghost_densemat_traits_t xtraits = GHOST_DENSEMAT_TRAITS_INITIALIZER;
        
    xtraits.flags |= GHOST_DENSEMAT_NO_HALO; 
    xtraits.ncols = n;
    xtraits.nrows = m;
    xtraits.storage = GHOST_DENSEMAT_COLMAJOR;
    xtraits.datatype = DT;
    GHOST_CALL_GOTO(ghost_densemat_create(&x,NULL,xtraits),err,ret);
    GHOST_CALL_GOTO(x->fromScalar(x,&zero),err,ret);
    //ldx = *x->stride;

    T *  xval;
    GHOST_CALL_GOTO(ghost_densemat_valptr(x,(void **)&xval),err,ret);
    
    GHOST_CALL_GOTO(ghost_tsmttsm_kahan( x, v, w,&one,&zero,GHOST_GEMM_ALL_REDUCE,1),err,ret);
    GHOST_CALL_GOTO(ghost_tsmm( w, v, x, &one, &minusone),err,ret);
       
    goto out;
err:

out: 
    x->destroy(x);
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

ghost_error_t ghost_blockortho(ghost_densemat_t * w , ghost_densemat_t * v)
{
    if (v->traits.datatype & GHOST_DT_COMPLEX) {
        if (v->traits.datatype & GHOST_DT_DOUBLE) {
            return ghost_blockortho_tmpl<std::complex<double>, double>(w, v);
        } else {
            return ghost_blockortho_tmpl<std::complex<float>, float>(w, v);
        }
    } else {
        if (v->traits.datatype & GHOST_DT_DOUBLE) {
            return ghost_blockortho_tmpl<double, double>(w, v);
        } else {
            return ghost_blockortho_tmpl<float, float>(w, v);
        }
    }
}

    template <typename T, typename T_b>
static ghost_error_t ghost_svd_deflation_tmpl ( ghost_lidx_t *svd_offset, ghost_densemat_t * ot_vec, ghost_densemat_t * vec, float limit)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret = GHOST_SUCCESS;
    T one = 1.0;
    T zero = 0.0;
    ghost_idx_t i,j;
    ghost_idx_t n = vec->traits.ncols;
    ghost_datatype_t DT = vec->traits.datatype;
    ghost_densemat_t *x;
    ghost_idx_t ldx;
    ghost_densemat_traits_t xtraits = GHOST_DENSEMAT_TRAITS_INITIALIZER;
    T_b * eigs;
    
    GHOST_CALL_GOTO(ghost_malloc((void **)&eigs, n*sizeof(T_b)),err,ret);
    
    xtraits.flags |= GHOST_DENSEMAT_NO_HALO; 
    xtraits.ncols = n;
    xtraits.nrows = n;
    xtraits.storage = GHOST_DENSEMAT_COLMAJOR;
    xtraits.datatype = DT;
    GHOST_CALL_GOTO(ghost_densemat_create(&x,NULL,xtraits),err,ret);
    GHOST_CALL_GOTO(x->fromScalar(x,&zero),err,ret);
    ldx = x->stride;

    T *  xval;
    GHOST_CALL_GOTO(ghost_densemat_valptr(x,(void **)&xval),err,ret);
    
    
    GHOST_CALL_GOTO(ghost_tsmttsm_kahan( x, vec, vec,&one,&zero,GHOST_GEMM_ALL_REDUCE,1),err,ret);
    
    if (call_eig_function<T,T_b>( LAPACK_COL_MAJOR, 'V' , 'U', n, xval, ldx, eigs)) {
        ERROR_LOG("LAPACK eigenvalue function failed!");
        ret = GHOST_ERR_LAPACK;
        goto err;
    }
    
    *svd_offset=0;
    for ( i=0;i<n;i++){  
      if( eigs[i] <  ((T_b)(limit)) ){
        (*svd_offset)++;
        eigs[i] = -eigs[i];
        }
        eigs[i] = (T_b)1./std::sqrt(eigs[i]);
    }
    
    for ( i=0;i<n;i++) {
         for( j=0;j<n;j++) {
            xval[i*ldx+j] *= eigs[i];
         }
    }
    GHOST_CALL_GOTO(ghost_tsmm( ot_vec, vec, x, &one, &zero),err,ret);
    
    goto out;
err:

out: 
    x->destroy(x);
    free(eigs);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);
    
    return ret;
}

ghost_error_t ghost_svd_deflation( ghost_lidx_t *svd_offset, ghost_densemat_t * ot_vec, ghost_densemat_t * vec, float limit)
{
    if (vec->traits.datatype & GHOST_DT_COMPLEX) {
        if (vec->traits.datatype & GHOST_DT_DOUBLE) {
            return ghost_svd_deflation_tmpl<std::complex<double>, double>( svd_offset, ot_vec,  vec, limit);
        } else {
            return ghost_svd_deflation_tmpl<std::complex<float>, float>( svd_offset, ot_vec,  vec, limit);
        }
    } else {
        if (vec->traits.datatype & GHOST_DT_DOUBLE) {
            return ghost_svd_deflation_tmpl<double, double>( svd_offset, ot_vec,  vec, limit);
        } else {
            return ghost_svd_deflation_tmpl<float, float>( svd_offset, ot_vec,  vec, limit);
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
