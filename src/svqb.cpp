#include "ghost/config.h"
#include "ghost/svqb.h"
#include "ghost/util.h"
#include "ghost/tsmttsm.h"
#include "ghost/tsmm.h"
#include <complex>
#include <cstdlib>


    template <typename T, typename T_b>
static ghost_error ghost_blockortho_tmpl (ghost_densemat * w , ghost_densemat * v)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_SOLVER);
    ghost_error ret = GHOST_SUCCESS;
    T one = 1.0;
    T zero = 0.0;
    T minusone = -1.;
    ghost_lidx m = v->traits.ncols;
    ghost_lidx n = w->traits.ncols;
    ghost_datatype DT = v->traits.datatype;
    ghost_densemat *x;
    ghost_densemat_traits xtraits = GHOST_DENSEMAT_TRAITS_INITIALIZER;
        
    xtraits.flags |= GHOST_DENSEMAT_NO_HALO; 
    xtraits.ncols = n;
    xtraits.nrows = m;
    xtraits.storage = GHOST_DENSEMAT_COLMAJOR;
    xtraits.datatype = DT;
    GHOST_CALL_GOTO(ghost_densemat_create(&x,NULL,xtraits),err,ret);
    GHOST_CALL_GOTO(x->fromScalar(x,&zero),err,ret);
    //ldx = *x->stride;

    
    //GHOST_CALL_GOTO(ghost_tsmttsm( x, v, w,&one,&zero,GHOST_GEMM_ALL_REDUCE,1),err,ret);
    GHOST_CALL_GOTO(ghost_tsmttsm( x, v, w,&one,&zero,GHOST_GEMM_ALL_REDUCE,1,GHOST_GEMM_KAHAN),err,ret);
    GHOST_CALL_GOTO(ghost_tsmm( w, v, x, &one, &minusone),err,ret);
       
    goto out;
err:

out: 
    ghost_densemat_destroy(x);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_SOLVER);

    return ret;
}

ghost_error ghost_blockortho(ghost_densemat * w , ghost_densemat * v)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);
    
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

#ifdef GHOST_HAVE_LAPACK
#ifdef GHOST_HAVE_MKL
#include <mkl_lapacke.h>
#else 
#include <lapacke.h>
#endif

template<typename T, typename T_b>
static lapack_int call_eig_function(int matrix_order, char jobz, char uplo, lapack_int n, T *a, lapack_int lda, T_b *w)
{
    UNUSED(matrix_order);
    UNUSED(jobz);
    UNUSED(uplo);
    UNUSED(n);
    UNUSED(a);
    UNUSED(lda);
    UNUSED(w);
    ERROR_LOG("This should not be called!");
    return -999;
}

template<>
lapack_int call_eig_function<double,double>(int matrix_order, char jobz, char uplo, lapack_int n, double *a, lapack_int lda, double *w)
{
    return LAPACKE_dsyev(matrix_order, jobz, uplo, n, a, lda, w);
}

template<>
lapack_int call_eig_function<float,float>(int matrix_order, char jobz, char uplo, lapack_int n, float *a, lapack_int lda, float *w)
{
    return LAPACKE_ssyev(matrix_order, jobz, uplo, n, a, lda, w);
}

template<>
lapack_int call_eig_function<std::complex<float>,float>(int matrix_order, char jobz, char uplo, lapack_int n, std::complex<float> *a, lapack_int lda, float *w)
{
    return LAPACKE_cheev(matrix_order, jobz, uplo, n, (lapack_complex_float *)a, lda, w);
}

template<>
lapack_int call_eig_function<std::complex<double>,double>(int matrix_order, char jobz, char uplo, lapack_int n, std::complex<double> *a, lapack_int lda, double *w)
{
    return LAPACKE_zheev(matrix_order, jobz, uplo, n, (lapack_complex_double *)a, lda, w);
}

    template <typename T, typename T_b>
static ghost_error ghost_svqb_tmpl (ghost_densemat * v_ot , ghost_densemat * v)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_SOLVER);
    ghost_error ret = GHOST_SUCCESS;
    T one = 1.0;
    T zero = 0.0;
    ghost_lidx n_set_rand = 0;
    ghost_lidx *set_rand = NULL;
    ghost_lidx i,j;
    ghost_lidx n = v->traits.ncols;
    ghost_datatype DT = v->traits.datatype;
    ghost_densemat *x = NULL;
    T *  xval = NULL;
    ghost_lidx ldx;
    T_b *eigs = NULL, *D = NULL;
    ghost_densemat_traits xtraits = GHOST_DENSEMAT_TRAITS_INITIALIZER;
    
    GHOST_CALL_GOTO(ghost_malloc((void **)&eigs, n*sizeof(T_b)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&D, n*sizeof(T_b)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&set_rand, n*sizeof(ghost_lidx)),err,ret);
    
    xtraits.flags |= GHOST_DENSEMAT_NO_HALO; 
    xtraits.ncols = n;
    xtraits.nrows = n;
    xtraits.storage = GHOST_DENSEMAT_COLMAJOR;
    xtraits.datatype = DT;
    xtraits.location = GHOST_LOCATION_HOST;
    if (v->traits.location & GHOST_LOCATION_DEVICE) {
        xtraits.location |= GHOST_LOCATION_DEVICE;
    }
    GHOST_CALL_GOTO(ghost_densemat_create(&x,NULL,xtraits),err,ret);
    GHOST_CALL_GOTO(x->fromScalar(x,&zero),err,ret);
    xval = (T *)x->val;
    ldx = x->stride;

    
    //GHOST_CALL_GOTO(ghost_tsmttsm( x, v, v,&one,&zero,GHOST_GEMM_ALL_REDUCE,1),err,ret);
    GHOST_CALL_GOTO(ghost_tsmttsm( x, v, v,&one,&zero,GHOST_GEMM_ALL_REDUCE,1,GHOST_GEMM_KAHAN),err,ret);
    

    x->download(x);
   
    for (i=0;i<n;i++) {
       if( std::real(xval[i*ldx+i]) <  0. ){
           xval[i*ldx+i] = -xval[i*ldx+i];
        }
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
#ifdef GHOST_HAVE_MPI
        ghost_mpi_datatype dt, dt_b;
        ghost_mpi_datatype_get(&dt,DT);
        ghost_mpi_datatype_get(&dt_b,(ghost_datatype)(GHOST_DT_REAL | (DT&(GHOST_DT_FLOAT|GHOST_DT_DOUBLE))));
        MPI_Bcast( xval, ldx*n, dt  , 0, MPI_COMM_WORLD);
        MPI_Bcast( eigs,     n, dt_b, 0, MPI_COMM_WORLD);
#endif
    
    
    for ( i=0;i<n;i++){  
        if( eigs[i] <  0. ){
           eigs[i] = -eigs[i];
        }
        if( eigs[i] <  (T_b)1.e-13*eigs[n-1] ){  // TODO make it for single precision, too
           eigs[i] +=  (T_b)1.e-13*eigs[n-1];
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

#ifdef GHOST_HAVE_MPI   
    MPI_Bcast( &n_set_rand, 1, ghost_mpi_dt_lidx , 0, MPI_COMM_WORLD);
    MPI_Bcast( set_rand,    n, ghost_mpi_dt_lidx , 0, MPI_COMM_WORLD);
#endif
   
   if( n_set_rand > 0 ){
      ghost_densemat * vec_view2rand;
      v_ot->viewScatteredCols(v_ot, &vec_view2rand, n_set_rand, set_rand);
      vec_view2rand->fromRand( vec_view2rand );
      ghost_densemat_destroy(vec_view2rand);
     }  
   
   
    goto out;
err:

out: 
    ghost_densemat_destroy(x);
    free(eigs);
    free(D);
    free(set_rand);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_SOLVER);

    return ret;
}

ghost_error ghost_svqb(ghost_densemat * v_ot , ghost_densemat * v)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);
    
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



    template <typename T, typename T_b>
static ghost_error ghost_svd_deflation_tmpl ( ghost_lidx *svd_offset, ghost_densemat * ot_vec, ghost_densemat * vec, float limit)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_SOLVER);
    ghost_error ret = GHOST_SUCCESS;
    T one = 1.0;
    T zero = 0.0;
    ghost_lidx i,j;
    ghost_lidx n = vec->traits.ncols;
    ghost_datatype DT = vec->traits.datatype;
    ghost_densemat *x = NULL;
    T *  xval = NULL;
    ghost_lidx ldx;
    ghost_densemat_traits xtraits = GHOST_DENSEMAT_TRAITS_INITIALIZER;
    T_b * eigs = NULL;
    
    GHOST_CALL_GOTO(ghost_malloc((void **)&eigs, n*sizeof(T_b)),err,ret);
    
    xtraits.flags |= GHOST_DENSEMAT_NO_HALO; 
    xtraits.ncols = n;
    xtraits.nrows = n;
    xtraits.storage = GHOST_DENSEMAT_COLMAJOR;
    xtraits.location = GHOST_LOCATION_HOST;
    if (ot_vec->traits.location & GHOST_LOCATION_DEVICE) {
        xtraits.location |= GHOST_LOCATION_DEVICE;
    }
    xtraits.datatype = DT;
    GHOST_CALL_GOTO(ghost_densemat_create(&x,NULL,xtraits),err,ret);
    GHOST_CALL_GOTO(x->fromScalar(x,&zero),err,ret);
    xval = (T *)x->val;
    ldx = x->stride;
    
    
    //GHOST_CALL_GOTO(ghost_tsmttsm( x, vec, vec,&one,&zero,GHOST_GEMM_ALL_REDUCE,1),err,ret);
    GHOST_CALL_GOTO(ghost_tsmttsm( x, vec, vec,&one,&zero,GHOST_GEMM_ALL_REDUCE,1,GHOST_GEMM_KAHAN),err,ret);
    
    x->download(x);

    if (call_eig_function<T,T_b>( LAPACK_COL_MAJOR, 'V' , 'U', n, xval, ldx, eigs)) {
        ERROR_LOG("LAPACK eigenvalue function failed!");
        ret = GHOST_ERR_LAPACK;
        goto err;
    }

#ifdef GHOST_HAVE_MPI
        ghost_mpi_datatype dt, dt_b;
        ghost_mpi_datatype_get(&dt,DT);
        ghost_mpi_datatype_get(&dt_b,(ghost_datatype)(GHOST_DT_REAL | (DT&(GHOST_DT_FLOAT|GHOST_DT_DOUBLE))));
        MPI_Bcast( xval, ldx*n, dt  , 0, MPI_COMM_WORLD);
        MPI_Bcast( eigs,     n, dt_b, 0, MPI_COMM_WORLD);
#endif
    
    
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
    
#ifdef GHOST_HAVE_MPI 
    MPI_Bcast( svd_offset, 1, ghost_mpi_dt_lidx , 0, MPI_COMM_WORLD);
#endif

    x->upload(x);
    GHOST_CALL_GOTO(ghost_tsmm( ot_vec, vec, x, &one, &zero),err,ret);
    
    goto out;
err:

out: 
    ghost_densemat_destroy(x);
    free(eigs);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_SOLVER);
    
    return ret;
}

ghost_error ghost_svd_deflation( ghost_lidx *svd_offset, ghost_densemat * ot_vec, ghost_densemat * vec, float limit)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH); 
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH); 
    
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
ghost_error ghost_svqb(ghost_densemat * v_ot , ghost_densemat * v)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH); 
    
    UNUSED(v_ot);
    UNUSED(v);
    ERROR_LOG("LAPACKE not found!");
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH); 
    return GHOST_ERR_NOT_IMPLEMENTED;
}

ghost_error ghost_svd_deflation( ghost_lidx *svd_offset, ghost_densemat * ot_vec, ghost_densemat * vec, float limit)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH); 
    
    UNUSED(svd_offset);
    UNUSED(ot_vec);
    UNUSED(vec);
    UNUSED(limit);   
    ERROR_LOG("LAPACKE not found!");
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH); 
    return GHOST_ERR_NOT_IMPLEMENTED;
}
#endif
