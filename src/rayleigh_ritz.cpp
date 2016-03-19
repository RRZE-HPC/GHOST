#include "ghost/config.h"
#include "ghost/sparsemat.h"
#include "ghost/spmv.h"
#include "ghost/rayleigh_ritz.h"
#include "ghost/util.h"
#include "ghost/tsmttsm.h"
#include "ghost/tsmm.h"

#include "ghost/types.h"


#include <complex>
#include <cstdlib>

#ifdef GHOST_HAVE_LAPACK
#ifdef GHOST_HAVE_MKL
#include <mkl_lapacke.h>
#else 
#include <lapacke.h>
#endif


template<typename T, typename T_b>
static lapack_int call_geig_function(int matrix_order, char jobz, char uplo, lapack_int n, T *a, lapack_int lda, T *b, lapack_int ldb, T_b *w)
{
    UNUSED(matrix_order);
    UNUSED(jobz);
    UNUSED(uplo);
    UNUSED(n);
    UNUSED(a);
    UNUSED(lda);
    UNUSED(b);
    UNUSED(ldb);
    UNSUED(w);
    ERROR_LOG("This should not be called!");
    return -999;
}

template<>
lapack_int call_geig_function<double,double>(int matrix_order, char jobz, char uplo, lapack_int n, double *a, lapack_int lda, double *b, lapack_int ldb, double *w)
{
  if ( b )
    return LAPACKE_dsygv( matrix_order, 1, jobz, uplo, n, a, lda, b, ldb, w);
  else
    return LAPACKE_dsyev( matrix_order,    jobz, uplo, n, a, lda,         w);
}

template<>
lapack_int call_geig_function<float,float>(int matrix_order, char jobz, char uplo, lapack_int n, float *a, lapack_int lda, float *b, lapack_int ldb, float *w)
{
  if ( b )
    return LAPACKE_ssygv( matrix_order, 1, jobz, uplo, n, a, lda, b, ldb, w);
  else
    return LAPACKE_ssyev( matrix_order,    jobz, uplo, n, a, lda,         w);

}

template<>
lapack_int call_geig_function<std::complex<float>,float>(int matrix_order, char jobz, char uplo, lapack_int n, std::complex<float> *a, lapack_int lda, std::complex<float> *b, lapack_int ldb, float *w)
{
  if ( b )
    return LAPACKE_chegv( matrix_order, 1, jobz, uplo, n, (lapack_complex_float *)a, lda, (lapack_complex_float *)b, ldb, w);
  else
    return LAPACKE_cheev( matrix_order,    jobz, uplo, n, (lapack_complex_float *)a, lda,                                 w);

}

template<>
lapack_int call_geig_function<std::complex<double>,double>(int matrix_order, char jobz, char uplo, lapack_int n, std::complex<double> *a, lapack_int lda, std::complex<double> *b, lapack_int ldb, double *w)
{
  if ( b )
    return LAPACKE_zhegv( matrix_order, 1, jobz, uplo, n, (lapack_complex_double *)a, lda, (lapack_complex_double *)b, ldb, w);
  else
    return LAPACKE_zheev( matrix_order,    jobz, uplo, n, (lapack_complex_double *)a, lda,                                  w);
}

    template <typename T, typename T_b>
static ghost_error ghost_rayleigh_ritz_tmpl (ghost_sparsemat * mat, void * void_eigs, void * void_res,  ghost_densemat * v_eigs , ghost_densemat * v_res, ghost_rayleighritz_flags RR_Obtion, ghost_spmv_flags spMVM_Options)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_SOLVER);
    
    ghost_error ret = GHOST_SUCCESS;
    T one = 1.0;
    T zero = 0.0;
    ghost_lidx i,j;
    ghost_lidx n = v_res->traits.ncols;
    ghost_densemat * view_v_eigs, * view_v_res;
    int n_block;
    ghost_datatype DT = v_res->traits.datatype;
    ghost_densemat *x = NULL, *b = NULL;
    T *  xval = NULL;
    T *  bval = NULL;
    ghost_lidx ldx, ldb;
    T *eigs_T = NULL, *res_T = NULL;
    ghost_densemat_traits xtraits = GHOST_DENSEMAT_TRAITS_INITIALIZER;
    ghost_spmv_opts spmvtraits = GHOST_SPMV_OPTS_INITIALIZER;
    spmvtraits.flags = spMVM_Options;
    
    T_b * eigs = (T_b *)void_eigs;
    T_b * res  = (T_b *)void_res;
    T_b * D;
    
    xtraits.flags |= GHOST_DENSEMAT_NO_HALO; 
    xtraits.ncols = n;
    xtraits.nrows = n;
    xtraits.storage = GHOST_DENSEMAT_COLMAJOR;
    xtraits.datatype = DT;
    xtraits.location = GHOST_LOCATION_HOST;
    if (v_eigs->traits.location & GHOST_LOCATION_DEVICE) {
        xtraits.location |= GHOST_LOCATION_DEVICE;
    }
    GHOST_CALL_GOTO(ghost_densemat_create(&x,NULL,xtraits),err,ret);
    GHOST_CALL_GOTO(x->fromScalar(x,&zero),err,ret);
    xval = (T *)x->val;
    ldx = x->stride;
    
    if( mat ){
    GHOST_CALL_GOTO(ghost_malloc((void **)&eigs_T, n*sizeof(T)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&res_T, 3*n*sizeof(T)),err,ret);

    //spMVM_Options &=  ~GHOST_SPMV_VSHIFT;
    //spMVM_Options &=  ~GHOST_SPMV_SHIFT;
    //spMVM_Options &=  ~GHOST_SPMV_SCALE;
    
    n_block = ghost_get_next_cfg_densemat_dim( n );
    if ( !n_block ) n_block = n;
    for ( i=0; i<n; i+=n_block){
       if( i+n_block > n ) n_block = n - i;
       v_eigs->viewCols( v_eigs, &view_v_eigs, n_block, i);
       v_res->viewCols(  v_res , &view_v_res , n_block, i);
       ghost_spmv( view_v_eigs, mat, view_v_res, spmvtraits);
    }
    
    if( RR_Obtion & GHOST_RAYLEIGHRITZ_KAHAN ){
      GHOST_CALL_GOTO(ghost_tsmttsm( x, v_eigs, v_res,&one,&zero,GHOST_GEMM_ALL_REDUCE,1,GHOST_GEMM_KAHAN  ),err,ret);
    }else{
      GHOST_CALL_GOTO(ghost_tsmttsm( x, v_eigs, v_res,&one,&zero,GHOST_GEMM_ALL_REDUCE,1,GHOST_GEMM_DEFAULT),err,ret);
    }

    if( RR_Obtion & GHOST_RAYLEIGHRITZ_GENERALIZED ){
      GHOST_CALL_GOTO(ghost_densemat_create(&b,NULL,xtraits),err,ret);
      GHOST_CALL_GOTO(b->fromScalar(b,&zero),err,ret);
      bval = (T *)b->val;
      ldb = b->stride;
       if( RR_Obtion & GHOST_RAYLEIGHRITZ_KAHAN ){
          GHOST_CALL_GOTO(ghost_tsmttsm( b, v_res, v_res,&one,&zero,GHOST_GEMM_ALL_REDUCE,1,GHOST_GEMM_KAHAN  ),err,ret);
       }else{
          GHOST_CALL_GOTO(ghost_tsmttsm( b, v_res, v_res,&one,&zero,GHOST_GEMM_ALL_REDUCE,1,GHOST_GEMM_DEFAULT),err,ret);
       }
       b->download(b); 
    }
    }else{
       if( RR_Obtion & GHOST_RAYLEIGHRITZ_KAHAN ) {
          GHOST_CALL_GOTO(ghost_tsmttsm( x, v_res, v_res,&one,&zero,GHOST_GEMM_ALL_REDUCE,1,GHOST_GEMM_KAHAN  ),err,ret);
       } else {
          GHOST_CALL_GOTO(ghost_tsmttsm( x, v_res, v_res,&one,&zero,GHOST_GEMM_ALL_REDUCE,1,GHOST_GEMM_DEFAULT),err,ret);
       }
    GHOST_CALL_GOTO(ghost_malloc((void **)&D,    n*sizeof(T_b)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&eigs, n*sizeof(T_b)),err,ret);
    }
    x->download(x);
    
    if( !mat ){
      for (i=0;i<n;i++) {
        if( std::real(xval[i*ldx+i]) <= 0. ){
            ERROR_LOG("ghost_rayleigh_ritz_tmpl(): zero vector in vectorblock");
            ret = GHOST_ERR_NOT_IMPLEMENTED;
        }
        D[i] = (T_b)1./std::sqrt(std::real(xval[i*ldx+i]));
        for( j=0;j<n;j++) {
            xval[i*ldx+j] *= D[i]*D[j];
        }
      }
    }
    
    if (call_geig_function<T,T_b>( LAPACK_COL_MAJOR, 'V' , 'U', n, xval, ldx, bval, ldb, eigs)) {
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

  if( !mat ){
    for ( i=0;i<n;i++){
      if( eigs[i] <= 0. ){
         ERROR_LOG("ghost_rayleigh_ritz_tmpl(): vector block singular");
         ret = GHOST_ERR_NOT_IMPLEMENTED;
      }
      eigs[i] = (T_b)1./std::sqrt(eigs[i]);
      for( j=0;j<n;j++) {
            xval[i*ldx+j] *= D[j]*eigs[i];
         }
    }
  }
  
    x->upload(x);
    GHOST_CALL_GOTO(ghost_tsmm( v_eigs, v_res, x, &one, &zero),err,ret);

   if( mat ) {
    for ( i=0;i<n;i++) eigs_T[i] = (T)(eigs[i]);

    if ( RR_Obtion & GHOST_RAYLEIGHRITZ_RESIDUAL ){
        spmvtraits.flags = spmvtraits.flags|GHOST_SPMV_VSHIFT;
        spmvtraits.flags = spmvtraits.flags|GHOST_SPMV_DOT_YY;
        spmvtraits.flags = (ghost_spmv_flags)(spmvtraits.flags & ~GHOST_SPMV_NOT_REDUCE);
        
        n_block = ghost_get_next_cfg_densemat_dim( n );
        if ( !n_block ) n_block = n;
        for ( i=0; i<n; i+=n_block){
           if( i+n_block > n ) n_block = n - i;
           v_eigs->viewCols( v_eigs, &view_v_eigs, n_block, i);
           v_res->viewCols(  v_res , &view_v_res , n_block, i);
           spmvtraits.gamma = eigs_T + i;
           spmvtraits.dot   = res_T  + i;
           ghost_spmv( view_v_res, mat, view_v_eigs, spmvtraits);
        }
        for(i=0;i<n;i++) res[i] = std::sqrt(std::real(res_T[i]));
    }
   }
    
    goto out;
err:

out: 
    ghost_densemat_destroy(x);
    if( mat ){
    if( RR_Obtion & GHOST_RAYLEIGHRITZ_GENERALIZED )
      ghost_densemat_destroy(b);
    free(eigs_T);
    free(res_T);
    }else{
    free(D);
    free(eigs);
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_SOLVER);

    return ret;
}


ghost_error ghost_rayleigh_ritz(ghost_sparsemat * mat, void * eigs, void * res,  ghost_densemat * v_eigs , ghost_densemat * v_res, ghost_rayleighritz_flags RR_Obtion, ghost_spmv_flags spMVM_Options)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);
    
    if (v_res->traits.datatype & GHOST_DT_COMPLEX) {
        if (v_res->traits.datatype & GHOST_DT_DOUBLE) {
            return ghost_rayleigh_ritz_tmpl<std::complex<double>, double>( mat, eigs, res,  v_eigs , v_res, RR_Obtion, spMVM_Options);
        } else {
            return ghost_rayleigh_ritz_tmpl<std::complex<float>, float>( mat, eigs, res,  v_eigs , v_res, RR_Obtion, spMVM_Options);
        }
    } else {
        if (v_res->traits.datatype & GHOST_DT_DOUBLE) {
            return ghost_rayleigh_ritz_tmpl<double, double>( mat, eigs, res,  v_eigs , v_res, RR_Obtion, spMVM_Options);
        } else {
            return ghost_rayleigh_ritz_tmpl<float, float>( mat, eigs, res,  v_eigs , v_res, RR_Obtion, spMVM_Options);
        }
    }
}

ghost_error ghost_grayleigh_ritz(ghost_sparsemat * mat, void * eigs, void * res,  ghost_densemat * v_eigs , ghost_densemat * v_res, ghost_rayleighritz_flags RR_Obtion, ghost_spmv_flags spMVM_Options)
{
    PERFWARNING_LOG("ghost_grayleigh_ritz() is obsoleted. Fallback to ghost_rayleigh_ritz() with the flag GHOST_RAYLEIGHRITZ_GENERALIZED");
    return ghost_rayleigh_ritz( mat, eigs, res, v_eigs , v_res, (ghost_rayleighritz_flags)(RR_Obtion|GHOST_RAYLEIGHRITZ_GENERALIZED), spMVM_Options);
}

#else
ghost_error ghost_rayleigh_ritz(ghost_sparsemat * mat, void * eigs, void * res,  ghost_densemat * v_eigs , ghost_densemat * v_res, ghost_rayleighritz_flags RR_Obtion, ghost_spmv_flags spMVM_Options)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    
    UNUSED(mat);
    UNUSED(eigs);
    UNUSED(res);
    UNUSED(v_eigs);
    UNUSED(v_res);
    UNUSED(RR_Obtion);
    UNUSED(spMVM_Options);
    ERROR_LOG("LAPACKE not found!");
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);
    return GHOST_ERR_NOT_IMPLEMENTED;
}

ghost_error ghost_grayleigh_ritz(ghost_sparsemat * mat, void * eigs, void * res,  ghost_densemat * v_eigs , ghost_densemat * v_res, ghost_rayleighritz_flags RR_Obtion, ghost_spmv_flags spMVM_Options)
{
    return ghost_rayleigh_ritz( mat, eigs, res, v_eigs , v_res, RR_Obtion, spMVM_Options);
}
#endif

