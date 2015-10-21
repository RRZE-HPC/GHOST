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
static lapack_int call_eig_function(int matrix_order, char jobz, char uplo, lapack_int n, T *a, lapack_int lda, T_b *w)
{
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

template<typename T, typename T_b>
static lapack_int call_geig_function(int matrix_order, char jobz, char uplo, lapack_int n, T *a, lapack_int lda, T *b, lapack_int ldb, T_b *w)
{
    ERROR_LOG("This should not be called!");
    return -999;
}

template<>
lapack_int call_geig_function<double,double>(int matrix_order, char jobz, char uplo, lapack_int n, double *a, lapack_int lda, double *b, lapack_int ldb, double *w)
{
    return LAPACKE_dsygv( matrix_order, 1, jobz, uplo, n, a, lda, b, ldb, w);
}

template<>
lapack_int call_geig_function<float,float>(int matrix_order, char jobz, char uplo, lapack_int n, float *a, lapack_int lda, float *b, lapack_int ldb, float *w)
{
    return LAPACKE_ssygv( matrix_order, 1, jobz, uplo, n, a, lda, b, ldb, w);
}

template<>
lapack_int call_geig_function<std::complex<float>,float>(int matrix_order, char jobz, char uplo, lapack_int n, std::complex<float> *a, lapack_int lda, std::complex<float> *b, lapack_int ldb, float *w)
{
    return LAPACKE_chegv( matrix_order, 1, jobz, uplo, n, (lapack_complex_float *)a, lda, (lapack_complex_float *)b, ldb, w);
}

template<>
lapack_int call_geig_function<std::complex<double>,double>(int matrix_order, char jobz, char uplo, lapack_int n, std::complex<double> *a, lapack_int lda, std::complex<double> *b, lapack_int ldb, double *w)
{
    return LAPACKE_zhegv( matrix_order, 1, jobz, uplo, n, (lapack_complex_double *)a, lda, (lapack_complex_double *)b, ldb, w);
}

    template <typename T, typename T_b>
static ghost_error_t ghost_rayleigh_ritz_tmpl (ghost_sparsemat_t * mat, void * void_eigs, void * void_res,  ghost_densemat_t * v_eigs , ghost_densemat_t * v_res, int obtion, ghost_spmv_flags_t spMVM_Options)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret = GHOST_SUCCESS;
    T one = 1.0;
    T zero = 0.0;
    ghost_idx_t i;
    ghost_idx_t n = v_res->traits.ncols;
    ghost_datatype_t DT = v_res->traits.datatype;
    ghost_densemat_t *x;
    ghost_idx_t ldx;
    T *eigs_T, *res_T;
    ghost_densemat_traits_t xtraits = GHOST_DENSEMAT_TRAITS_INITIALIZER;
    
    T_b * eigs = (T_b *)void_eigs;
    T_b * res  = (T_b *)void_res;
    
    GHOST_CALL_GOTO(ghost_malloc((void **)&eigs_T, n*sizeof(T)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&res_T, n*sizeof(T)),err,ret);
    
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
    ldx = x->stride;

    T *  xval;
    GHOST_CALL_GOTO(ghost_densemat_valptr(x,(void **)&xval),err,ret);
    
    //spMVM_Options &=  ~GHOST_SPMV_VSHIFT;
    //spMVM_Options &=  ~GHOST_SPMV_SHIFT;
    //spMVM_Options &=  ~GHOST_SPMV_SCALE;
    ghost_spmv( v_eigs, mat, v_res, &spMVM_Options, NULL);
        
    //GHOST_CALL_GOTO(ghost_tsmttsm( x, v_eigs, v_res,&one,&zero,GHOST_GEMM_ALL_REDUCE,1),err,ret);
    GHOST_CALL_GOTO(ghost_tsmttsm( x, v_eigs, v_res,&one,&zero,GHOST_GEMM_ALL_REDUCE,1,GHOST_GEMM_KAHAN),err,ret);
    
    x->download(x); 
    
    if (call_eig_function<T,T_b>( LAPACK_COL_MAJOR, 'V' , 'U', n, xval, ldx, eigs)) {
        ERROR_LOG("LAPACK eigenvalue function failed!");
        ret = GHOST_ERR_LAPACK;
        goto err;
    }

#ifdef GHOST_HAVE_MPI
        ghost_mpi_datatype_t dt, dt_b;
        ghost_mpi_datatype(&dt,DT);
        ghost_mpi_datatype(&dt_b,(ghost_datatype_t)(GHOST_DT_REAL | (DT&(GHOST_DT_FLOAT|GHOST_DT_DOUBLE))));
        MPI_Bcast( xval, ldx*n, dt  , 0, MPI_COMM_WORLD);
        MPI_Bcast( eigs,     n, dt_b, 0, MPI_COMM_WORLD);
#endif

    x->upload(x); 
    GHOST_CALL_GOTO(ghost_tsmm( v_eigs, v_res, x, &one, &zero),err,ret);


    for ( i=0;i<n;i++) eigs_T[i] = (T)(eigs[i]);

    if (obtion){
        spMVM_Options = spMVM_Options|GHOST_SPMV_VSHIFT;
        ghost_spmv( v_res, mat, v_eigs, &spMVM_Options,eigs_T  ,NULL);
        ghost_dot( res_T, v_res, v_res);
        for(i=0;i<n;i++) res[i] = std::sqrt(std::real(res_T[i]));
    }
    
    goto out;
err:

out: 
    x->destroy(x);
    free(eigs_T);
    free(res_T);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
}


ghost_error_t ghost_rayleigh_ritz(ghost_sparsemat_t * mat, void * eigs, void * res,  ghost_densemat_t * v_eigs , ghost_densemat_t * v_res, int obtion, ghost_spmv_flags_t spMVM_Options)
{
    if (v_res->traits.datatype & GHOST_DT_COMPLEX) {
        if (v_res->traits.datatype & GHOST_DT_DOUBLE) {
            return ghost_rayleigh_ritz_tmpl<std::complex<double>, double>( mat, eigs, res,  v_eigs , v_res, obtion, spMVM_Options);
        } else {
            return ghost_rayleigh_ritz_tmpl<std::complex<float>, float>( mat, eigs, res,  v_eigs , v_res, obtion, spMVM_Options);
        }
    } else {
        if (v_res->traits.datatype & GHOST_DT_DOUBLE) {
            return ghost_rayleigh_ritz_tmpl<double, double>( mat, eigs, res,  v_eigs , v_res, obtion, spMVM_Options);
        } else {
            return ghost_rayleigh_ritz_tmpl<float, float>( mat, eigs, res,  v_eigs , v_res, obtion, spMVM_Options);
        }
    }
}

    template <typename T, typename T_b>
static ghost_error_t ghost_grayleigh_ritz_tmpl (ghost_sparsemat_t * mat, void * void_eigs, void * void_res,  ghost_densemat_t * v_eigs , ghost_densemat_t * v_res, int obtion, ghost_spmv_flags_t spMVM_Options)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret = GHOST_SUCCESS;
    T one = 1.0;
    T zero = 0.0;
    ghost_idx_t i;
    ghost_idx_t n = v_res->traits.ncols;
    ghost_datatype_t DT = v_res->traits.datatype;
    ghost_densemat_t *x, *b;
    ghost_idx_t ldx, ldb;
    T *eigs_T, *res_T;
    ghost_densemat_traits_t xtraits = GHOST_DENSEMAT_TRAITS_INITIALIZER;
    
    T_b * eigs = (T_b *)void_eigs;
    T_b * res  = (T_b *)void_res;
    
    GHOST_CALL_GOTO(ghost_malloc((void **)&eigs_T, n*sizeof(T)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&res_T, n*sizeof(T)),err,ret);
    
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
    ldx = x->stride;

    GHOST_CALL_GOTO(ghost_densemat_create(&b,NULL,xtraits),err,ret);
    GHOST_CALL_GOTO(b->fromScalar(b,&zero),err,ret);
    ldb = b->stride;


    T *  xval;
    GHOST_CALL_GOTO(ghost_densemat_valptr(x,(void **)&xval),err,ret);
    T *  bval;
    GHOST_CALL_GOTO(ghost_densemat_valptr(b,(void **)&bval),err,ret);
    
    //spMVM_Options &=  ~GHOST_SPMV_VSHIFT;
    //spMVM_Options &=  ~GHOST_SPMV_SHIFT;
    //spMVM_Options &=  ~GHOST_SPMV_SCALE;
    
    //GHOST_CALL_GOTO(ghost_tsmttsm( b, v_res, v_res,&one,&zero,GHOST_GEMM_ALL_REDUCE,1),err,ret);
    GHOST_CALL_GOTO(ghost_tsmttsm( b, v_res, v_res,&one,&zero,GHOST_GEMM_ALL_REDUCE,1,GHOST_GEMM_KAHAN),err,ret);
    
    ghost_spmv( v_eigs, mat, v_res, &spMVM_Options, NULL);
        
    //GHOST_CALL_GOTO(ghost_tsmttsm( x, v_eigs, v_res,&one,&zero,GHOST_GEMM_ALL_REDUCE,1),err,ret);
    GHOST_CALL_GOTO(ghost_tsmttsm( x, v_eigs, v_res,&one,&zero,GHOST_GEMM_ALL_REDUCE,1,GHOST_GEMM_KAHAN),err,ret);
    
    x->download(x); 
    b->download(b); 

    if (call_geig_function<T,T_b>( LAPACK_COL_MAJOR, 'V' , 'U', n, xval, ldx, bval, ldb, eigs)) {
        ERROR_LOG("LAPACK eigenvalue function failed!");
        ret = GHOST_ERR_LAPACK;
        goto err;
    }
    
#ifdef GHOST_HAVE_MPI
        ghost_mpi_datatype_t dt, dt_b;
        ghost_mpi_datatype(&dt,DT);
        ghost_mpi_datatype(&dt_b,(ghost_datatype_t)(GHOST_DT_REAL | (DT&(GHOST_DT_FLOAT|GHOST_DT_DOUBLE))));
        MPI_Bcast( xval, ldx*n, dt  , 0, MPI_COMM_WORLD);
        MPI_Bcast( eigs,     n, dt_b, 0, MPI_COMM_WORLD);
#endif

    x->upload(x); 
    GHOST_CALL_GOTO(ghost_tsmm( v_eigs, v_res, x, &one, &zero),err,ret);


    for ( i=0;i<n;i++) eigs_T[i] = (T)(eigs[i]);

    if (obtion){
        spMVM_Options = spMVM_Options|GHOST_SPMV_VSHIFT;
        ghost_spmv( v_res, mat, v_eigs, &spMVM_Options,eigs_T  ,NULL);
        ghost_dot( res_T, v_res, v_res);
        for(i=0;i<n;i++) res[i] = std::sqrt(std::real(res_T[i]));
    }
    
    goto out;
err:

out: 
    x->destroy(x);
    b->destroy(b);
    free(eigs_T);
    free(res_T);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
}


ghost_error_t ghost_grayleigh_ritz(ghost_sparsemat_t * mat, void * eigs, void * res,  ghost_densemat_t * v_eigs , ghost_densemat_t * v_res, int obtion, ghost_spmv_flags_t spMVM_Options)
{
    if (v_res->traits.datatype & GHOST_DT_COMPLEX) {
        if (v_res->traits.datatype & GHOST_DT_DOUBLE) {
            return ghost_grayleigh_ritz_tmpl<std::complex<double>, double>( mat, eigs, res,  v_eigs , v_res, obtion, spMVM_Options);
        } else {
            return ghost_grayleigh_ritz_tmpl<std::complex<float>, float>( mat, eigs, res,  v_eigs , v_res, obtion, spMVM_Options);
        }
    } else {
        if (v_res->traits.datatype & GHOST_DT_DOUBLE) {
            return ghost_grayleigh_ritz_tmpl<double, double>( mat, eigs, res,  v_eigs , v_res, obtion, spMVM_Options);
        } else {
            return ghost_grayleigh_ritz_tmpl<float, float>( mat, eigs, res,  v_eigs , v_res, obtion, spMVM_Options);
        }
    }
}

#else
ghost_error_t ghost_rayleigh_ritz(ghost_sparsemat_t * mat, void * eigs, void * res,  ghost_densemat_t * v_eigs , ghost_densemat_t * v_res, int obtion, ghost_spmv_flags_t spMVM_Options)
{
    UNUSED(mat);
    UNUSED(eigs);
    UNUSED(res);
    UNUSED(v_eigs);
    UNUSED(v_res);
    UNUSED(obtion);
    UNUSED(spMVM_Options);
    ERROR_LOG("LAPACKE not found!");
    return GHOST_ERR_NOT_IMPLEMENTED;
}

ghost_error_t ghost_grayleigh_ritz(ghost_sparsemat_t * mat, void * eigs, void * res,  ghost_densemat_t * v_eigs , ghost_densemat_t * v_res, int obtion, ghost_spmv_flags_t spMVM_Options)
{
    UNUSED(mat);
    UNUSED(eigs);
    UNUSED(res);
    UNUSED(v_eigs);
    UNUSED(v_res);
    UNUSED(obtion);
    UNUSED(spMVM_Options);
    ERROR_LOG("LAPACKE not found!");
    return GHOST_ERR_NOT_IMPLEMENTED;
}
#endif

