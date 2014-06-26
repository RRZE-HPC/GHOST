/**
 * @file blas_mangle.h
 * @brief Macros to "de-mangle" BLAS routines and includes and allow generic calls to them in the code.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_BLAS_MANGLE_H
#define GHOST_BLAS_MANGLE_H

#ifdef GHOST_HAVE_MKL
#include <mkl_cblas.h>
#elif defined(GHOST_HAVE_GSL)
#include <gsl_cblas.h>
#else
#include <cblas.h>
#endif

#define SETVARS(order,transa,transb)\
    enum CBLAS_TRANSPOSE blas_transa, blas_transb;\
    enum CBLAS_ORDER blas_order;\
    if (order == GHOST_DENSEMAT_COLMAJOR) {\
        blas_order = CblasColMajor;\
    } else {\
        blas_order = CblasRowMajor;\
    }\
    if (!strncasecmp(transa,"N",1)) {\
        blas_transa = CblasNoTrans; \
    } else if (!strncasecmp(transa,"C",1)) { \
        blas_transa = CblasConjTrans; \
    } else {\
        blas_transa = CblasTrans; \
    }\
    if (!strncasecmp(transb,"N",1)) {\
        blas_transb = CblasNoTrans; \
    } else if (!strncasecmp(transb,"C",1)) { \
        blas_transb = CblasConjTrans; \
    } else {\
        blas_transb = CblasTrans; \
    }\

#define sgemm(order,transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc) \
    SETVARS(order,transa,transb)\
    cblas_sgemm(blas_order,blas_transa,blas_transb,*m,*n,*k,*alpha,a,*lda,b,*ldb,*beta,c,*ldc)
#define dgemm(order,transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc) \
    SETVARS(order,transa,transb)\
    cblas_dgemm(blas_order,blas_transa,blas_transb,*m,*n,*k,*alpha,a,*lda,b,*ldb,*beta,c,*ldc)
#define cgemm(order,transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc) \
    SETVARS(order,transa,transb)\
    cblas_cgemm(blas_order,blas_transa,blas_transb,*m,*n,*k,alpha,a,*lda,b,*ldb,beta,c,*ldc)
#define zgemm(order,transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc) \
    SETVARS(order,transa,transb)\
    cblas_zgemm(blas_order,blas_transa,blas_transb,*m,*n,*k,alpha,a,*lda,b,*ldb,beta,c,*ldc)

#endif
