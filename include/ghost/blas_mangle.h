/**
 * @file blas_mangle.h
 * @brief Macros to "de-mangle" BLAS routines and includes and allow generic calls to them in the code.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_BLAS_MANGLE_H
#define GHOST_BLAS_MANGLE_H

#include <strings.h>

#ifdef GHOST_HAVE_MKL
#include <mkl_cblas.h>
#elif defined(GHOST_HAVE_GSL)
#include <gsl_cblas.h>
#else
#include <cblas.h>
#endif

#define blas_order(order) order==GHOST_DENSEMAT_COLMAJOR?CblasColMajor:CblasRowMajor
#define blas_trans(trans) trans[0]=='N'?CblasNoTrans:(trans[0]=='C'?CblasConjTrans:CblasTrans)

#define sgemm(order,transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc) \
    cblas_sgemm(blas_order(order),blas_trans(transa),blas_trans(transb),*m,*n,*k,*alpha,a,*lda,b,*ldb,*beta,c,*ldc)
#define dgemm(order,transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc) \
    cblas_dgemm(blas_order(order),blas_trans(transa),blas_trans(transb),*m,*n,*k,*alpha,a,*lda,b,*ldb,*beta,c,*ldc)
#define cgemm(order,transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc) \
    cblas_cgemm(blas_order(order),blas_trans(transa),blas_trans(transb),*m,*n,*k,alpha,a,*lda,b,*ldb,beta,c,*ldc)
#define zgemm(order,transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc) \
    cblas_zgemm(blas_order(order),blas_trans(transa),blas_trans(transb),*m,*n,*k,alpha,a,*lda,b,*ldb,beta,c,*ldc)

#endif
