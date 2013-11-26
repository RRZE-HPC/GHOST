// This file is used to define how blas routines
// are "de-mangled", i.e. how they are called in C.
// This is machine dependent and should be fixed in
// a general way in the ESSR (for instance by CMake).

#ifdef GHOST_HAVE_MKL
#include <mkl.h>
#endif

#ifdef GHOST_HAVE_GSL
#include <gsl_blas.h>
#include <gsl_cblas.h>
#endif

#ifdef GHOST_HAVE_MKL
#define BLAS_MANGLE(name,NAME) name
#define BLAS_Complex8 MKL_Complex8
#define BLAS_Complex16 MKL_Complex16 
#elif defined(GHOST_HAVE_GSL)
#define BLAS_MANGLE(name,NAME) cblas_##name
#define BLAS_Complex8 gsl_complex_float
#define BLAS_Complex16 gsl_complex 
#else
#define BLAS_MANGLE(name,NAME) name ## _
#define BLAS_Complex8 void
#define BLAS_Complex16 void
#endif

// any routines used in ghost should be added here
#if defined(GHOST_HAVE_GSL)
#define sgemm(transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc) \
    enum CBLAS_TRANSPOSE gsl_transa, gsl_transb;\
    if (!strncasecmp(transa,"N",1)) {\
        gsl_transa = CblasNoTrans; \
    } else { \
        gsl_transa = CblasTrans; \
    }\
    if (!strncasecmp(transb,"N",1)) {\
        gsl_transb = CblasNoTrans; \
    } else {\
        gsl_transb = CblasTrans; \
    }\
    BLAS_MANGLE(sgemm,SGEMM)(CblasColMajor,gsl_transa,gsl_transb,*m,*n,*k,*alpha,a,*lda,b,*ldb,*beta,c,*ldc)
#define dgemm(transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc) \
    enum CBLAS_TRANSPOSE gsl_transa, gsl_transb;\
    if (!strncasecmp(transa,"N",1)) {\
        gsl_transa = CblasNoTrans; \
    } else {\
        gsl_transa = CblasTrans; \
    }\
    if (!strncasecmp(transb,"N",1)) {\
        gsl_transb = CblasNoTrans; \
    } else {\
        gsl_transb = CblasTrans; \
    }\
    BLAS_MANGLE(dgemm,DGEMM)(CblasColMajor,gsl_transa,gsl_transb,*m,*n,*k,*alpha,a,*lda,b,*ldb,*beta,c,*ldc)
#define cgemm(transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc) \
    enum CBLAS_TRANSPOSE gsl_transa, gsl_transb;\
    if (!strncasecmp(transa,"N",1)) {\
        gsl_transa = CblasNoTrans; \
    } else {\
        gsl_transa = CblasTrans; \
    }\
    if (!strncasecmp(transb,"N",1)) {\
        gsl_transb = CblasNoTrans; \
    } else {\
        gsl_transb = CblasTrans; \
    }\
    BLAS_MANGLE(cgemm,CGEMM)(CblasColMajor,gsl_transa,gsl_transb,*m,*n,*k,alpha,a,*lda,b,*ldb,beta,c,*ldc)
#define zgemm(transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc) \
    enum CBLAS_TRANSPOSE gsl_transa, gsl_transb;\
    if (!strncasecmp(transa,"N",1)) {\
        gsl_transa = CblasNoTrans; \
    } else {\
        gsl_transa = CblasTrans; \
    }\
    if (!strncasecmp(transb,"N",1)) {\
        gsl_transb = CblasNoTrans; \
    } else {\
        gsl_transb = CblasTrans; \
    }\
    BLAS_MANGLE(zgemm,ZGEMM)(CblasColMajor,gsl_transa,gsl_transb,*m,*n,*k,alpha,a,*lda,b,*ldb,beta,c,*ldc)
#else
#define sgemm(transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc) BLAS_MANGLE(sgemm,SGEMM)(transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc)
#define dgemm(transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc) BLAS_MANGLE(dgemm,DGEMM)(transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc)
#define cgemm(transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc) BLAS_MANGLE(cgemm,CGEMM)(transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc)
#define zgemm(transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc) BLAS_MANGLE(zgemm,ZGEMM)(transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc)
#endif
