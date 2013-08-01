// This file is used to define how blas routines
// are "de-mangled", i.e. how they are called in C.
// This is machine dependent and should be fixed in
// a general way in the ESSR (for instance by CMake).
#ifdef MKL_BLAS
#define BLAS_MANGLE(name,NAME) name
#define BLAS_Complex8 MKL_Complex8
#define BLAS_Complex16 MKL_Complex16 
#else
#define BLAS_MANGLE(name,NAME) name ## _
#define BLAS_Complex8 void
#define BLAS_Complex16 void
#endif

// any routines used in ghost should be added here
#define sgemm BLAS_MANGLE(sgemm,SGEMM)
#define dgemm BLAS_MANGLE(dgemm,DGEMM)
#define cgemm BLAS_MANGLE(cgemm,CGEMM)
#define zgemm BLAS_MANGLE(zgemm,ZGEMM)
