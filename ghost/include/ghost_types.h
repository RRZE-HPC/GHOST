#ifndef __GHOST_TYPES_H__
#define __GHOST_TYPES_H__

#ifdef CUDAKERNEL
#undef GHOST_MPI // TODO
#endif

#ifdef GHOST_MPI
#include <mpi.h>
#endif

#ifdef OPENCL
#include <CL/cl.h>
#endif
#ifdef CUDA
#include <cuComplex.h>
#endif

#include <stdint.h>


/*#ifdef GHOST_DT_S
#define ABS(a) fabsf(a)
#define REAL(a) a
#define IMAG(a) 0.0
#define SQRT(a) sqrtf(a)
typedef float ghost_dt;
typedef float ghost_dt_el;
#ifdef OPENCL
typedef float ghost_cl_dt;
#endif
#ifdef CUDA
typedef float ghost_cu_dt;
#define CUREAL(a) a
#define CUIMAG(a) 0 
#endif
#ifdef GHOST_MPI
#define ghost_mpi_dt MPI_FLOAT
#define ghost_mpi_sum MPI_SUM
#endif
#define GHOST_MY_DT (GHOST_BINCRS_DT_FLOAT | GHOST_BINCRS_DT_REAL)
#endif

#ifdef GHOST_DT_D
#define ABS(a) fabs(a)
#define REAL(a) a
#define IMAG(a) 0.0
#define SQRT(a) sqrt(a)
typedef double ghost_dt;
typedef double ghost_dt_el;
#ifdef OPENCL
typedef double ghost_cl_dt;
#endif
#ifdef CUDA
typedef double ghost_cu_dt;
#define CUREAL(a) a
#define CUIMAG(a) 0 
#endif
#ifdef GHOST_MPI
#define ghost_mpi_dt MPI_DOUBLE
#define ghost_mpi_sum MPI_SUM
#endif
#define GHOST_MY_DT (GHOST_BINCRS_DT_DOUBLE | GHOST_BINCRS_DT_REAL)
#endif

#ifdef GHOST_DT_C
#define ABS(a) cabsf(a)
#define REAL(a) crealf(a)
#define IMAG(a) cimagf(a)
#define SQRT(a) csqrtf(a)
typedef _Complex float ghost_dt;
typedef float ghost_dt_el;
#ifdef OPENCL
typedef float2 ghost_cl_dt;
#endif
#ifdef CUDA
typedef cuFloatComplex ghost_cu_dt;
#define CUREAL(a) cuCrealf(a)
#define CUIMAG(a) cuCimagf(a)
#endif
#ifdef GHOST_MPI
#define ghost_mpi_dt GHOST_MPI_DT_C
#define ghost_mpi_sum GHOST_MPI_OP_SUM_C
#endif
#define GHOST_MY_DT (GHOST_BINCRS_DT_FLOAT | GHOST_BINCRS_DT_COMPLEX)
#endif

#ifdef GHOST_DT_Z
#define ABS(a) cabs(a)
#define REAL(a) creal(a)
#define IMAG(a) cimag(a)
#define SQRT(a) csqrt(a)
typedef _Complex double ghost_dt;
typedef double ghost_dt_el;
#ifdef OPENCL
typedef double2 ghost_cl_dt;
#endif
#ifdef CUDA
typedef cuDoubleComplex ghost_cu_dt;
#define CUREAL(a) cuCreal(a)
#define CUIMAG(a) cuCimag(a)
#endif
#ifdef GHOST_MPI
#define ghost_mpi_dt GHOST_MPI_DT_Z
#define ghost_mpi_sum GHOST_MPI_OP_SUM_Z
#endif
#define GHOST_MY_DT (GHOST_BINCRS_DT_DOUBLE | GHOST_BINCRS_DT_COMPLEX)
#endif

// TODO
#if defined(GHOST_MAT_COMPLEX) || defined(GHOST_VEC_COMPLEX)
#define FLOPS_PER_ENTRY 8.0
#else
#define FLOPS_PER_ENTRY 2.0
#endif*/


// TODO adjust

/*#ifdef GHOST_VEC_DP
#define EPSILON 1e-7
#endif
#ifdef GHOST_VEC_SP
#define EPSILON 1e-5 // TODO
#endif
#define MEQUALS(a,b) (MABS(MREAL(a)-MREAL(b))<EPSILON && MABS(MIMAG(a)-MIMAG(b))<EPSILON)*/


#ifdef LONGIDX
typedef int64_t ghost_midx_t; // type for the index of the matrix
typedef int64_t ghost_mnnz_t; // type for the number of nonzeros in the matrix

typedef ghost_midx_t ghost_vidx_t; // type for the index of the vector

#define ghost_mpi_dt_midx MPI_LONG_LONG
#define ghost_mpi_dt_mnnz MPI_LONG_LONG

#ifdef OPENCL
typedef cl_long ghost_cl_midx_t;
typedef cl_long ghost_cl_mnnz_t;
#endif

#define PRmatNNZ PRId64
#define PRmatIDX PRId64
#define PRvecIDX PRId64

#else

typedef int32_t ghost_midx_t; // type for the index of the matrix
typedef int32_t ghost_mnnz_t; // type for the number of nonzeros in the matrix

typedef ghost_midx_t ghost_vidx_t; // type for the index of the vector

#define ghost_mpi_dt_midx MPI_INT
#define ghost_mpi_dt_mnnz MPI_INT

#ifdef OPENCL
typedef cl_int ghost_cl_midx_t;
typedef cl_int ghost_cl_mnnz_t;
#endif

#define PRmatNNZ PRId32
#define PRmatIDX PRId32
#define PRvecIDX PRId32

#endif


#endif
