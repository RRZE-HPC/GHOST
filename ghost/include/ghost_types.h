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


#ifdef LONGIDX
typedef int64_t ghost_midx_t; // type for the index of the matrix
typedef int64_t ghost_mnnz_t; // type for the number of nonzeros in the matrix
typedef int64_t ghost_vidx_t; // type for the index of the vector
typedef long long int ghost_blas_idx_t;


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
typedef int32_t ghost_vidx_t; // type for the index of the vector
typedef int ghost_blas_idx_t;


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
