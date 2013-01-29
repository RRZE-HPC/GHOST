#ifndef __GHOST_TYPES_H__
#define __GHOST_TYPES_H__

#include "ghost_types_gen.h"
#ifdef MPI
#include <mpi.h>
#endif
#ifdef OPENCL
#include <CL/cl.h>
#endif

#ifdef GHOST_MAT_DP
#ifdef GHOST_MAT_COMPLEX
typedef _Complex double ghost_mdat_t;
typedef double ghost_mdat_el_t;
#ifdef OPENCL
typedef double2 ghost_cl_mdat_t;
#endif
#ifdef MPI
MPI_Datatype ghost_mpi_dt_mdat;
MPI_Op ghost_mpi_sum_mdat;
#endif
#define GHOST_MY_MDATATYPE (GHOST_BINCRS_DT_DOUBLE | GHOST_BINCRS_DT_COMPLEX)
#else // GHOST_MAT_COMPLEX
typedef double ghost_mdat_t;
typedef double ghost_mdat_el_t;
#ifdef OPENCL
typedef double ghost_cl_mdat_t;
#endif
#ifdef MPI
#define ghost_mpi_dt_mdat MPI_DOUBLE
#define ghost_mpi_sum_mdat MPI_SUM
#endif
#define GHOST_MY_MDATATYPE (GHOST_BINCRS_DT_DOUBLE | GHOST_BINCRS_DT_REAL)
#endif // GHOST_MAT_COMPLEX
#define PRmatDAT "lg"
#endif // GHOST_MAT_DP

#ifdef GHOST_MAT_SP
#ifdef GHOST_MAT_COMPLEX
typedef _Complex float ghost_mdat_t;
typedef float ghost_mdat_el_t;
#ifdef OPENCL
typedef float2 ghost_cl_mdat_t;
#endif
#ifdef MPI
MPI_Datatype ghost_mpi_dt_mdat;
MPI_Op ghost_mpi_sum_mdat;
#endif
#define GHOST_MY_MDATATYPE (GHOST_BINCRS_DT_FLOAT | GHOST_BINCRS_DT_COMPLEX)
#else // GHOST_MAT_COMPLEX
typedef float ghost_mdat_t;
typedef float ghost_mdat_el_t;
#ifdef OPENCL
typedef float ghost_cl_mdat_t;
#endif
#ifdef MPI
#define ghost_mpi_dt_mdat MPI_FLOAT
#define ghost_mpi_sum_mdat MPI_SUM
#endif
#define GHOST_MY_MDATATYPE (GHOST_BINCRS_DT_FLOAT | GHOST_BINCRS_DT_REAL)
#endif // GHOST_MAT_COMPLEX
#define PRmatDAT "g"
#endif // GHOST_MAT_SP

#ifdef GHOST_VEC_DP
#ifdef GHOST_VEC_COMPLEX
typedef _Complex double ghost_vdat_t;
typedef double ghost_vdat_el_t;
#ifdef OPENCL
typedef double2 ghost_cl_vdat_t;
#endif
#ifdef MPI
MPI_Datatype ghost_mpi_dt_vdat;
MPI_Op ghost_mpi_sum_vdat;
#endif
#define GHOST_MY_VDATATYPE  (GHOST_BINCRS_DT_DOUBLE | GHOST_BINCRS_DT_COMPLEX)
#else // GHOST_VEC_COMPLEX
typedef double ghost_vdat_t;
typedef double ghost_vdat_el_t;
#ifdef OPENCL
typedef double ghost_cl_vdat_t;
#endif
#ifdef MPI
#define ghost_mpi_dt_vdat MPI_DOUBLE
#define ghost_mpi_sum_vdat MPI_SUM
#endif
#define GHOST_MY_VDATATYPE  (GHOST_BINCRS_DT_DOUBLE | GHOST_BINCRS_DT_REAL)
#endif // GHOST_VEC_COMPLEX
#define PRvecDAT "lg"
#endif // GHOST_VEC_DP

#ifdef GHOST_VEC_SP
#ifdef GHOST_VEC_COMPLEX
typedef _Complex float ghost_vdat_t;
typedef float ghost_vdat_el_t;
#ifdef OPENCL
typedef float2 ghost_cl_vdat_t;
#endif
#ifdef MPI
MPI_Datatype ghost_mpi_dt_vdat;
MPI_Op ghost_mpi_sum_vdat;
#endif
#define GHOST_MY_VDATATYPE  (GHOST_BINCRS_DT_FLOAT | GHOST_BINCRS_DT_COMPLEX)
#else // GHOST_VEC_COMPLEX
typedef float ghost_vdat_t;
typedef float ghost_vdat_el_t;
#ifdef OPENCL
typedef float ghost_cl_vdat_t;
#endif
#ifdef MPI
#define ghost_mpi_dt_vdat MPI_FLOAT
#define ghost_mpi_sum_vdat MPI_SUM
#endif
#define GHOST_MY_VDATATYPE  (GHOST_BINCRS_DT_FLOAT | GHOST_BINCRS_DT_REAL)
#endif // GHOST_VEC_COMPLEX
#define PRvecDAT "g"
#endif // GHOST_VEC_SP

#if defined(GHOST_MAT_COMPLEX) || defined(GHOST_VEC_COMPLEX)
#define FLOPS_PER_ENTRY 8.0
#else
#define FLOPS_PER_ENTRY 2.0
#endif

#ifdef GHOST_MAT_DP
#ifdef GHOST_MAT_COMPLEX
#define MABS(a) cabs(a)
#define MREAL(a) creal(a)
#define MIMAG(a) cimag(a)
#define MSQRT(a) csqrt(a)
#else
#define MABS(a) fabs(a)
#define MREAL(a) a
#define MIMAG(a) 0.0
#define MSQRT(a) sqrt(a)
#endif
#endif

#ifdef GHOST_MAT_SP
#ifdef GHOST_MAT_COMPLEX
#define MABS(a) cabsf(a)
#define MREAL(a) crealf(a)
#define MIMAG(a) cimagf(a)
#define MSQRT(a) csqrtf(a)
#else
#define MABS(a) fabsf(a)
#define MREAL(a) a
#define MIMAG(a) 0.0
#define MSQRT(a) sqrtf(a)
#endif
#endif

#ifdef GHOST_VEC_DP
#ifdef GHOST_VEC_COMPLEX
#define VABS(a) cabs(a)
#define VREAL(a) creal(a)
#define VIMAG(a) cimag(a)
#define VSQRT(a) csqrt(a)
#else
#define VABS(a) fabs(a)
#define VREAL(a) a
#define VIMAG(a) 0.0
#define VSQRT(a) sqrt(a)
#endif
#endif

#ifdef GHOST_VEC_SP
#ifdef GHOST_VEC_COMPLEX
#define VABS(a) cabsf(a)
#define VREAL(a) crealf(a)
#define VIMAG(a) cimagf(a)
#define VSQRT(a) csqrtf(a)
#else
#define VABS(a) fabsf(a)
#define VREAL(a) a
#define VIMAG(a) 0.0
#define VSQRT(a) sqrtf(a)
#endif
#endif

#ifndef MIN
#define MIN(x,y) ((x)<(y)?(x):(y))
#endif
#ifndef MAX
#define MAX(x,y) ((x)<(y)?(y):(x))
#endif

// TODO adjust

#ifdef GHOST_VEC_DP
#define EPSILON 1e-7
#endif
#ifdef GHOST_VEC_SP
#define EPSILON 1e-5 // TODO
#endif
#define MEQUALS(a,b) (MABS(MREAL(a)-MREAL(b))<EPSILON && MABS(MIMAG(a)-MIMAG(b))<EPSILON)


#ifdef GHOST_LONGIDX
typedef long int ghost_midx_t; // type for the index of the matrix
typedef long int ghost_mnnz_t; // type for the number of nonzeros in the matrix

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

typedef int ghost_midx_t; // type for the index of the matrix
typedef int ghost_mnnz_t; // type for the number of nonzeros in the matrix

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
