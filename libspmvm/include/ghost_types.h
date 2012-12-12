#ifndef __GHOST_TYPES_H__
#define __GHOST_TYPES_H__

#include "ghost_types_gen.h"

#ifdef GHOST_MAT_DP
#ifdef GHOST_MAT_COMPLEX
typedef _Complex double ghost_mdat_t;
typedef double2 ghost_cl_mdat_t;
#ifdef MPI
MPI_Datatype ghost_mpi_dt_mdat;
MPI_Op ghost_mpi_sum_mdat;
#endif
#define GHOST_MY_MDATATYPE GHOST_DATATYPE_Z
#else // GHOST_MAT_COMPLEX
typedef double ghost_mdat_t;
typedef double ghost_cl_mdat_t;
#ifdef MPI
#define ghost_mpi_dt_mdat MPI_DOUBLE
#define ghost_mpi_sum_mdat MPI_SUM
#endif
#define GHOST_MY_MDATATYPE GHOST_DATATYPE_D
#endif // GHOST_MAT_COMPLEX
#endif // GHOST_MAT_DP

#ifdef GHOST_MAT_SP
#ifdef GHOST_MAT_COMPLEX
typedef _Complex float ghost_mdat_t;
#ifdef MPI
MPI_Datatype ghost_mpi_dt_mdat;
MPI_Op ghost_mpi_sum_mdat;
#endif
#define GHOST_MY_MDATATYPE GHOST_DATATYPE_C
#else // GHOST_MAT_COMPLEX
typedef float ghost_mdat_t;
#ifdef MPI
#define ghost_mpi_dt_mdat MPI_FLOAT
#define ghost_mpi_sum_mdat MPI_SUM
#endif
#define GHOST_MY_MDATATYPE GHOST_DATATYPE_S
#endif // GHOST_MAT_COMPLEX
#endif // GHOST_MAT_SP

#ifdef GHOST_VEC_DP
#ifdef GHOST_VEC_COMPLEX
typedef _Complex double ghost_vdat_t;
typedef double2 ghost_cl_vdat_t;
#ifdef MPI
MPI_Datatype ghost_mpi_dt_vdat;
MPI_Op ghost_mpi_sum_vdat;
#endif
#define GHOST_MY_VDATATYPE GHOST_DATATYPE_Z
#else // GHOST_VEC_COMPLEX
typedef double ghost_vdat_t;
typedef double ghost_cl_vdat_t;
#ifdef MPI
#define ghost_mpi_dt_vdat MPI_DOUBLE
#define ghost_mpi_sum_vdat MPI_SUM
#endif
#define GHOST_MY_VDATATYPE GHOST_DATATYPE_D
#endif // GHOST_VEC_COMPLEX
#endif // GHOST_VEC_DP

#ifdef GHOST_VEC_SP
#ifdef GHOST_VEC_COMPLEX
typedef _Complex float ghost_vdat_t;
#ifdef MPI
MPI_Datatype ghost_mpi_dt_vdat;
MPI_Op ghost_mpi_sum_vdat;
#endif
#define GHOST_MY_VDATATYPE GHOST_DATATYPE_C
#else // GHOST_VEC_COMPLEX
typedef float ghost_vdat_t;
#ifdef MPI
#define ghost_mpi_dt_vdat MPI_FLOAT
#define ghost_mpi_sum_vdat MPI_SUM
#endif
#define GHOST_MY_VDATATYPE GHOST_DATATYPE_S
#endif // GHOST_VEC_COMPLEX
#endif // GHOST_VEC_SP


#ifdef GHOST_LONGIDX
typedef long long ghost_midx_t; // type for the index of the matrix
typedef long long ghost_mnnz_t; // type for the number of nonzeros in the matrix

typedef ghost_midx_t ghost_vidx_t; // type for the index of the vector

#ifdef OPENCL
typedef cl_long ghost_cl_midx_t;
typedef cl_long ghost_cl_mnnz_t;
#endif

#define PRmatNNZ PRId64
#define PRmatIDX PRId64

#else

typedef int ghost_midx_t; // type for the index of the matrix
typedef int ghost_mnnz_t; // type for the number of nonzeros in the matrix

typedef ghost_midx_t ghost_vidx_t; // type for the index of the vector

#ifdef OPENCL
typedef cl_int ghost_cl_midx_t;
typedef cl_int ghost_cl_mnnz_t;
#endif

#define PRmatNNZ PRId32
#define PRmatIDX PRId32

#endif


#endif
