#ifndef __GHOST_TYPES_H__
#define __GHOST_TYPES_H__

#include "ghost_types_gen.h"

#ifdef GHOST_MAT_DP
#ifdef GHOST_MAT_COMPLEX
typedef _Complex double ghost_mdat_t;
typedef double2 ghost_cl_mdat_t;
#ifdef MPI
MPI_Datatype MPI_MYDATATYPE;
MPI_Op MPI_MYSUM;
#endif
#define DATATYPE_DESIRED GHOST_DATATYPE_Z
#define SIZEOF_DATATYPE_DESIRED sizeof(_Complex double)
#else // GHOST_MAT_COMPLEX
typedef double ghost_mdat_t;
typedef double ghost_cl_mdat_t;
#ifdef MPI
#define MPI_MYDATATYPE MPI_DOUBLE
#define MPI_MYSUM MPI_SUM
#endif
#define DATATYPE_DESIRED GHOST_DATATYPE_D
#define SIZEOF_DATATYPE_DESIRED sizeof(double)
#endif // GHOST_MAT_COMPLEX
#endif // GHOST_MAT_DP

#ifdef GHOST_MAT_SP
#ifdef GHOST_MAT_COMPLEX
typedef _Complex float ghost_mdat_t;
#ifdef MPI
MPI_Datatype MPI_MYDATATYPE;
MPI_Op MPI_MYSUM;
#endif
#define DATATYPE_DESIRED GHOST_DATATYPE_C
#define SIZEOF_DATATYPE_DESIRED sizeof(_Complex float)
#else // GHOST_MAT_COMPLEX
typedef float ghost_mdat_t;
#ifdef MPI
#define MPI_MYDATATYPE MPI_FLOAT
#define MPI_MYSUM MPI_SUM
#endif
#define DATATYPE_DESIRED GHOST_DATATYPE_S
#define SIZEOF_DATATYPE_DESIRED sizeof(float)
#endif // GHOST_MAT_COMPLEX
#endif // GHOST_MAT_SP

typedef unsigned int ghost_midx_t; // type for the index of the matrix
typedef unsigned int ghost_mnnz_t; // type for the number of nonzeros in the matrix

typedef ghost_midx_t ghost_vidx_t; // type for the index of the vector

#ifdef OPENCL
typedef cl_uint ghost_cl_midx_t;
typedef cl_uint ghost_cl_mnnz_t;
#endif

#define PRmatNNZ PRIu32
#define PRmatIDX PRIu32

#endif
