#ifndef GHOST_TYPES_H
#define GHOST_TYPES_H

#include "error.h"

#ifdef GHOST_HAVE_MPI
#ifdef __INTEL_COMPILER
#pragma warning (disable : 869)
#pragma warning (disable : 424)
#endif
#include <mpi.h>
typedef MPI_Comm ghost_mpi_comm_t;
#ifdef __INTEL_COMPILER
#pragma warning (enable : 424)
#pragma warning (enable : 869)
#endif
#else
typedef int ghost_mpi_comm_t;
#define MPI_COMM_WORLD 0
//typedef int MPI_Comm;
//#define MPI_COMM_WORLD 0 // TODO unschoen
#endif

#include <inttypes.h>
#include <stdint.h>
#include <sys/types.h>

#define GHOST_DT_FLOAT   (1)
#define GHOST_DT_DOUBLE  (2)
#define GHOST_DT_REAL    (4)
#define GHOST_DT_COMPLEX (8)
#define GHOST_DT_S_IDX 0
#define GHOST_DT_D_IDX 1
#define GHOST_DT_C_IDX 2
#define GHOST_DT_Z_IDX 3

#if GHOST_HAVE_LONGIDX
typedef int64_t ghost_midx_t; // type for the index of the matrix
typedef int64_t ghost_mnnz_t; // type for the number of nonzeros in the matrix
typedef int64_t ghost_vidx_t; // type for the index of the vector
typedef long long int ghost_blas_idx_t;

#define ghost_mpi_dt_midx MPI_LONG_LONG
#define ghost_mpi_dt_mnnz MPI_LONG_LONG

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

#define PRmatNNZ PRId32
#define PRmatIDX PRId32
#define PRvecIDX PRId32

#endif




typedef enum {
    GHOST_SPMVM_DEFAULT = 0,
    GHOST_SPMVM_AXPY = 1,
    GHOST_SPMVM_MODE_NOMPI = 2,
    GHOST_SPMVM_MODE_VECTORMODE = 4,
    GHOST_SPMVM_MODE_GOODFAITH = 8,
    GHOST_SPMVM_MODE_TASKMODE = 16,
    GHOST_SPMVM_APPLY_SHIFT = 32,
    GHOST_SPMVM_APPLY_SCALE = 64,
    GHOST_SPMVM_AXPBY = 128,
    GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT = 256
} ghost_spmv_flags_t;

#define GHOST_SPMVM_MODES_FULL     (GHOST_SPMVM_MODE_NOMPI | GHOST_SPMVM_MODE_VECTORMODE)
#define GHOST_SPMVM_MODES_SPLIT    (GHOST_SPMVM_MODE_GOODFAITH | GHOST_SPMVM_MODE_TASKMODE)
#define GHOST_SPMVM_MODES_ALL      (GHOST_SPMVM_MODES_FULL | GHOST_SPMVM_MODES_SPLIT)


typedef enum {
    GHOST_SPMFROMROWFUNC_DEFAULT = 0
} ghost_spmFromRowFunc_flags_t;

//typedef struct ghost_comm_t ghost_comm_t;
typedef struct ghost_acc_info_t ghost_acc_info_t;
typedef struct ghost_matfile_header_t ghost_matfile_header_t;
typedef struct ghost_mpi_c ghost_mpi_c;
typedef struct ghost_mpi_z ghost_mpi_z;
typedef void (*ghost_spmFromRowFunc_t)(ghost_midx_t, ghost_midx_t *, ghost_midx_t *, void *);





/*struct ghost_comm_t
{
};*/

struct ghost_acc_info_t
{
    int nDistinctDevices;
    int *nDevices;
    char **names;
};

struct ghost_mpi_c
{
    float x;
    float y;
};

struct ghost_mpi_z
{
    double x;
    double y;
};

struct ghost_matfile_header_t
{
    int32_t endianess;
    int32_t version;
    int32_t base;
    int32_t symmetry;
    int32_t datatype;
    int64_t nrows;
    int64_t ncols;
    int64_t nnz;
};

#endif
