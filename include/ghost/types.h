/**
 * @file types.h
 * @brief Header file for type definitions. 
 */
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
typedef MPI_Op ghost_mpi_op_t;
typedef MPI_Datatype ghost_mpi_datatype_t;
#ifdef __INTEL_COMPILER
#pragma warning (enable : 424)
#pragma warning (enable : 869)
#endif
#else
typedef int ghost_mpi_comm_t;
typedef int ghost_mpi_op_t;
typedef int ghost_mpi_datatype_t;
#define MPI_COMM_NULL 0
#define MPI_OP_NULL 0
#define MPI_DATATYPE_NULL 0
#define MPI_COMM_WORLD 0
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

/**
 * @brief Macro to "register" a double data type in an application
 *
 * @param name An identifier.
 *
 * This macro enables easy switching of data types in applications.
 * After calling the macros with identifier "mydata" a typedef "typedef mydata_t double;"
 * and a variable "int mydata = GHOST_DT_DOUBLE|GHOST_DT_REAL;" will be present.
 */
#define GHOST_REGISTER_DT_D(name) \
    typedef double name ## _t; \
int name = GHOST_DT_DOUBLE|GHOST_DT_REAL; \

/**
 * @see GHOST_REGISTER_DT_D with float instead of double.
 */
#define GHOST_REGISTER_DT_S(name) \
    typedef float name ## _t; \
int name = GHOST_DT_FLOAT|GHOST_DT_REAL; \

/**
 * @see GHOST_REGISTER_DT_D with float complex instead of double.
 */
#define GHOST_REGISTER_DT_C(name) \
    typedef complex float name ## _t; \
int name = GHOST_DT_FLOAT|GHOST_DT_COMPLEX; \

/**
 * @see GHOST_REGISTER_DT_D with double complex instead of double.
 */
#define GHOST_REGISTER_DT_Z(name) \
    typedef complex double name ## _t; \
int name = GHOST_DT_DOUBLE|GHOST_DT_COMPLEX; \

#if GHOST_HAVE_LONGIDX
/**
 * @brief Type for row/column indices of matrix
 */
typedef int64_t ghost_midx_t; 
/**
 * @brief Type for nonzero indices of matrix
 */
typedef int64_t ghost_mnnz_t;
/**
 * @brief Type for row/column indices of a vector
 */
typedef int64_t ghost_vidx_t;
/**
 * @brief Type for indices used in BLAS calls
 */
typedef long long int ghost_blas_idx_t;

/**
 * @brief MPI data type for matrix row/column indices
 */
#define ghost_mpi_dt_midx MPI_LONG_LONG
/**
 * @brief MPI data type for matrix nonzero indices
 */
#define ghost_mpi_dt_mnnz MPI_LONG_LONG

/**
 * @brief Matro to print matrix nonzero indices depending on index size
 */
#define PRmatNNZ PRId64
/**
 * @brief Matro to print matrix row/column indices depending on index size
 */
#define PRmatIDX PRId64
/**
 * @brief Matro to print vector indices depending on index size
 */
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


#endif
