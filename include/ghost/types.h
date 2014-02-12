/**
 * @file types.h
 * @brief Header file for type definitions. 
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_TYPES_H
#define GHOST_TYPES_H

#include "error.h"

#ifdef GHOST_HAVE_MPI
#include <mpi.h>
typedef MPI_Comm ghost_mpi_comm_t;
typedef MPI_Op ghost_mpi_op_t;
typedef MPI_Datatype ghost_mpi_datatype_t;
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

/**
 * @brief Available primitive data types.
 *
 * The validity of a data type can be checked with ghost_datatypeValid().
 */
typedef enum {
    /**
     * @brief Single precision.
     */
    GHOST_DT_FLOAT = 1,
    /**
     * @brief Double precision.
     */
    GHOST_DT_DOUBLE = 2,
    /**
     * @brief Real numbers.
     */
    GHOST_DT_REAL = 4,
    /**
     * @brief Complex numbers.
     */
    GHOST_DT_COMPLEX = 8
} ghost_datatype_t;

/**
 * @brief Contiguous indices for data types.
 *
 * This is used, e.g., when template instantiations for different data types are stored in an array.
 */
typedef enum {
    /**
     * @brief Real float.
     */
    GHOST_DT_S_IDX = 0,
    /**
     * @brief Real double.
     */
    GHOST_DT_D_IDX = 1,
    /**
     * @brief Complex float.
     */
    GHOST_DT_C_IDX = 2,
    /**
     * @brief Complex double.
     */
    GHOST_DT_Z_IDX = 3,
} ghost_datatype_idx_t;

/**
 * @brief Macro to "register" a double data type in an application
 *
 * @param name An identifier.
 *
 * This macro enables easy switching of data types in applications.
 * After calling the macros with identifier "mydata" a typedef "typedef mydata_t double;"
 * and a variable "ghost_datatype_t mydata = GHOST_DT_DOUBLE|GHOST_DT_REAL;" will be present.
 */
#define GHOST_REGISTER_DT_D(name) \
    typedef double name ## _t; \
ghost_datatype_t name = GHOST_DT_DOUBLE|GHOST_DT_REAL; \

/**
 * @see GHOST_REGISTER_DT_D with float instead of double.
 */
#define GHOST_REGISTER_DT_S(name) \
    typedef float name ## _t; \
ghost_datatype_t name = GHOST_DT_FLOAT|GHOST_DT_REAL; \

/**
 * @see GHOST_REGISTER_DT_D with float complex instead of double.
 */
#define GHOST_REGISTER_DT_C(name) \
    typedef complex float name ## _t; \
ghost_datatype_t name = GHOST_DT_FLOAT|GHOST_DT_COMPLEX; \

/**
 * @see GHOST_REGISTER_DT_D with double complex instead of double.
 */
#define GHOST_REGISTER_DT_Z(name) \
    typedef complex double name ## _t; \
ghost_datatype_t name = GHOST_DT_DOUBLE|GHOST_DT_COMPLEX; \

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
 * @brief Type for row/column indices of a dense matrix/vector
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
 * @brief Macro to print matrix nonzero indices depending on index size
 */
#define PRmatNNZ PRId64
/**
 * @brief Macro to print matrix row/column indices depending on index size
 */
#define PRmatIDX PRId64
/**
 * @brief Macro to print vector indices depending on index size
 */
#define PRvecIDX PRId64

#else

typedef int32_t ghost_midx_t;
typedef int32_t ghost_mnnz_t; 
typedef int32_t ghost_vidx_t;
typedef int ghost_blas_idx_t;

#define ghost_mpi_dt_midx MPI_INT
#define ghost_mpi_dt_mnnz MPI_INT

#define PRmatNNZ PRId32
#define PRmatIDX PRId32
#define PRvecIDX PRId32

#endif

typedef struct ghost_mpi_c ghost_mpi_c;
typedef struct ghost_mpi_z ghost_mpi_z;


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

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @brief Check whether a given data type is valid. 
     *
     * @param datatype The data type.
     *
     * @return 1 if the data type is valid and 0 if it isn't.
     *
     * An data type is valid if exactly one of GHOST_DT_FLOAT and GHOST_DT_DOUBLE and 
     * exactly one of GHOST_DT_REAL and GHOST_DT_COMPLEX is set.
     */
    char ghost_datatypeValid(ghost_datatype_t datatype);
    /**
     * @brief Stringify a ghost_datatype_t.
     *
     * @param datatype The data type.
     *
     * @return A string representation of the data type. 
     */
    char * ghost_datatypeString(ghost_datatype_t datatype);
    ghost_error_t ghost_datatypeIdx(ghost_datatype_idx_t *idx, ghost_datatype_t datatype);
    ghost_error_t ghost_sizeofDatatype(size_t *size, ghost_datatype_t datatype);
    ghost_error_t ghost_mpi_datatype(ghost_mpi_datatype_t *dt, ghost_datatype_t datatype);
    ghost_error_t ghost_mpi_createDatatypes();
    ghost_error_t ghost_mpi_destroyDatatypes();

#ifdef __cplusplus
}
#endif

#endif
