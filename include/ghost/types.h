/**
 * @file types.h
 * @brief Header file for type definitions. 
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_TYPES_H
#define GHOST_TYPES_H

#include "config.h"

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

#include "error.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/types.h>

#ifndef __cplusplus
#include <complex.h>
#endif

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

#ifdef __cplusplus
inline ghost_datatype_t operator|(const ghost_datatype_t &a, const ghost_datatype_t &b)
{return static_cast<ghost_datatype_t>(static_cast<int>(a) | static_cast<int>(b));}
#endif

typedef enum {
    GHOST_IMPLEMENTATION_AVX,
    GHOST_IMPLEMENTATION_SSE,
    GHOST_IMPLEMENTATION_MIC,
    GHOST_IMPLEMENTATION_PLAIN
} ghost_implementation_t;

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
 * @brief Size of the largest data type (complex double).
 */
#define GHOST_DT_MAX_SIZE 16 

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
ghost_datatype_t name = (ghost_datatype_t)(GHOST_DT_DOUBLE|GHOST_DT_REAL); \

/**
 * @see GHOST_REGISTER_DT_D with float instead of double.
 */
#define GHOST_REGISTER_DT_S(name) \
    typedef float name ## _t; \
ghost_datatype_t name = (ghost_datatype_t)(GHOST_DT_FLOAT|GHOST_DT_REAL); \

#ifdef __cplusplus
/**
 * @see GHOST_REGISTER_DT_D with float complex instead of double.
 */
#define GHOST_REGISTER_DT_C(name) \
    typedef ghost_complex<float> name ## _t; \
ghost_datatype_t name = (ghost_datatype_t)(GHOST_DT_FLOAT|GHOST_DT_COMPLEX);
#else
/**
 * @see GHOST_REGISTER_DT_D with float complex instead of double.
 */
#define GHOST_REGISTER_DT_C(name) \
    typedef complex float name ## _t; \
ghost_datatype_t name = (ghost_datatype_t)(GHOST_DT_FLOAT|GHOST_DT_COMPLEX);
#endif

#ifdef __cplusplus
/**
 * @see GHOST_REGISTER_DT_D with double complex instead of double.
 */
#define GHOST_REGISTER_DT_Z(name) \
    typedef ghost_complex<double> name ## _t; \
ghost_datatype_t name = (ghost_datatype_t)(GHOST_DT_DOUBLE|GHOST_DT_COMPLEX);
#else
/**
 * @see GHOST_REGISTER_DT_D with double complex instead of double.
 */
#define GHOST_REGISTER_DT_Z(name) \
    typedef complex double name ## _t; \
ghost_datatype_t name = (ghost_datatype_t)(GHOST_DT_DOUBLE|GHOST_DT_COMPLEX);
#endif

#ifdef GHOST_HAVE_LONGIDX_GLOBAL

/**
 * @brief Type for global indices.
 */
typedef int64_t ghost_gidx_t; 
/**
 * @brief MPI data type for matrix row/column indices
 */
#define ghost_mpi_dt_gidx MPI_INT64_T
/**
 * @brief Macro to print matrix/vector row/column indices depending on index size
 */
#define PRGIDX PRId64

#define GHOST_GIDX_MAX INT64_MAX

#else

/**
 * @brief Type for global indices.
 */
typedef int32_t ghost_gidx_t; 
/**
 * @brief MPI data type for matrix row/column indices
 */
#define ghost_mpi_dt_gidx MPI_INT32_T
/**
 * @brief Macro to print matrix/vector row/column indices depending on index size
 */
#define PRGIDX PRId32

#define GHOST_GIDX_MAX INT32_MAX

#endif

#ifdef GHOST_HAVE_LONGIDX_LOCAL

/**
 * @brief Type for local indices.
 */
typedef int64_t ghost_lidx_t; 
/**
 * @brief MPI data type for matrix row/column indices
 */
#define ghost_mpi_dt_lidx MPI_INT64_T
/**
 * @brief Macro to print matrix/vector row/column indices depending on index size
 */
#define PRLIDX PRId64

#ifdef GHOST_HAVE_MKL
/**
 * @brief Type for indices used in BLAS calls
 */
typedef long long int ghost_blas_idx_t;
#define PRBLASIDX PRId64
#else
typedef int ghost_blas_idx_t;
#define PRBLASIDX PRId32
#endif

#define PRLIDX PRId64

#define GHOST_LIDX_MAX INT64_MAX

#else

typedef int32_t ghost_lidx_t;
#define ghost_mpi_dt_lidx MPI_INT32_T
typedef int ghost_blas_idx_t;
#define PRBLASIDX PRId32

#define PRLIDX PRId32

#define GHOST_LIDX_MAX INT32_MAX

#endif

#if defined(GHOST_HAVE_LONGIDX_LOCAL) && defined(GHOST_HAVE_LONGIDX_GLOBAL)
#define GHOST_HAVE_UNIFORM_IDX
#endif

#if !defined(GHOST_HAVE_LONGIDX_LOCAL) && !defined(GHOST_HAVE_LONGIDX_GLOBAL)
#define GHOST_HAVE_UNIFORM_IDX
#endif


/**
 * @brief For backward compatibility.
 */
typedef ghost_gidx_t ghost_idx_t;
#define PRIDX PRGIDX
#define ghost_mpi_dt_idx ghost_mpi_dt_gidx



typedef struct ghost_mpi_c ghost_mpi_c;
typedef struct ghost_mpi_z ghost_mpi_z;


/**
 * @brief A float complex number (used for MPI).
 */
struct ghost_mpi_c
{
    float x;
    float y;
};

/**
 * @brief A double complex number (used for MPI).
 */
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
    bool ghost_datatype_valid(ghost_datatype_t datatype);
    /**
     * @ingroup stringification
     *
     * @brief Stringify a ghost_datatype_t.
     *
     * @param datatype The data type.
     *
     * @return A string representation of the data type. 
     */
    char * ghost_datatype_string(ghost_datatype_t datatype);
    ghost_error_t ghost_datatype_idx(ghost_datatype_idx_t *idx, ghost_datatype_t datatype);
    ghost_error_t ghost_idx2datatype(ghost_datatype_t *datatype, ghost_datatype_idx_t idx);
    ghost_error_t ghost_datatype_size(size_t *size, ghost_datatype_t datatype);
    ghost_error_t ghost_mpi_datatype(ghost_mpi_datatype_t *dt, ghost_datatype_t datatype);
    ghost_error_t ghost_mpi_datatypes_create();
    ghost_error_t ghost_mpi_datatypes_destroy();

#ifdef __cplusplus
}
#endif

#endif
