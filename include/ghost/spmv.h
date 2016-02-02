/**
 * @file spmv.h
 * @brief Types, constants and macros for SpMV.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_SPMV_H
#define GHOST_SPMV_H

#include "sparsemat.h"
#include "densemat.h"

/**
 * @brief Flags to be passed to sparse matrix-vector multiplication.
 */
typedef enum {
    GHOST_SPMV_DEFAULT = 0,
    GHOST_SPMV_AXPY = 1,
    GHOST_SPMV_MODE_OVERLAP = 8,
    GHOST_SPMV_MODE_TASK = 16,
    GHOST_SPMV_SHIFT = 32,
    GHOST_SPMV_SCALE = 64,
    GHOST_SPMV_AXPBY = 128,
    GHOST_SPMV_DOT = 256,
    GHOST_SPMV_DOT_YY = 512,
    GHOST_SPMV_DOT_XY = 1024,
    GHOST_SPMV_DOT_XX = 2048,
    GHOST_SPMV_NOT_REDUCE = 4096,
    GHOST_SPMV_LOCAL = 8192,
    GHOST_SPMV_REMOTE = 16384,
    GHOST_SPMV_VSHIFT = 32768,
    GHOST_SPMV_CHAIN_AXPBY = 65536
} ghost_spmv_flags;


#define GHOST_SPMV_DOT_ANY (GHOST_SPMV_DOT_YY|GHOST_SPMV_DOT_XY|\
        GHOST_SPMV_DOT_XX)

/**
 * @brief Parse the SPMV arguments.
 *
 * This macro parses the varargs given to an SpMV call and initializes given
 * variables for the SpMV parameters alpha, beta, gamma, and dot.
 * Also, it checks whether the current SpMV works on the local or remote matrix
 * in case of split computation. Depending on that, the flags are manipulated,
 * e.g., to not compute dot products for the local matrix.
 *
 * @param flags The defined flags.
 * @param argp The argument pointer.
 * @param alpha Where to store alpha.
 * @param beta Where to store beta.
 * @param gamma Where to store gamma.
 * @param dot Where to store the dot array.
 * @param z Where to store the z densemat.
 * @param delta Where to store deltea.
 * @param eta Where to store eta.
 * @param dt_in The data type in which the args are present. 
 * @param dt_out The data of which alpha, beta, gamma, and dot.
 *
 * @return 
 */
#define GHOST_SPMV_PARSE_ARGS(flags,argp,alpha,beta,gamma,dot,z,delta,eta,dt_in,dt_out){\
    dt_in *arg = NULL;\
    if (flags & GHOST_SPMV_SCALE) {\
        arg = va_arg(argp,dt_in *);\
        if (!arg) {\
            ERROR_LOG("Scale argument is NULL!");\
            return GHOST_ERR_INVALID_ARG;\
        }\
        alpha = *(dt_out *)arg;\
    }\
    if (flags & GHOST_SPMV_AXPBY) {\
        arg = va_arg(argp,dt_in *);\
        if (!arg) {\
            ERROR_LOG("AXPBY argument is NULL!");\
            return GHOST_ERR_INVALID_ARG;\
        }\
        beta = *(dt_out *)arg;\
    }\
    if (flags & (GHOST_SPMV_SHIFT | GHOST_SPMV_VSHIFT)) {\
        arg = va_arg(argp,dt_in *);\
        if (!arg) {\
            ERROR_LOG("Shift argument is NULL!");\
            return GHOST_ERR_INVALID_ARG;\
        }\
        gamma = (dt_out *)arg;\
    }\
    if (flags & GHOST_SPMV_DOT_ANY) {\
        arg = va_arg(argp,dt_in *);\
        if (!arg) {\
            ERROR_LOG("Dot argument is NULL!");\
            return GHOST_ERR_INVALID_ARG;\
        }\
        dot = arg;\
    }\
    if (flags & GHOST_SPMV_CHAIN_AXPBY) {\
        ghost_densemat *zarg;\
        zarg = va_arg(argp,ghost_densemat *);\
        if (!zarg) {\
            ERROR_LOG("z argument is NULL!");\
            return GHOST_ERR_INVALID_ARG;\
        }\
        z = zarg;\
        arg = va_arg(argp,dt_in *);\
        if (!arg) {\
            ERROR_LOG("delta argument is NULL!");\
            return GHOST_ERR_INVALID_ARG;\
        }\
        delta = *(dt_out *)arg;\
        arg = va_arg(argp,dt_in *);\
        if (!arg) {\
            ERROR_LOG("eta argument is NULL!");\
            return GHOST_ERR_INVALID_ARG;\
        }\
        eta = *(dt_out *)arg;\
    }\
    if (flags & GHOST_SPMV_REMOTE) {\
        flags = (ghost_spmv_flags)(flags & ~GHOST_SPMV_AXPBY);\
        flags = (ghost_spmv_flags)(flags & ~GHOST_SPMV_SHIFT);\
        flags = (ghost_spmv_flags)(flags & ~GHOST_SPMV_VSHIFT);\
        flags = (ghost_spmv_flags)(flags | GHOST_SPMV_AXPY);\
    } else if (flags & GHOST_SPMV_LOCAL) {\
        flags = (ghost_spmv_flags)(flags & ~GHOST_SPMV_DOT_ANY);\
        flags = (ghost_spmv_flags)(flags & ~GHOST_SPMV_CHAIN_AXPBY);\
    }\
}\

/**
 * @brief SpMV solver which do combined computation.
 */
#define GHOST_SPMV_MODES_FULL     (GHOST_SPMV_MODE_NOMPI | GHOST_SPMV_MODE_VECTOR)
/**
 * @brief SpMV solvers which do split computation.
 */
#define GHOST_SPMV_MODES_SPLIT    (GHOST_SPMV_MODE_OVERLAP | GHOST_SPMV_MODE_TASK)
/**
 * @brief All SpMV solver modes.
 */
#define GHOST_SPMV_MODES_ALL      (GHOST_SPMV_MODES_FULL | GHOST_SPMV_MODES_SPLIT)

#define GHOST_SPMV_MODES_MPI (GHOST_SPMV_MODE_VECTOR | GHOST_SPMV_MODES_SPLIT)

#ifdef __cplusplus
/**
 * @brief Bitwise OR operator for ghost_spmv_flags.
 *
 * @param a First input.
 * @param b Second input.
 *
 * @return Bitwise OR of the inputs cast to int.
 */
inline ghost_spmv_flags operator|(const ghost_spmv_flags &a, 
        const ghost_spmv_flags &b)
{
    return static_cast<ghost_spmv_flags>(
            static_cast<int>(a) | static_cast<int>(b));
}

inline ghost_spmv_flags operator&(const ghost_spmv_flags &a, 
        const ghost_spmv_flags &b)
{
    return static_cast<ghost_spmv_flags>(
            static_cast<int>(a) & static_cast<int>(b));
}

#endif

#endif
