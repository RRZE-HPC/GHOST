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
    GHOST_SPMV_DEFAULT        = 0,
    GHOST_SPMV_AXPY           = 1<<0,
    GHOST_SPMV_MODE_NOCOMM    = 1<<1,
    GHOST_SPMV_BARRIER        = 1<<2,
    GHOST_SPMV_MODE_OVERLAP   = 1<<3,
    GHOST_SPMV_MODE_TASK      = 1<<4,
    GHOST_SPMV_SHIFT          = 1<<5,
    GHOST_SPMV_SCALE          = 1<<6,
    GHOST_SPMV_AXPBY          = 1<<7,
    GHOST_SPMV_DOT_YY         = 1<<8,
    GHOST_SPMV_DOT_XY         = 1<<9,
    GHOST_SPMV_DOT_XX         = 1<<10,
    GHOST_SPMV_NOT_REDUCE     = 1<<11,
    GHOST_SPMV_LOCAL          = 1<<12,
    GHOST_SPMV_REMOTE         = 1<<13,
    GHOST_SPMV_VSHIFT         = 1<<14,
    GHOST_SPMV_CHAIN_AXPBY    = 1<<15,
    GHOST_SPMV_MODE_PIPELINED = 1<<16,
} ghost_spmv_flags;


#define GHOST_SPMV_DOT (GHOST_SPMV_DOT_YY|GHOST_SPMV_DOT_XY|\
        GHOST_SPMV_DOT_XX)

/**
 * @brief All flags which case an SpMV augmentation.
 */
#define GHOST_SPMV_AUG_FLAGS (GHOST_SPMV_SHIFT|GHOST_SPMV_VSHIFT|\
        GHOST_SPMV_SCALE|GHOST_SPMV_AXPY|GHOST_SPMV_AXPBY|\
        GHOST_SPMV_DOT|GHOST_SPMV_CHAIN_AXPBY)

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
#define GHOST_SPMV_PARSE_TRAITS(traits,_alpha,_beta,_gamma,_dot,_z,_delta,_eta,dt_in,dt_out){\
    dt_in *arg = NULL;\
    if (traits.flags & GHOST_SPMV_SCALE) {\
        arg = (dt_in *)traits.alpha;\
        if (!arg) {\
            ERROR_LOG("Scale argument is NULL!");\
            return GHOST_ERR_INVALID_ARG;\
        }\
        _alpha = *(dt_out *)arg;\
    }\
    if (traits.flags & GHOST_SPMV_AXPBY) {\
        arg = (dt_in *)traits.beta;\
        if (!arg) {\
            ERROR_LOG("AXPBY argument is NULL!");\
            return GHOST_ERR_INVALID_ARG;\
        }\
        _beta = *(dt_out *)arg;\
    }\
    if (traits.flags & (GHOST_SPMV_SHIFT | GHOST_SPMV_VSHIFT)) {\
        arg = (dt_in *)traits.gamma;\
        if (!arg) {\
            ERROR_LOG("Shift argument is NULL!");\
            return GHOST_ERR_INVALID_ARG;\
        }\
        _gamma = (dt_out *)arg;\
    }\
    if (traits.flags & GHOST_SPMV_DOT) {\
        arg = (dt_in *)traits.dot;\
        if (!arg) {\
            ERROR_LOG("Dot argument is NULL!");\
            return GHOST_ERR_INVALID_ARG;\
        }\
        _dot = arg;\
    }\
    if (traits.flags & GHOST_SPMV_CHAIN_AXPBY) {\
        ghost_densemat *zarg;\
        zarg = (ghost_densemat *)traits.z;\
        if (!zarg) {\
            ERROR_LOG("z argument is NULL!");\
            return GHOST_ERR_INVALID_ARG;\
        }\
        _z = zarg;\
        arg = (dt_in *)traits.delta;\
        if (!arg) {\
            ERROR_LOG("delta argument is NULL!");\
            return GHOST_ERR_INVALID_ARG;\
        }\
        _delta = *(dt_out *)arg;\
        arg = (dt_in *)traits.eta;\
        if (!arg) {\
            ERROR_LOG("eta argument is NULL!");\
            return GHOST_ERR_INVALID_ARG;\
        }\
        _eta = *(dt_out *)arg;\
    }\
    if (traits.flags & GHOST_SPMV_REMOTE) {\
        traits.flags = (ghost_spmv_flags)(traits.flags & ~GHOST_SPMV_AXPBY);\
        traits.flags = (ghost_spmv_flags)(traits.flags & ~GHOST_SPMV_SHIFT);\
        traits.flags = (ghost_spmv_flags)(traits.flags & ~GHOST_SPMV_VSHIFT);\
        traits.flags = (ghost_spmv_flags)(traits.flags | GHOST_SPMV_AXPY);\
    } else if (traits.flags & GHOST_SPMV_LOCAL) {\
        traits.flags = (ghost_spmv_flags)(traits.flags & ~GHOST_SPMV_DOT);\
        traits.flags = (ghost_spmv_flags)(traits.flags & ~GHOST_SPMV_CHAIN_AXPBY);\
    }\
}\

/**
 * @brief SpMV solver which do combined computation.
 */
#define GHOST_SPMV_MODES_FULL     (GHOST_SPMV_MODE_NOMPI | GHOST_SPMV_MODE_VECTOR | GHOST_SPMV_MODE_PIPELINED)
/**
 * @brief SpMV solvers which do split computation.
 */
#define GHOST_SPMV_MODES_SPLIT    (GHOST_SPMV_MODE_OVERLAP | GHOST_SPMV_MODE_TASK)
/**
 * @brief All SpMV solver modes.
 */
#define GHOST_SPMV_MODES_ALL      (GHOST_SPMV_MODES_FULL | GHOST_SPMV_MODES_SPLIT)

#define GHOST_SPMV_MODES_MPI (GHOST_SPMV_MODE_VECTOR | GHOST_SPMV_MODES_SPLIT | GHOST_SPMV_MODE_PIPELINED)

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
