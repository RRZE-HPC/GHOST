/**
 * @file spmv.h
 * @brief Types, constants and macros for SpMV.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_SPMV_H
#define GHOST_SPMV_H

#include "ghost/densemat.h"

/**
 * @brief Flags to be passed to sparse matrix-vector multiplication.
 */
typedef enum {
    GHOST_SPMV_DEFAULT = 0,
    GHOST_SPMV_AXPY = 1,
    GHOST_SPMV_MODE_NOMPI = 2,
    GHOST_SPMV_MODE_VECTOR = 4,
    GHOST_SPMV_MODE_OVERLAP = 8,
    GHOST_SPMV_MODE_TASK = 16,
    GHOST_SPMV_SHIFT = 32,
    GHOST_SPMV_VSHIFT = 4096,
    GHOST_SPMV_SCALE = 64,
    GHOST_SPMV_AXPBY = 128,
    GHOST_SPMV_DOT = 256,
    GHOST_SPMV_NOT_REDUCE = 512,
    GHOST_SPMV_LOCAL = 1024,
    GHOST_SPMV_REMOTE = 2048
} ghost_spmv_flags_t;

#define GHOST_SPMV_PARSE_ARGS(flags,argp,alpha,beta,gamma,dot,dt){\
    dt *arg = NULL;\
    if (flags & GHOST_SPMV_SCALE) {\
        arg = va_arg(argp,dt *);\
        if (!arg) {\
            ERROR_LOG("Scale argument is NULL!");\
            return GHOST_ERR_INVALID_ARG;\
        }\
        alpha = *arg;\
    }\
    if (flags & GHOST_SPMV_AXPBY) {\
        arg = va_arg(argp,dt *);\
        if (!arg) {\
            ERROR_LOG("AXPBY argument is NULL!");\
            return GHOST_ERR_INVALID_ARG;\
        }\
        beta = *arg;\
    }\
    if (flags & (GHOST_SPMV_SHIFT | GHOST_SPMV_VSHIFT)) {\
        arg = va_arg(argp,dt *);\
        if (!arg) {\
            ERROR_LOG("Shift argument is NULL!");\
            return GHOST_ERR_INVALID_ARG;\
        }\
        gamma = arg;\
    }\
    if (flags & GHOST_SPMV_DOT) {\
        arg = va_arg(argp,dt *);\
        if (!arg) {\
            ERROR_LOG("Dot argument is NULL!");\
            return GHOST_ERR_INVALID_ARG;\
        }\
        dot = arg;\
    }\
    if (flags & GHOST_SPMV_REMOTE) {\
        flags = (ghost_spmv_flags_t)(flags & ~GHOST_SPMV_AXPBY);\
        flags = (ghost_spmv_flags_t)(flags & ~GHOST_SPMV_SHIFT);\
        flags = (ghost_spmv_flags_t)(flags & ~GHOST_SPMV_VSHIFT);\
        flags = (ghost_spmv_flags_t)(flags | GHOST_SPMV_AXPY);\
    } else if (flags & GHOST_SPMV_LOCAL) {\
        flags = (ghost_spmv_flags_t)(flags & ~GHOST_SPMV_DOT);\
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
extern "C" {
#endif

ghost_error_t ghost_spmv_haloexchange_initiate(ghost_densemat_t *vec, ghost_permutation_t *permutation, bool assembled);
ghost_error_t ghost_spmv_haloexchange_assemble(ghost_densemat_t *vec, ghost_permutation_t *permutation);
ghost_error_t ghost_spmv_haloexchange_finalize(ghost_densemat_t *vec);
    

#ifdef __cplusplus
} extern "C"
#endif

#endif
