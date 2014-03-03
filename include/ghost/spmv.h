#ifndef GHOST_SPMV_H
#define GHOST_SPMV_H

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
    GHOST_SPMV_SCALE = 64,
    GHOST_SPMV_AXPBY = 128,
    GHOST_SPMV_DOT = 256,
    GHOST_SPMV_NOT_REDUCE = 512
} ghost_spmv_flags_t;

#define GHOST_SPMV_PARSE_ARGS(flags,argp,alpha,beta,gamma,dot,dt){\
    dt *arg = NULL;\
    if (flags & GHOST_SPMV_SCALE) {\
        printf("here\n");\
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
    if (flags & GHOST_SPMV_SHIFT) {\
        arg = va_arg(argp,dt *);\
        if (!arg) {\
            ERROR_LOG("Shift argument is NULL!");\
            return GHOST_ERR_INVALID_ARG;\
        }\
        gamma = *arg;\
    }\
    if (flags & GHOST_SPMV_DOT) {\
        arg = va_arg(argp,dt *);\
        if (!arg) {\
            ERROR_LOG("Dot argument is NULL!");\
            return GHOST_ERR_INVALID_ARG;\
        }\
        if (dot) {\
            dot = arg;\
        }\
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

#endif
