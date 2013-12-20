#ifndef __GHOST_ERROR_H__
#define __GHOST_ERROR_H__

typedef enum ghost_error_t {
    GHOST_SUCCESS,
    GHOST_ERR_INVALID_ARG,
    GHOST_ERR_MPI,
    GHOST_ERR_CUDA,
    GHOST_ERR_UNKNOWN,
    GHOST_ERR_INTERNAL
} ghost_error_t;

#endif
