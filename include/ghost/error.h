#ifndef __GHOST_ERROR_H__
#define __GHOST_ERROR_H__

typedef enum ghost_error_t {
    GHOST_SUCCESS,
    GHOST_ERR_INVALID_ARG,
    GHOST_ERR_MPI,
    GHOST_ERR_CUDA,
    GHOST_ERR_UNKNOWN,
    GHOST_ERR_INTERNAL,
    GHOST_ERR_IO
} ghost_error_t;

char * ghost_errorString(ghost_error_t);

#define GHOST_SAFECALL(__call) {\
    ghost_error_t __ret = __call;\
    if (__ret != GHOST_SUCCESS) {\
        LOG(GHOST_ERROR,ANSI_COLOR_RED,"%s",ghost_errorString(__ret));\
        return __ret;\
    }\
}\


#endif
