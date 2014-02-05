#ifndef GHOST_CORE_H
#define GHOST_CORE_H

#include "error.h"

typedef enum {
    GHOST_TYPE_INVALID, 
    GHOST_TYPE_COMPUTE, 
    GHOST_TYPE_CUDAMGMT
} ghost_type_t;

#ifdef __cplusplus
extern "C" {
#endif

ghost_error_t ghost_init(int argc, char **argv);
ghost_error_t ghost_finalize();
ghost_error_t ghost_setType(ghost_type_t t);
ghost_error_t ghost_getType(ghost_type_t *t);

#ifdef __cplusplus
}
#endif

#endif
