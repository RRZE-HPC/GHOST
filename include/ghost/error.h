#ifndef __GHOST_ERROR_H__
#define __GHOST_ERROR_H__

#include "config.h"
#include "types.h"

char * ghost_errorString(ghost_error_t e);

#define GHOST_SAFECALL(__call) {\
    ghost_error_t __ret = __call;\
    if (__ret != GHOST_SUCCESS) {\
        LOG(GHOST_ERROR,ANSI_COLOR_RED,"%s",ghost_errorString(__ret));\
        return __ret;\
    }\
}\


#endif
