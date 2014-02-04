#ifndef GHOST_ERROR_H
#define GHOST_ERROR_H

#include "config.h"
#include "types.h"

char * ghost_errorString(ghost_error_t e);

/**
 * @brief This macro should be used for calling a GHOST function inside
 * a function which itself returns a ghost_error_t.
 * It calls the function and in case of an error LOGS the error message
 * and returns the according ghost_error_t which was return by the called function.
 *
 * @param __call The complete function call.
 *
 * @return A ghost_error_t in case the function return an error.
 */
#define GHOST_CALL_RETURN(__call) {\
    ghost_error_t __ret = __call;\
    if (__ret != GHOST_SUCCESS) {\
        LOG(GHOST_ERROR,ANSI_COLOR_RED,"%s",ghost_errorString(__ret));\
        return __ret;\
    }\
}\


#endif
