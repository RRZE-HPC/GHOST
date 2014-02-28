#include <stdlib.h>
#include <sys/time.h>
#include <errno.h>
#include "ghost/timing.h"


ghost_error_t ghost_timing_wc(double *time)
{
    struct timeval tp;
    int err;
    err = gettimeofday(&tp, NULL);
    if (err) {
        ERROR_LOG("Error in gettimeofday: %s",strerror(errno));
        return GHOST_ERR_UNKNOWN;
    }

    *time = (double)tp.tv_sec + (double)tp.tv_usec/1000000.0;

    return GHOST_SUCCESS;
}

ghost_error_t ghost_timing_wcmilli(double *time)
{
    struct timeval tp;
    int err;
    err = gettimeofday(&tp, NULL);
    if (err) {
        ERROR_LOG("Error in gettimeofday: %s",strerror(errno));
        return GHOST_ERR_UNKNOWN;
    }

    *time = (double)tp.tv_sec*1000.0 + (double)tp.tv_usec/1000.0;

    return GHOST_SUCCESS;
}

