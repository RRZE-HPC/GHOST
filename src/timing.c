#define _XOPEN_SOURCE 600
#include <stdlib.h>
#include <time.h>
#include <errno.h>
#include "ghost/timing.h"
#include "ghost/func_util.h"

static double start_time;

ghost_error ghost_timing_start()
{
    return ghost_timing_wc(&start_time);
}

ghost_error ghost_timing_elapsed(double *time)
{
    double now;
    GHOST_CALL_RETURN(ghost_timing_wc(&now));

    *time = now-start_time;

    return GHOST_SUCCESS;
}

ghost_error ghost_timing_wc(double *time)
{
    struct timespec tp;
    int err;
    err = clock_gettime(CLOCK_MONOTONIC,&tp);
    if (err) {
        GHOST_ERROR_LOG("Error in clock_gettime(): %s",strerror(errno));
        return GHOST_ERR_UNKNOWN;
    }

    *time = (double)tp.tv_sec + (double)tp.tv_nsec/1.e9;
    return GHOST_SUCCESS;
}

ghost_error ghost_timing_wcmilli(double *time)
{
    struct timespec tp;
    int err;
    err = clock_gettime(CLOCK_MONOTONIC,&tp);
    if (err) {
        GHOST_ERROR_LOG("Error in clock_gettime(): %s",strerror(errno));
        return GHOST_ERR_UNKNOWN;
    }

    *time = (double)tp.tv_sec*1000.0 + (double)tp.tv_nsec/1.e6;
    return GHOST_SUCCESS;
}

