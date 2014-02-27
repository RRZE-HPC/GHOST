/**
 * @file timing.h
 * @brief Macros and functions related to time measurement.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_TIMING_H
#define GHOST_TIMING_H

#include <float.h>

#include "error.h"

#ifndef MIN
#define MIN(x,y) ((x)<(y)?(x):(y))
#endif
#ifndef MAX
#define MAX(x,y) ((x)<(y)?(y):(x))
#endif

/**
 * @brief Measure the execution time of a function with a given numbe of iterations.
 *
 * @param nIter The number of iterations.
 * @param func The function.
 * @param ... The function's arguments.
 *
 * This macro creates the variables <func>_tmin/_tmax/_tavg of type double holding the minimal, maximal and average execution time.
 */
#define GHOST_TIME(nIter,func,...) \
    double func ## _start, func ## _end, func ## _tstart, func ## _tend;\
    double func ## _tmin = DBL_MAX;\
    double func ## _tmax = 0.;\
    double func ## _tavg = 0.;\
    int func ## _it;\
    ghost_wctime(&func ## _tstart);\
    for (func ## _it=0; func ## _it<nIter; func ## _it++) {\
        ghost_wctime(&func ## _start);\
        func(__VA_ARGS__);\
        ghost_wctime(&func ## _end);\
        func ## _tmin = MIN(func ## _end-func ## _start,func ## _tmin);\
        func ## _tmax = MAX(func ## _end-func ## _start,func ## _tmax);\
    }\
    ghost_wctime(&func ## _tend);\
    func ## _tavg = (func ## _tend - func ## _tstart)/((double)nIter);\

typedef struct
{
    int nCalls;
    double *times;
    double avgTime;
    double minTime;
    double maxTime;
}
ghost_timing_regionInfo_t;

#ifdef __cplusplus
extern "C" {
#endif

    void ghost_timing_tick(char *tag);
    void ghost_timing_tock(char *tag);
    ghost_error_t ghost_timing_summaryString(char **str);
    ghost_error_t ghost_timing_regionInfo_create(ghost_timing_regionInfo_t ** ri, char *region);
    void ghost_timing_regionInfo_destroy(ghost_timing_regionInfo_t * ri);

    ghost_error_t ghost_wctime(double *time);
    ghost_error_t ghost_wctimeMilli(double *time);

#ifdef __cplusplus
}
#endif
#endif
