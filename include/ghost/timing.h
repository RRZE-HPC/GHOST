/**
 * @file timing.h
 * @brief Macros and functions related to time measurement.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_TIMING_H
#define GHOST_TIMING_H

#include "error.h"

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
    double func ## _start, func ## _end, func ## _tstart;\
    double func ## _tmin = DBL_MAX;\
    double func ## _tmax = 0.;\
    double func ## _tavg = 0.;\
    int func ## _it;\
    func ## _tstart=ghost_wctime();\
    for (func ## _it=0; func ## _it<nIter; func ## _it++) {\
        func ## _start = ghost_wctime();\
        func(__VA_ARGS__);\
        func ## _end = ghost_wctime();\
        func ## _tmin = MIN(func ## _end-func ## _start,func ## _tmin);\
        func ## _tmin = MAX(func ## _end-func ## _start,func ## _tmin);\
    }\
    func ## _tavg = (ghost_wctime()-func ## _tstart)/((double)nIter);\

#ifdef __cplusplus
extern "C" {
#endif

    ghost_error_t ghost_wctime(double *time);
    ghost_error_t ghost_wctimeMilli(double *time);

#ifdef __cplusplus
}
#endif
#endif
