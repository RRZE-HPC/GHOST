#ifndef GHOST_TIMING_H
#define GHOST_TIMING_H

#include "error.h"

#define GHOST_TIME(_niter,_func,...) {\
    double _func ## _start, _func ## _end, _func ## _tstart;\
    double _func ## _tmin = DBL_MAX;\
    double _func ## _tmax = 0.;\
    double _func ## _tavg = 0.;\
    int _func ## _it;\
    _func ## _tstart=ghost_wctime();\
    for (_func ## _it=0; _func ## _it<_niter; _func ## _it++) {\
        _func ## _start = ghost_wctime();\
        _func(__VA_ARGS__);\
        _func ## _end = ghost_wctime();\
        _func ## _tmin = MIN(_func ## _end-_func ## _start,_func ## _tmin);\
        _func ## _tmin = MAX(_func ## _end-_func ## _start,_func ## _tmin);\
    }\
    _func ## _tavg = (ghost_wctime()-_func ## _tstart)/((double)_niter);\
}\

#ifdef __cplusplus
extern "C" {
#endif

    ghost_error_t ghost_wctime(double *time);
    ghost_error_t ghost_wctimeMilli(double *time);

#ifdef __cplusplus
}
#endif
#endif
