#ifndef GHOST_UTIL_H
#define GHOST_UTIL_H

#include "config.h"
#include "types.h"

#if GHOST_HAVE_CUDA
#include "cu_util.h"
#endif

#include <stdio.h>

#ifndef __cplusplus
#include <complex.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#else
#include <complex>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#endif

#ifndef MIN
#define MIN(x,y) ((x)<(y)?(x):(y))
#endif
#ifndef MAX
#define MAX(x,y) ((x)<(y)?(y):(x))
#endif

#define GHOST_TIME(_niter,_func,...)\
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
    _func ## _tavg = (ghost_wctime()-_func ## _tstart)/((double)_niter);




#define UNUSED(x) (void)(x)
/******************************************************************************/


#ifdef __cplusplus
extern "C" {
#endif

    void ghost_printHeader(const char *fmt, ...);
    void ghost_printFooter(); 
    void ghost_printLine(const char *label, const char *unit, const char *format, ...);
    ghost_error_t ghost_printSysInfo();
    ghost_error_t ghost_printGhostInfo();

    char * ghost_workdistName(int ghostOptions);
    char * ghost_modeName(int spmvmOptions);
    char * ghost_datatypeName(int datatype);
    char * ghost_symmetryName(int symmetry);

    /**
     * @brief Pad a number to a multiple of a second number.
     *
     * @param nrows The number to be padded.
     * @param padding The desired padding.
     *
     * @return nrows padded to a multiple of padding or nrows if padding or nrows are smaller than 1.
     */
    ghost_midx_t ghost_pad(ghost_midx_t nrows, ghost_midx_t padding);

    int ghost_datatypeValid(int datatype);
    int ghost_symmetryValid(int symmetry);
    int ghost_dataTypeIdx(int datatype);

    double ghost_wctime();
    double ghost_wctimemilli();

    void *ghost_malloc(const size_t size);
    void *ghost_malloc_align(const size_t size, const size_t align);
    
    void ghost_ompSetNumThreads(int nthreads);
    int ghost_ompGetThreadNum();
    int ghost_ompGetNumThreads();
    ghost_error_t ghost_mpi_datatype(ghost_mpi_datatype_t *dt, int datatype);
    ghost_error_t ghost_mpi_op_sum(ghost_mpi_op_t * op, int datatype);

    /**
     * @brief Computes a hash from three integral input values.
     *
     * @param a First parameter.
     * @param b Second parameter.
     * @param c Third parameter.
     *
     * @return Hash value.
     *
     * The function has been taken from http://burtleburtle.net/bob/hash/doobs.html
     */
    int ghost_hash(int a, int b, int c);

#ifdef __cplusplus
}
#endif
#endif
