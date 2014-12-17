/**
 * @file timing.h
 * @brief Functions and types related to time measurement.
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
 * This macro creates the variables \<func\>_tmin/_tmax/_tavg of type double holding the minimal, maximal and average execution time.
 */
#define GHOST_TIME(nIter,func,...) \
    double func ## _start, func ## _end, func ## _tstart, func ## _tend;\
double func ## _tmin = DBL_MAX;\
double func ## _tmax = 0.;\
double func ## _tavg = 0.;\
int func ## _it;\
ghost_timing_wc(&func ## _tstart);\
for (func ## _it=0; func ## _it<nIter; func ## _it++) {\
    ghost_timing_wc(&func ## _start);\
    func(__VA_ARGS__);\
    ghost_timing_wc(&func ## _end);\
    func ## _tmin = MIN(func ## _end-func ## _start,func ## _tmin);\
    func ## _tmax = MAX(func ## _end-func ## _start,func ## _tmax);\
}\
ghost_timing_wc(&func ## _tend);\
func ## _tavg = (func ## _tend - func ## _tstart)/((double)nIter);\

typedef int (*ghost_compute_performance_func_t)(double *perf, double time, void *arg);

/**
 * @brief Information about a timed region.
 */
typedef struct
{
    /**
     * @brief The number of times the region has been called.
     */
    int nCalls;
    /**
     * @brief The runtime of each call. (length: nCalls)
     */
    double *times;
    /**
     * @brief The average runtime of each call.
     */
    double avgTime;
    /**
     * @brief The minimum runtime.
     */
    double minTime;
    /**
     * @brief The maximum runtime.
     */
    double maxTime;
    /**
     * @brief The accumulated runtime.
     */
    double accTime;
    
}
ghost_timing_region_t;

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @brief Save the start time for a region.
     *
     * @param tag The region tag.
     */
    void ghost_timing_tick(const char *tag);
    /**
     * @brief Save the runtime for a region using the start time from ghost_timing_tick().
     *
     * @param tag The region tag.
     */
    void ghost_timing_tock(const char *tag);
    /**
     * @brief Summarize all timed regions into a string.
     *
     * @param str Where to store the string.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_timing_summarystring(char **str);
    /**
     * @brief Obtain timing info about a specific region.
     *
     * @param region Where to store the information.
     * @param tag The region tag.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_timing_region_create(ghost_timing_region_t ** region, const char *tag);
    /**
     * @brief Destroy a timing region.
     *
     * @param region The region.
     */
    void ghost_timing_region_destroy(ghost_timing_region_t * region);

    /**
     * @brief Get the wallclock time in seconds.
     *
     * @param time Where to store the time.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_timing_wc(double *time);
    /**
     * @brief Get the wallclock time in milliseconds.
     *
     * @param time Where to store the time.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_timing_wcmilli(double *time);

    /**
     * @brief Set a performance computation function to a given tag.
     *
     * @param[in] tag The region tag.
     * @param[in] func The performance callback function.
     * @param[in] arg Argument to the function.
     * @param[in] unit The unit of performance.
     */
    void ghost_timing_set_perfFunc(const char *tag, ghost_compute_performance_func_t func, void *arg, const char *unit);

#ifdef __cplusplus
}
#endif
#endif
