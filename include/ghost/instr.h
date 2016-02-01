/**
 * @file instr.h
 * @brief Macros used for code instrumentation.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_INSTR_H
#define GHOST_INSTR_H

#include "config.h"
#include <ghost/log.h>
#include <ghost/error.h>
#include <pthread.h>

#ifdef GHOST_HAVE_INSTR_TIMING

#include <ghost/timing.h>

#endif

#ifdef GHOST_HAVE_INSTR_LIKWID

#include <likwid.h>

#endif

extern pthread_key_t ghost_instr_enable_key;

#ifdef GHOST_HAVE_INSTR_TIMING

#ifdef GHOST_HAVE_INSTR_LIKWID

#define GHOST_INSTR_START(tag) {\
    if (pthread_getspecific(ghost_instr_enable_key)) {\
        if (GHOST_VERBOSITY > 1) {\
            DEBUG_LOG(1,"Enter instrumented region %s",tag);\
        }\
        char region[256] = "";\
        snprintf(region,256,"%s%s%s",ghost_instr_prefix_get(), tag, ghost_instr_suffix_get());\
        ghost_timing_tick(region);\
        _Pragma("omp parallel")\
        likwid_markerStartRegion(region);\
    }\
}\

#define GHOST_INSTR_STOP(tag) {\
    if (pthread_getspecific(ghost_instr_enable_key)) {\
        if (GHOST_VERBOSITY > 1) {\
            DEBUG_LOG(1,"Exit instrumented region %s",tag);\
        }\
        char region[256] = "";\
        snprintf(region,256,"%s%s%s",ghost_instr_prefix_get(), tag, ghost_instr_suffix_get());\
        ghost_timing_tock(region);\
        _Pragma("omp parallel")\
        likwid_markerStopRegion(region);\
    }\
}\
    
#else

#define GHOST_INSTR_START(tag) {\
    if (pthread_getspecific(ghost_instr_enable_key)) {\
        if (GHOST_VERBOSITY > 1) {\
            DEBUG_LOG(1,"Enter instrumented region %s",tag);\
        }\
        char region[256] = "";\
        snprintf(region,256,"%s%s%s",ghost_instr_prefix_get(), tag, ghost_instr_suffix_get());\
        ghost_timing_tick(region);\
    }\
}\

#define GHOST_INSTR_STOP(tag) {\
    if (pthread_getspecific(ghost_instr_enable_key)) {\
        if (GHOST_VERBOSITY > 1) {\
            DEBUG_LOG(1,"Exit instrumented region %s",tag);\
        }\
        char region[256] = "";\
        snprintf(region,256,"%s%s%s",ghost_instr_prefix_get(), tag, ghost_instr_suffix_get());\
        ghost_timing_tock(region);\
    }\
}\

#endif

#else //GHOST_HAVE_INSTR_TIMING

#ifdef GHOST_HAVE_INSTR_LIKWID

/**
 * @brief Start a LIKWID marker region.
 *
 * @param tag The tag identifying the region.
 */
#define GHOST_INSTR_START(tag) {\
    if (pthread_getspecific(ghost_instr_enable_key)) {\
        if (GHOST_VERBOSITY > 1) {\
            DEBUG_LOG(1,"Enter instrumented region %s",tag);\
        }\
        char region[256] = "";\
        snprintf(region,256,"%s%s%s",ghost_instr_prefix_get(), tag, ghost_instr_suffix_get());\
        _Pragma("omp parallel")\
        likwid_markerStartRegion(region);\
    }\
}\

/**
 * @brief Stop a LIKWID marker region.
 *
 * @param tag The tag identifying the region.
 */
#define GHOST_INSTR_STOP(tag) {\
    if (pthread_getspecific(ghost_instr_enable_key)) {\
        if (GHOST_VERBOSITY > 1) {\
            DEBUG_LOG(1,"Exit instrumented region %s",tag);\
        }\
        char region[256] = "";\
        snprintf(region,256,"%s%s%s",ghost_instr_prefix_get(), tag, ghost_instr_suffix_get());\
        _Pragma("omp parallel")\
        likwid_markerStopRegion(region);\
    }\
}\
    
#else

/**
 * @brief Instrumentation will be ignored. 
 */
#define GHOST_INSTR_START(tag)\
    if (GHOST_VERBOSITY > 1) {\
        DEBUG_LOG(1,"Enter instrumented region %s",tag);\
    }\

/**
 * @brief Instrumentation will be ignored. 
 */
#define GHOST_INSTR_STOP(tag) \
    if (GHOST_VERBOSITY > 1) {\
        DEBUG_LOG(1,"Exit instrumented region %s",tag);\
    }\

#endif

#endif


#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @brief Set the instrumentation prefix.
     *
     * @param prefix The prefix.
     *
     * The prefix will be prepended to the instrumentation tag.
     */
    void ghost_instr_prefix_set(const char *prefix);
    
    char *ghost_instr_prefix_get();
    /**
     * @brief Set the instrumentation suffix.
     *
     * @param suffix The suffix.
     *
     * The suffix will be appended to the instrumentation tag.
     */
    void ghost_instr_suffix_set(const char *suffix);

    char *ghost_instr_suffix_get();
    ghost_error ghost_instr_create();
    ghost_error ghost_instr_destroy();

#ifdef __cplusplus
}
#endif

#endif
