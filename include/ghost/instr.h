/**
 * @file instr.h
 * @brief Macros used for code instrumentation.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_INSTR_H
#define GHOST_INSTR_H

#ifdef GHOST_HAVE_INSTR_TIMING

#include "log.h"
#include "timing.h"

#endif

#ifdef GHOST_HAVE_INSTR_LIKWID

#include <likwid.h>

#endif

#ifdef GHOST_HAVE_INSTR_TIMING

#ifdef GHOST_HAVE_INSTR_LIKWID

#define GHOST_INSTR_START(tag) {\
    char region[256] = "";\
    snprintf(region,256,"%s%s%s",ghost_instr_prefix_get(), #tag, ghost_instr_suffix_get());\
    ghost_timing_tick(region);\
    _Pragma("omp parallel")\
    LIKWID_MARKER_START(region);\
}\

#define GHOST_INSTR_STOP(tag) {\
    char region[256] = "";\
    snprintf(region,256,"%s%s%s",ghost_instr_prefix_get(), #tag, ghost_instr_suffix_get());\
    ghost_timing_tock(region);\
    _Pragma("omp parallel")\
    LIKWID_MARKER_STOP(region);\
}\
    
#else

#define GHOST_INSTR_START(tag) {\
    char region[256] = "";\
    snprintf(region,256,"%s%s%s",ghost_instr_prefix_get(), #tag, ghost_instr_suffix_get());\
    ghost_timing_tick(region);\
}\

#define GHOST_INSTR_STOP(tag) {\
    char region[256] = "";\
    snprintf(region,256,"%s%s%s",ghost_instr_prefix_get(), #tag, ghost_instr_suffix_get());\
    ghost_timing_tock(region);\
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
    char region[256] = "";\
    snprintf(region,256,"%s%s%s",ghost_instr_prefix_get(), #tag, ghost_instr_suffix_get());\
    _Pragma("omp parallel")\
    LIKWID_MARKER_START(region);\
}\

/**
 * @brief Stop a LIKWID marker region.
 *
 * @param tag The tag identifying the region.
 */
#define GHOST_INSTR_STOP(tag) {\
    char region[256] = "";\
    snprintf(region,256,"%s%s%s",ghost_instr_prefix_get(), #tag, ghost_instr_suffix_get());\
    _Pragma("omp parallel")\
    LIKWID_MARKER_STOP(region);\
}\
    
#else

/**
 * @brief Instrumentation will be ignored. 
 */
#define GHOST_INSTR_START(tag)

/**
 * @brief Instrumentation will be ignored. 
 */
#define GHOST_INSTR_STOP(tag)

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
    void ghost_instr_prefix_set(char *prefix);
    
    char *ghost_instr_prefix_get();
    /**
     * @brief Set the instrumentation suffix.
     *
     * @param suffix The suffix.
     *
     * The suffix will be appended to the instrumentation tag.
     */
    void ghost_instr_suffix_set(char *suffix);

    char *ghost_instr_suffix_get();

#ifdef __cplusplus
}
#endif

#endif
