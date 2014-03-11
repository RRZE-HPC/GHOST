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

#define GHOST_INSTR_START(tag) {\
    char region[256];\
    snprintf(region,256,"%s%s%s",ghost_instr_prefix_get(), #tag, ghost_instr_suffix_get());\
    ghost_timing_tick(region);\
}\

#define GHOST_INSTR_STOP(tag) {\
    char region[256];\
    snprintf(region,256,"%s%s%s",ghost_instr_prefix_get(), #tag, ghost_instr_suffix_get());\
    ghost_timing_tock(region);\
}\
    
//double __start_##tag;\
//ghost_error_t __err_##tag;\
//GHOST_CALL(ghost_wctime(&__start_##tag),__err_##tag);\

//#define GHOST_INSTR_STOP(tag)\
//    double __end_##tag;\
//GHOST_CALL(ghost_wctime(&__end_##tag),__err_##tag);\
//LOG(TIMING,ANSI_COLOR_BLUE, "%s%s%s: %e secs" ANSI_COLOR_RESET,ghost_instr_getPrefix(),#tag,ghost_instr_getSuffix(),__end_##tag-__start_##tag);

#elif defined(GHOST_HAVE_INSTR_LIKWID)

#include <likwid.h>

/**
 * @brief Start a LIKWID marker region.
 *
 * @param tag The tag identifying the region.
 */
#define GHOST_INSTR_START(tag) {\
    char region[256];\
    snprintf(region,256,"%s%s%s",ghost_instr_prefix_get(), #tag, ghost_instr_suffix_get());\
    _Pragma("omp parallel")\
    LIKWID_MARKER_START(region);\
}

/**
 * @brief Stop a LIKWID marker region.
 *
 * @param tag The tag identifying the region.
 */
#define GHOST_INSTR_STOP(tag) {\
    char region[256];\
    snprintf(region,256,"%s%s%s",ghost_instr_prefix_get(), #tag, ghost_instr_suffix_get());\
    _Pragma("omp parallel")\
    LIKWID_MARKER_STOP(region);\
}

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