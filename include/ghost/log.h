/**
 * @file log.h
 * @brief Macros for logging.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_LOG_H
#define GHOST_LOG_H

#ifdef GHOST_FUJITSU
#define __func__ "unknown"
#endif

#include "config.h"

#ifdef GHOST_HAVE_MPI
#include <mpi.h>
#endif

#ifdef __cplusplus
#include <cstdio>
#include <cstring>
#else
#include <stdio.h>
#include <string.h>
#endif

#define GHOST_ANSI_COLOR_RED     "\x1b[31m"
#define GHOST_ANSI_COLOR_GREEN   "\x1b[32m"
#define GHOST_ANSI_COLOR_YELLOW  "\x1b[33m"
#define GHOST_ANSI_COLOR_BLUE    "\x1b[34m"
#define GHOST_ANSI_COLOR_MAGENTA "\x1b[35m"
#define GHOST_ANSI_COLOR_CYAN    "\x1b[36m"
#define GHOST_ANSI_COLOR_RESET   "\x1b[0m"

#define GHOST_IF_DEBUG(level) if(GHOST_VERBOSITY > level)
#define GHOST_FILE_BASENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

/* taken from http://stackoverflow.com/a/11172679 */
/* expands to the first argument */
#define GHOST_FIRST(...) GHOST_FIRST_HELPER(__VA_ARGS__, throwaway)
#define GHOST_FIRST_HELPER(first, ...) first

/*
 * if there's only one argument, expands to nothing.  if there is more
 * than one argument, expands to a comma followed by everything but
 * the first argument.  only supports up to 9 arguments but can be
 * trivially expanded.
 */
#define REST(...) REST_HELPER(NUM(__VA_ARGS__), __VA_ARGS__)
#define REST_HELPER(qty, ...) REST_HELPER2(qty, __VA_ARGS__)
#define REST_HELPER2(qty, ...) REST_HELPER_##qty(__VA_ARGS__)
#define REST_HELPER_ONE(first)
#define REST_HELPER_TWOORMORE(first, ...) , __VA_ARGS__
#define NUM(...) \
    SELECT_20TH(__VA_ARGS__, TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE,\
            TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE, ONE, throwaway)
#define SELECT_20TH(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, ...) a20

#ifdef GHOST_HAVE_MPI

#ifdef GHOST_LOG_TIMESTAMP

#define GHOST_LOG(type,color,...) {\
    double logmacrotime;\
    ghost_timing_elapsed(&logmacrotime);\
    int logmacrome;\
    int logmacroerr = MPI_Comm_rank(MPI_COMM_WORLD,&logmacrome);\
    if (logmacroerr != MPI_SUCCESS) {\
        logmacrome = -1;\
    }\
    if (logmacrome == GHOST_LOG_RANK || -1 == GHOST_LOG_RANK) {\
        fprintf(stderr, color "[GHOST] PE%d %.3f " #type " at %s() <%s:%d>: " GHOST_FIRST(__VA_ARGS__) GHOST_ANSI_COLOR_RESET "\n", logmacrome, logmacrotime, __func__, GHOST_FILE_BASENAME, __LINE__ REST(__VA_ARGS__)); \
        fflush(stderr);\
    }\
}\

#else 

#define GHOST_LOG(type,color,...) {\
    int logmacrome;\
    int logmacroerr = MPI_Comm_rank(MPI_COMM_WORLD,&logmacrome);\
    if (logmacroerr != MPI_SUCCESS) {\
        logmacrome = -1;\
    }\
    if (logmacrome == GHOST_LOG_RANK || -1 == GHOST_LOG_RANK) {\
        fprintf(stderr, color "[GHOST] PE%d " #type " at %s() <%s:%d>: " GHOST_FIRST(__VA_ARGS__) GHOST_ANSI_COLOR_RESET "\n", logmacrome, __func__, GHOST_FILE_BASENAME, __LINE__ REST(__VA_ARGS__)); \
        fflush(stderr);\
    }\
}\

#endif

#else

#ifdef GHOST_LOG_TIMESTAMP

#define GHOST_LOG(type,color,...) {\
    double locmacrotime;\
    ghost_timing_elapsed(&logmacrotime);\
    fprintf(stderr, color "[GHOST] %.3f " #type " at %s() <%s:%d>: " GHOST_FIRST(__VA_ARGS__) GHOST_ANSI_COLOR_RESET "\n", logmacrotime, __func__, GHOST_FILE_BASENAME, __LINE__ REST(__VA_ARGS__));\
}\

#else

#define GHOST_LOG(type,color,...) {\
    fprintf(stderr, color "[GHOST] " #type " at %s() <%s:%d>: " GHOST_FIRST(__VA_ARGS__) GHOST_ANSI_COLOR_RESET "\n", __func__, GHOST_FILE_BASENAME, __LINE__ REST(__VA_ARGS__));\
}\

#endif

#endif


#define GHOST_DEBUG_LOG(level,...) {if(GHOST_VERBOSITY > level) { GHOST_LOG(DEBUG,GHOST_ANSI_COLOR_RESET,__VA_ARGS__) }}

#ifdef GHOST_LOG_ONLYFIRST
#define GHOST_INFO_LOG(...) {static int __printed = 0; if(!__printed && GHOST_VERBOSITY) { GHOST_LOG(INFO,GHOST_ANSI_COLOR_BLUE,__VA_ARGS__); __printed=1; }}
#define GHOST_WARNING_LOG(...) {static int __printed = 0; if(!__printed && GHOST_VERBOSITY) { GHOST_LOG(WARNING,GHOST_ANSI_COLOR_YELLOW,__VA_ARGS__); __printed=1; }}
#define GHOST_PERFWARNING_LOG(...) {static int __printed = 0; if(!__printed && GHOST_VERBOSITY) { GHOST_LOG(PERFWARNING,GHOST_ANSI_COLOR_MAGENTA,__VA_ARGS__); __printed=1; }}
#define GHOST_ERROR_LOG(...) {static int __printed = 0; if(!__printed && GHOST_VERBOSITY) { GHOST_LOG(ERROR,GHOST_ANSI_COLOR_RED,__VA_ARGS__); __printed=1; }}
#else
#define GHOST_INFO_LOG(...) {if (GHOST_VERBOSITY) { GHOST_LOG(INFO,GHOST_ANSI_COLOR_BLUE,__VA_ARGS__); }}
#define GHOST_WARNING_LOG(...) {if (GHOST_VERBOSITY) { GHOST_LOG(WARNING,GHOST_ANSI_COLOR_YELLOW,__VA_ARGS__); }}
#define GHOST_PERFWARNING_LOG(...) {if (GHOST_VERBOSITY) { GHOST_LOG(PERFWARNING,GHOST_ANSI_COLOR_MAGENTA,__VA_ARGS__); }}
#define GHOST_ERROR_LOG(...) {if (GHOST_VERBOSITY) { GHOST_LOG(ERROR,GHOST_ANSI_COLOR_RED,__VA_ARGS__); }}
#endif

#endif
