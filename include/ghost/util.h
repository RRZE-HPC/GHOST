/**
 * @file util.h
 * @brief General utility functions.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_UTIL_H
#define GHOST_UTIL_H

#include "config.h"
#include "types.h"

#ifdef GHOST_HAVE_CUDA
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

/**
 * @brief Pad a number to a multiple of a second number.
 *
 * @param n The number to be padded.
 * @param p The desired padding.
 *
 * @return n padded to a multiple of p or n if p or n smaller than 1.
 */
#define PAD(n,p) (((n)<1 || (p)<1)?(n):(((n) % (p)) ? ((n) + (p) - (n) % (p)) : (n)))

/**
 * @brief Avoid unused variable/function warnings.
 */
#define UNUSED(x) (void)(x)


#ifdef __cplusplus
extern "C" {
#endif

    void ghost_printHeader(char **str, const char *fmt, ...);
    void ghost_printFooter(char **str); 
    void ghost_printLine(char **str, const char *label, const char *unit, const char *format, ...);


    /**
     * @brief Allocate memory.
     *
     * @param mem Where to store the allocated memory.
     * @param size The size (in bytes) to allocate.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_malloc(void **mem, const size_t size);
    /**
     * @brief Allocate aligned memory.
     *
     * @param mem Where to store the allocated memory.
     * @param size The size (in bytes) to allocate.
     * @param align The alignment size.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_malloc_align(void **mem, const size_t size, const size_t align);

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
