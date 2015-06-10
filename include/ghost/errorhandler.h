/**
 * @file errorhandler.h
 * @brief Functionality for user-defined error handlers.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_ERRORHANDLER_H
#define GHOST_ERRORHANDLER_H

#include "error.h"

typedef void * (*ghost_errorhandler_t)(void *error);

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @brief Get the handler of a given GHOST error.
     *
     * @param e The error.
     *
     * @return The error handler of NULL if none is set.
     */
    ghost_errorhandler_t ghost_errorhandler_get(ghost_error_t e);


    /**
     * @brief Get the handler of a given GHOST error.
     *
     * @param e The error.
     * @param h The handler.
     */
    ghost_error_t ghost_errorhandler_set(ghost_error_t e, ghost_errorhandler_t h);
#ifdef __cplusplus
}
#endif

#endif
