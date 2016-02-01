/**
 * @file dot.h
 * @brief The (block vector) dot product.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_DOT_H
#define GHOST_DOT_H

#include "config.h"
#include "types.h"
#include "densemat.h"
#include "math.h"

typedef struct
{
    /**
     * @brief The data type of the densemats.
     */
    ghost_datatype dt;
    /**
     * @brief The vector block size.
     */
    int blocksz;
    /**
     * @brief The second configure block size M.
     */
    ghost_implementation impl;
    ghost_alignment alignment;
    ghost_densemat_storage storage;
} ghost_dot_parameters;

typedef ghost_error (*ghost_dot_kernel)(void *, ghost_densemat *, ghost_densemat *);

#ifdef __cplusplus
extern "C" {
#endif

    ghost_error ghost_localdot(void *res, ghost_densemat *vec1, ghost_densemat *vec2);
    ghost_error ghost_dot(void *res, ghost_densemat *vec1, ghost_densemat *vec2);


#ifdef __cplusplus
}
#endif
#endif
