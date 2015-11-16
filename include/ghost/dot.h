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
    ghost_datatype_t dt;
    /**
     * @brief The vector block size.
     */
    int blocksz;
    /**
     * @brief The second configure block size M.
     */
    ghost_implementation_t impl;
    ghost_alignment_t alignment;
    ghost_densemat_storage_t storage;
} ghost_dot_parameters_t;

typedef ghost_error_t (*ghost_dot_kernel_t)(void *, ghost_densemat_t *, ghost_densemat_t *);

#ifdef __cplusplus
extern "C" {
#endif

    ghost_error_t ghost_localdot(void *res, ghost_densemat_t *vec1, ghost_densemat_t *vec2);
    ghost_error_t ghost_dot(void *res, ghost_densemat_t *vec1, ghost_densemat_t *vec2);


#ifdef __cplusplus
}
#endif
#endif
