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

    /**
     * @ingroup globops
     * @ingroup locops
     *
     * @brief Compute the dot product of two dense vectors/matrices.
     *
     * @param res Where to store the result.
     * @param vec1 The first vector/matrix.
     * @param vec2 The second vector/matrix.
     * @param mpicomm The communicator if a global reduction of the result for distributed densemats should be done.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     *
     * This function first computes the local dot product. 
     * If a context is given, it then performs an allreduce on the result.
     */
    ghost_error ghost_dot(void *res, ghost_densemat *vec1, ghost_densemat *vec2, ghost_mpi_comm mpicomm);


#ifdef __cplusplus
}
#endif
#endif
