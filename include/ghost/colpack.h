/**
 * @file colpack.h
 * @brief Functions for matrix coloring.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_COLPACK_H
#define GHOST_COLPACK_H

#include "error.h"
#include "sparsemat.h"

ghost_error_t ghost_sparsemat_coloring_create_cpp(ghost_sparsemat_t *mat);

#ifdef __cplusplus
extern "C" {
#endif

    ghost_error_t ghost_sparsemat_coloring_create(ghost_sparsemat_t *mat);
    
#ifdef __cplusplus
}
#endif

#endif
