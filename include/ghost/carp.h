/**
 * @file carp.h
 * @brief The CARP (component-averaged row projection) method.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_CARP_H
#define GHOST_CARP_H

#include "error.h"
#include "sparsemat.h"
#include "densemat.h"

ghost_error_t ghost_carp(ghost_sparsemat_t *mat, ghost_densemat_t *x, ghost_densemat_t *b, void *omega);

#endif
