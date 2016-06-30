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

ghost_error ghost_carp_rb(ghost_sparsemat *mat, ghost_densemat *x, ghost_densemat *b, void *omega, int flag_rb);

#endif
