/**
 * @file sell_kacz.h
 * @brief The Kaczmarz sweep for the SELL format.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_SELL_KACZ_H
#define GHOST_SELL_KACZ_H

#include "error.h"
#include "sparsemat.h"
#include "densemat.h"


#ifdef __cplusplus
extern "C" {
#endif
ghost_error_t ghost_sell_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega, int forward);
#ifdef __cplusplus
}
#endif

#endif

