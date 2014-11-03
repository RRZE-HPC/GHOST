/**
 * @file svqb.h
 * @brief The SVQB function.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 *
 * J. R. McCombs, R. T. Mills and A. Stathopoulos 
 * "Dynamic Load Balancing of an iterative eigensolver on Grids of heterogeneous clusters",
 * in International Parallel and Distributed Processing Symposium (IPDPS 2003), Nice, France.
 */
#ifndef GHOST_SVQB_H
#define GHOST_SVQB_H

#include "ghost/config.h"
#include "ghost/error.h"
#include "ghost/densemat.h"

#ifdef __cplusplus
extern "C" {
#endif

ghost_error_t ghost_svqb(ghost_densemat_t * v_ot , ghost_densemat_t * v);

#ifdef __cplusplus
}
#endif

#endif
