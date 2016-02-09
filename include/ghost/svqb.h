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

ghost_error ghost_svqb(ghost_densemat * v_ot , ghost_densemat * v);
ghost_error ghost_blockortho(ghost_densemat * w , ghost_densemat * v);
ghost_error ghost_svd_deflation( ghost_lidx *svd_offset, ghost_densemat * ot_vec, ghost_densemat * vec, float limit);


#ifdef __cplusplus
}
#endif

#endif
