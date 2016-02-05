/**
 * @file rayleigh_ritz.h
 * @brief The RAYLEIGH_RITZ function.
 * @author Andreas Pieper <pieper@physik.uni-greifswald.de>
 *
 */
#ifndef GHOST_RAYLEIGH_RITZ_H
#define GHOST_RAYLEIGH_RITZ_H

#include "ghost/config.h"
#include "ghost/error.h"
#include "ghost/densemat.h"
#include "ghost/spmv.h"
#include "ghost/sparsemat.h"

#ifdef __cplusplus
extern "C" {
#endif

ghost_error  ghost_rayleigh_ritz(ghost_sparsemat * mat, void * eigs, void * res,  ghost_densemat * v_eigs , ghost_densemat * v_res, int obtion, ghost_spmv_flags spMVM_Options);
ghost_error ghost_grayleigh_ritz(ghost_sparsemat * mat, void * eigs, void * res,  ghost_densemat * v_eigs , ghost_densemat * v_res, int obtion, ghost_spmv_flags spMVM_Options);


#ifdef __cplusplus
}
#endif

#endif
