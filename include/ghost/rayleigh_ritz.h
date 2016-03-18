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

/**
 * @brief Flags to configure the ghost_rayleigh_ritz routine.
 */
typedef enum{
    GHOST_RAYLEIGHRITZ_DEFAULT      = 0,
    GHOST_RAYLEIGHRITZ_RESIDUAL     = 1,
    GHOST_RAYLEIGHRITZ_KAHAN        = 2,
    GHOST_RAYLEIGHRITZ_GENERALIZED  = 4
}ghost_rayleighritz_flags;

ghost_error  ghost_rayleigh_ritz(ghost_sparsemat * mat, void * eigs, void * res,  ghost_densemat * v_eigs , ghost_densemat * v_res, ghost_rayleighritz_flags RR_Obtion, ghost_spmv_flags spMVM_Options);
ghost_error ghost_grayleigh_ritz(ghost_sparsemat * mat, void * eigs, void * res,  ghost_densemat * v_eigs , ghost_densemat * v_res, ghost_rayleighritz_flags RR_Obtion, ghost_spmv_flags spMVM_Options);


#ifdef __cplusplus
}
#endif

#endif
