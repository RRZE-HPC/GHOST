/**
 * @file densemat_cm.h
 * @brief Types and functions related to column major dense matrices/vectors.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_DENSEMAT_CM_H
#define GHOST_DENSEMAT_CM_H

#include "config.h"
#include "types.h"
#include "context.h"
#include "densemat.h"
#ifdef GHOST_HAVE_CUDA
#include "cu_densemat_cm.h"
#endif

#ifdef __cplusplus

extern "C" {
#endif

    ghost_error ghost_densemat_cm_malloc(ghost_densemat *vec, int* needInit);
    ghost_error ghost_densemat_cm_string_selector(ghost_densemat *vec, char **str);
    ghost_error ghost_densemat_cm_normalize_selector(ghost_densemat *vec);
    ghost_error ghost_densemat_cm_dotprod_selector(ghost_densemat *vec, void *, ghost_densemat *);
    ghost_error ghost_densemat_cm_vscale_selector(ghost_densemat *vec1, void *); 
    ghost_error ghost_densemat_cm_vaxpy_selector(ghost_densemat *vec1, ghost_densemat *vec2, void *); 
    ghost_error ghost_densemat_cm_vaxpby_selector(ghost_densemat *vec1, ghost_densemat *vec2, void *, void *); 
    ghost_error ghost_densemat_cm_fromScalar_selector(ghost_densemat *vec, void *);
    ghost_error ghost_densemat_cm_fromRand_selector(ghost_densemat *vec);
    ghost_error ghost_densemat_cm_fromVec_selector(ghost_densemat *vec1, ghost_densemat *vec2, ghost_lidx, ghost_lidx); 
    ghost_error ghost_densemat_cm_fromReal_selector(ghost_densemat *vec, ghost_densemat *re, ghost_densemat *im); 
    ghost_error ghost_densemat_cm_fromComplex_selector(ghost_densemat *re, ghost_densemat *im, ghost_densemat *c); 
    ghost_error ghost_densemat_cm_permute_selector(ghost_densemat *vec, ghost_context *ctx, ghost_permutation_direction dir);
    ghost_error ghost_densemat_cm_norm_selector(ghost_densemat *vec, void *res, void *p);
    ghost_error ghost_densemat_cm_averagehalo_selector(ghost_densemat *vec, ghost_context *ctx);
    ghost_error ghost_densemat_cm_conj_selector(ghost_densemat *vec);
    ghost_error ghost_densemat_cm_vaxpbypcz_selector(ghost_densemat *vec, ghost_densemat *vec2, void *scale, void *b, ghost_densemat *vec3, void *c);
    ghost_error ghost_densemat_cm_axpy(ghost_densemat *vec1, ghost_densemat *vec2, void *); 
    ghost_error ghost_densemat_cm_axpby(ghost_densemat *vec1, ghost_densemat *vec2, void *, void *); 
    ghost_error ghost_densemat_cm_axpbypcz(ghost_densemat *vec1, ghost_densemat *vec2, void *, void *, ghost_densemat *vec3, void *); 
    ghost_error ghost_densemat_cm_scale(ghost_densemat *vec, void *); 
    ghost_error ghost_densemat_cm_distributeVector(ghost_densemat *vec, ghost_densemat *nodeVec, ghost_context *ctx);
    ghost_error ghost_densemat_cm_collectVectors(ghost_densemat *vec, ghost_densemat *totalVec, ghost_context *ctx); 
    ghost_error ghost_densemat_cm_cloneVector(ghost_densemat *src, ghost_densemat **dst, ghost_lidx nr, ghost_lidx roffs, ghost_lidx nc, ghost_lidx coffs);
    ghost_error ghost_densemat_cm_compress(ghost_densemat *vec);
    ghost_error ghost_densemat_cm_halocommInit(ghost_densemat *vec, ghost_context *ctx, ghost_densemat_halo_comm *comm);
    ghost_error ghost_densemat_cm_halocommFinalize(ghost_densemat *vec, ghost_context *ctx, ghost_densemat_halo_comm *comm);
#ifdef __cplusplus
}
#endif

#endif
