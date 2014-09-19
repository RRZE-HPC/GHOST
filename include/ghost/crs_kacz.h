#ifndef GHOST_CRS_KACZ_H
#define GHOST_CRS_KACZ_H
#include "error.h"
#include "sparsemat.h"
#include "densemat.h"


#ifdef __cplusplus
extern "C" {
#endif

ghost_error_t dd_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega);
ghost_error_t ds_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega);
ghost_error_t dc_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega);
ghost_error_t dz_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega);
ghost_error_t sd_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega);
ghost_error_t ss_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega);
ghost_error_t sc_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega);
ghost_error_t sz_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega);
ghost_error_t cd_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega);
ghost_error_t cs_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega);
ghost_error_t cc_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega);
ghost_error_t cz_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega);
ghost_error_t zd_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega);
ghost_error_t zs_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega);
ghost_error_t zc_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega);
ghost_error_t zz_crs_kacz(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, void *omega);

#ifdef __cplusplus
}
#endif

#endif
