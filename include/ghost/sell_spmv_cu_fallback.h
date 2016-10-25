#ifndef GHOST_SELLSPMV_CU_FALLBACK
#define GHOST_SELLSPMV_CU_FALLBACK

#include "ghost/sparsemat.h"
#include "ghost/densemat.h"

#ifdef __cplusplus
extern "C" {
#endif

ghost_error ghost_sellspmv_cu_fallback_selector(ghost_densemat *lhs, ghost_sparsemat *mat, ghost_densemat *rhs, ghost_spmv_opts opts);

#ifdef __cplusplus
}
#endif

#endif
