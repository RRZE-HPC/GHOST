#ifndef RCM_DISSECTION_H
#define RCM_DISSECTION_H

#include "sparsemat.h"

#ifdef __cplusplus

extern "C" {
#endif

ghost_error find_transition_zone(ghost_sparsemat *mat, int n_threads);
ghost_error checker_rcm(ghost_sparsemat *mat);


ghost_error ghost_rcm_dissect(ghost_sparsemat *mat);

#ifdef __cplusplus
}
#endif

#endif
