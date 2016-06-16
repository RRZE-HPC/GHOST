#ifndef RCM_DISSECTION_H
#define RCM_DISSECTION_H

#include "sparsemat.h"

#ifdef __cplusplus

extern "C" {
#endif

ghost_error ghost_rcm_dissect(ghost_sparsemat *mat);

#ifdef __cplusplus
}
#endif

#endif
