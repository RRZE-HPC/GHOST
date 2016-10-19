#ifndef SELL_KACZ_RB
#define SELL_KACZ_RB

#include "sparsemat.h"
#include "densemat.h"


ghost_error ghost_initialize_kacz(ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts);
ghost_error ghost_kacz_rb(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts);
ghost_error ghost_kacz_rb_v1(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts);
ghost_error ghost_kacz_rb_v2(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts);
ghost_error ghost_kacz_rb_v3(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts);
ghost_error ghost_kacz_rb_v4(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts);
ghost_error ghost_kacz_rb_v5(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts);



#endif
