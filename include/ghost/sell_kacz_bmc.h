#ifndef SELL_KACZ_BMC
#define SELL_KACZ_BMC

#include "sparsemat.h"
#include "densemat.h"

ghost_error ghost_initialize_kacz_bmc(ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts);
ghost_error ghost_kacz_bmc(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts);

#endif
