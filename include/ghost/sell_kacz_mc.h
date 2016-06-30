#ifndef SELL_KACZ_MC
#define SELL_KACZ_MC

#include "sparsemat.h"
#include "densemat.h"


ghost_error ghost_kacz_mc(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts);


#endif
