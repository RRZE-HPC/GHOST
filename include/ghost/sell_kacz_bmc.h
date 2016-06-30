#ifndef SELL_KACZ_BMC
#define SELL_KACZ_BMC

#include "sparsemat.h"
#include "densemat.h"
#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/math.h"
#include "ghost/omp.h"
#include <omp.h>
#include <math.h>


ghost_error ghost_initialize_kacz_bmc(ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts);
ghost_error ghost_kacz_bmc(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts);
ghost_error ghost_kacz_shift_bmc(ghost_densemat *x_r, ghost_densemat *x_i, ghost_sparsemat *mat, ghost_densemat *b, double sigma_r, double sigma_i, ghost_kacz_opts opts);

#endif
