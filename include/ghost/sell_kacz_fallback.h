#ifndef KACZ_FALLBACK
#define KACZ_FALLBACK

#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/instr.h"
#include "ghost/omp.h"
#include "ghost/machine.h"
#include "ghost/math.h"
#include "ghost/sparsemat.h"
#include "ghost/densemat.h"
#include "ghost/locality.h"

ghost_error ghost_kacz_fallback(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts);

#endif

