#include "ghost/sparsemat.h"
#include "ghost/carp.h"

/* This file is needed because the struct initialization cannot be done in the CPP source file
 * if the Fujitsu compiler is used. 
 */


const ghost_kacz_opts GHOST_KACZ_OPTS_INITIALIZER = {
    .omega = NULL,
    .shift = NULL,
    .num_shifts = 0,
    .direction = GHOST_KACZ_DIRECTION_UNDEFINED,
    .mode = GHOST_KACZ_MODE_NORMAL,
    .best_block_size = 1,
    .normalize = GHOST_KACZ_NORMALIZE_NO,
    .scale = NULL,
    .initialized = false
};

const ghost_carp_opts GHOST_CARP_OPTS_INITIALIZER = {
    .omega = NULL,
    .shift = NULL,
    .num_shifts = 0,
    .mode = GHOST_KACZ_MODE_NORMAL,
    .best_block_size = 1,
    .normalize = GHOST_KACZ_NORMALIZE_NO,
    .scale = NULL,
    .initialized = false
};
