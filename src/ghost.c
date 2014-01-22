#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/constants.h"

#include <stddef.h>

const ghost_mtraits_t GHOST_MTRAITS_INITIALIZER = {.flags = GHOST_SPM_DEFAULT, .aux = NULL, .nAux = 0, .datatype = GHOST_BINCRS_DT_DOUBLE|GHOST_BINCRS_DT_REAL, .format = GHOST_SPM_FORMAT_CRS, .shift = NULL, .scale = NULL };
const ghost_vtraits_t GHOST_VTRAITS_INITIALIZER = {.flags = GHOST_VEC_DEFAULT, .aux = NULL, .datatype = GHOST_BINCRS_DT_DOUBLE|GHOST_BINCRS_DT_REAL, .nrows = 0, .nrowshalo = 0, .nrowspadded = 0, .nvecs = 1 };



