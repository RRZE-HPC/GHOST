/**
 * @file ghost.h
 * @brief Includes the most relevant GHOST headers for applications.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_H
#define GHOST_H

#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/core.h"
#include "ghost/sparsemat.h"
#include "ghost/densemat.h"
#include "ghost/context.h"
#include "ghost/task.h"
#include "ghost/math.h"
#include "ghost/funcptr_wrappers.h"
#include "ghost/timing.h"
#include "ghost/locality.h"
#include "ghost/rand.h"
#include "ghost/instr.h"
#include "ghost/machine.h"
#include "ghost/pumap.h"
#include "ghost/datatransfers.h"
#include "ghost/tsmttsm.h"
#include "ghost/tsmm.h"
#include "ghost/matrixmarket.h"

#include "ghost/rcm_dissection.h"

#include <hwloc.h>

#endif
