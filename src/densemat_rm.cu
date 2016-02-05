#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/densemat_rm.h"
#include "ghost/log.h"
#include "ghost/timing.h"
#include "ghost/locality.h"
#include "ghost/instr.h"
#include "ghost/rand.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <curand.h>
#include <sys/types.h>
#include <unistd.h>
#include <complex.h>

#include "ghost/cu_complex.h"


#define ROWMAJOR
#include "ghost/densemat_common.cu.def"

