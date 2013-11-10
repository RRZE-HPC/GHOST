#ifndef __GHOST_H__
#define __GHOST_H__

#include <ghost_config.h>

/*
#ifndef GHOST_CLKERNEL
#include <stdlib.h>
#ifndef __cplusplus
#include <math.h>
#include <complex.h>
#else
#include <complex>
#endif
#include <sys/types.h>
#include <pthread.h>

#include <hwloc.h>

#ifdef GHOST_HAVE_OPENCL
#include <CL/cl.h>
#endif

#ifdef GHOST_HAVE_CUDA
#include <cuda.h>
#endif
#endif*/
#define GHOST_NAME "GHOST"
#define GHOST_VERSION "0.5"

#include <ghost_types.h>
#include <ghost_util.h>
#include <ghost_math.h>
#include <ghost_constants.h>
#include <ghost_taskq.h>
#include <ghost_vec.h>
#include <ghost_math.h>
#include <ghost_mat.h>
#include <ghost_context.h>


#endif
