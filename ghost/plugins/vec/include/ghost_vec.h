#ifndef __GHOST_VEC_H__
#define __GHOST_VEC_H__

#include "ghost.h"

#ifdef MIC
//#define BJDS_LEN 8
#define VEC_PAD 16
#elif defined (AVX)
#define VEC_PAD 4 // TODO single/double precision
#elif defined (SSE)
#define VEC_PAD 2
#elif defined (OPENCL) || defined (CUDA)
#define VEC_PAD 256
#elif defined (VSX)
#define VEC_PAD 2
#else
#define VEC_PAD 16
#endif

ghost_vec_t * init(ghost_vtraits_t *);


#endif
