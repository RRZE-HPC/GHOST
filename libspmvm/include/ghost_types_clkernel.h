#ifndef __GHOST_TYPES_CLKERNEL_H__
#define __GHOST_TYPES_CLKERNEL_H__

#include "ghost_types_gen.h"

#ifdef GHOST_MAT_DP
#ifdef GHOST_MAT_COMPLEX
typedef double2 ghost_cl_mdat_t;
#else
typedef double ghost_cl_mdat_t;
#endif
#endif

#ifdef GHOST_MAT_SP
#ifdef GHOST_MAT_COMPLEX
typedef float2 ghost_cl_mdat_t;
#else
typedef float ghost_cl_mdat_t;
#endif
#endif

#ifdef GHOST_VEC_DP
#ifdef GHOST_VEC_COMPLEX
typedef double2 ghost_cl_vdat_t;
#else
typedef double ghost_cl_vdat_t;
#endif
#endif

#ifdef GHOST_VEC_SP
#ifdef GHOST_VEC_COMPLEX
typedef float2 ghost_cl_vdat_t;
#else
typedef float ghost_cl_vdat_t;
#endif
#endif


#endif
