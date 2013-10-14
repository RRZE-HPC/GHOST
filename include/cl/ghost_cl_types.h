#ifndef __GHOST_TYPES_CLKERNEL_H__
#define __GHOST_TYPES_CLKERNEL_H__


#ifdef GHOST_MAT_Z
#define GHOST_MAT_COMPLEX
typedef double2 ghost_cl_mdat_t;
#endif
#ifdef GHOST_MAT_D
#define GHOST_MAT_REAL
typedef double ghost_cl_mdat_t;
#endif

#ifdef GHOST_MAT_C
#define GHOST_MAT_COMPLEX
typedef float2 ghost_cl_mdat_t;
#endif
#ifdef GHOST_MAT_S
#define GHOST_MAT_REAL
typedef float ghost_cl_mdat_t;
#endif


#ifdef GHOST_VEC_Z
#define GHOST_VEC_COMPLEX
typedef double2 ghost_cl_vdat_t;
#endif
#ifdef GHOST_VEC_D
#define GHOST_VEC_REAL
typedef double ghost_cl_vdat_t;
#endif

#ifdef GHOST_VEC_C
#define GHOST_VEC_COMPLEX
typedef float2 ghost_cl_vdat_t;
#endif
#ifdef GHOST_VEC_S
#define GHOST_VEC_REAL
typedef float ghost_cl_vdat_t;
#endif
#endif
