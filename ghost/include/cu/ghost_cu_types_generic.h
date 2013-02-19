#ifndef __GHOST_CU_TYPES_GENERIC_H__
#define __GHOST_CU_TYPES_GENERIC_H__

#include <cuComplex.h>

#define s_ADD(a,b) a+b
#define s_MUL(a,b) a*b
#define s_ZERO 0.f
#define s_INIT(re,im) re

#define d_ADD(a,b) a+b
#define d_MUL(a,b) a*b
#define d_ZERO 0.
#define d_INIT(re,im) re

#define c_ADD(a,b) cuCaddf(a,b)
#define c_MUL(a,b) cuCmulf(a,b)
#define c_ZERO make_cuFloatComplex(0.,0.)
#define c_INIT(re,im) make_cuFloatComplex(re,im)

#define z_ADD(a,b) cuCadd(a,b)
#define z_MUL(a,b) cuCmul(a,b)
#define z_ZERO make_cuDoubleComplex(0.,0.)
#define z_INIT(re,im) make_cuDoubleComplex(re,im)


#endif
