#ifndef _GHOST_TEST_H
#define _GHOST_TEST_H

#include <math.h>

#define DATATYPE_D

#ifdef DATATYPE_D
#define EPS 1.e-14
GHOST_REGISTER_DT_D(mydata)
#define DIFFER(a,b,f) (fabs(a-b) > EPS*(double)f)
#endif

#define GHOST_TEST_CALL(call) {if (call != GHOST_SUCCESS) {return EXIT_FAILURE;}}


#endif
