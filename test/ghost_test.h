#ifndef _GHOST_TEST_H
#define _GHOST_TEST_H

#include <math.h>


#define D_EPS 1.e-14
#define S_EPS 1.e-6

#define RETURN_IF_DIFFER(a,b,f,t) \
        if (t & GHOST_DT_REAL) {\
            if (t & GHOST_DT_DOUBLE) {\
                if (fabs(*((double *)a) - *((double *)b)) > D_EPS*f) {\
                    return EXIT_FAILURE;\
                }\
            } else {\
                if (fabsf(*((float *)a) - *((float *)b)) > S_EPS*f) {\
                    return EXIT_FAILURE;\
                }\
            }\
        } else {\
            if (t & GHOST_DT_DOUBLE) {\
                if (fabs((*(ghost_complex<double> *)a).re - (*(ghost_complex<double> *)b).re) > D_EPS*f) {\
                    return EXIT_FAILURE;\
                }\
                if (fabs((*(ghost_complex<double> *)a).im - (*(ghost_complex<double> *)b).im) > D_EPS*f) {\
                    return EXIT_FAILURE;\
                }\
            } else {\
                if (fabsf((*(ghost_complex<float> *)a).re - (*(ghost_complex<float> *)b).re) > S_EPS*f) {\
                    return EXIT_FAILURE;\
                }\
                if (fabsf((*(ghost_complex<float> *)a).im - (*(ghost_complex<float> *)b).im) > S_EPS*f) {\
                    return EXIT_FAILURE;\
                }\
            }\
        }\


#define GHOST_TEST_CALL(call) {if (call != GHOST_SUCCESS) {return EXIT_FAILURE;}}


#endif
