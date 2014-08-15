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
                if (fabs((*(complex<double> *)a).real() - (*(complex<double> *)b).real()) > D_EPS*f) {\
                    return EXIT_FAILURE;\
                }\
                if (fabs((*(complex<double> *)a).imag() - (*(complex<double> *)b).imag()) > D_EPS*f) {\
                    return EXIT_FAILURE;\
                }\
            } else {\
                if (fabsf((*(complex<float> *)a).real() - (*(complex<float> *)b).real()) > S_EPS*f) {\
                    return EXIT_FAILURE;\
                }\
                if (fabsf((*(complex<float> *)a).imag() - (*(complex<float> *)b).imag()) > S_EPS*f) {\
                    return EXIT_FAILURE;\
                }\
            }\
        }\


#define GHOST_TEST_CALL(call) {if (call != GHOST_SUCCESS) {return EXIT_FAILURE;}}


#endif
