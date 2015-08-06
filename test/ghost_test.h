#ifndef _GHOST_TEST_H
#define _GHOST_TEST_H

#include <math.h>


#define D_EPS 1.e-14
#define S_EPS 1.e-6

#define RETURN_IF_DIFFER(a,b,f,t) \
        if (t & GHOST_DT_REAL) {\
            if (t & GHOST_DT_DOUBLE) {\
                double return_if_differ_err = fabs(*((double *)a) - *((double *)b));\
                if (return_if_differ_err > D_EPS*f) {\
                    ERROR_LOG("Values differ: error %8.4e > %8.4e", return_if_differ_err, D_EPS*f);\
                    return EXIT_FAILURE;\
                }\
            } else {\
                float return_if_differ_err = fabs(*((float *)a) - *((float *)b));\
                if (return_if_differ_err > S_EPS*f) {\
                    ERROR_LOG("Values differ: error %8.4e > %8.4e", return_if_differ_err, S_EPS*f);\
                    return EXIT_FAILURE;\
                }\
            }\
        } else {\
            if (t & GHOST_DT_DOUBLE) {\
                double return_if_differ_err = fabs((*(ghost_complex<double> *)a).re - (*(ghost_complex<double> *)b).re);\
                if (return_if_differ_err > D_EPS*f) {\
                    ERROR_LOG("Values differ: real part error %8.4e > %8.4e", return_if_differ_err, D_EPS*f);\
                    return EXIT_FAILURE;\
                }\
                return_if_differ_err = fabs((*(ghost_complex<double> *)a).im - (*(ghost_complex<double> *)b).im);\
                if (return_if_differ_err > D_EPS*f) {\
                    ERROR_LOG("Values differ: imag part error %8.4e > %8.4e", return_if_differ_err, D_EPS*f);\
                    return EXIT_FAILURE;\
                }\
            } else {\
                float return_if_differ_err = fabs((*(ghost_complex<float> *)a).re - (*(ghost_complex<float> *)b).re);\
                if (return_if_differ_err > D_EPS*f) {\
                    ERROR_LOG("Values differ: real part error %8.4e > %8.4e", return_if_differ_err, D_EPS*f);\
                    return EXIT_FAILURE;\
                }\
                return_if_differ_err = fabs((*(ghost_complex<float> *)a).im - (*(ghost_complex<float> *)b).im);\
                if (return_if_differ_err > D_EPS*f) {\
                    ERROR_LOG("Values differ: imag part error %8.4e > %8.4e", return_if_differ_err, D_EPS*f);\
                    return EXIT_FAILURE;\
                }\
            }\
        }\


#define GHOST_TEST_CALL(call) {if (call != GHOST_SUCCESS) {return EXIT_FAILURE;}}


#endif
