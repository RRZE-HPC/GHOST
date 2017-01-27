#ifndef _GHOST_TEST_H
#define _GHOST_TEST_H

#include <math.h>
#include <complex>

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
                float return_if_differ_err = fabsf(*((float *)a) - *((float *)b));\
                if (return_if_differ_err > S_EPS*f) {\
                    ERROR_LOG("Values differ: error %8.4e > %8.4e", return_if_differ_err, S_EPS*f);\
                    return EXIT_FAILURE;\
                }\
            }\
        } else {\
            if (t & GHOST_DT_DOUBLE) {\
                double return_if_differ_err = fabs(real(*(std::complex<double> *)a) - real((*(std::complex<double> *)b)));\
                if (return_if_differ_err > D_EPS*f) {\
                    ERROR_LOG("Values differ: real part error %8.4e > %8.4e", return_if_differ_err, D_EPS*f);\
                    return EXIT_FAILURE;\
                }\
                return_if_differ_err = fabs(imag(*(std::complex<double> *)a) - imag(*(std::complex<double> *)b));\
                if (return_if_differ_err > D_EPS*f) {\
                    ERROR_LOG("Values differ: imag part error %8.4e > %8.4e", return_if_differ_err, D_EPS*f);\
                    return EXIT_FAILURE;\
                }\
            } else {\
                float return_if_differ_err = fabsf(real(*(std::complex<float> *)a) - real(*(std::complex<float> *)b));\
                if (return_if_differ_err > D_EPS*f) {\
                    ERROR_LOG("Values differ: real part error %8.4e > %8.4e", return_if_differ_err, D_EPS*f);\
                    return EXIT_FAILURE;\
                }\
                return_if_differ_err = fabsf(imag(*(std::complex<float> *)a) - imag(*(std::complex<float> *)b));\
                if (return_if_differ_err > D_EPS*f) {\
                    ERROR_LOG("Values differ: imag part error %8.4e > %8.4e", return_if_differ_err, D_EPS*f);\
                    return EXIT_FAILURE;\
                }\
            }\
        }\

#endif
