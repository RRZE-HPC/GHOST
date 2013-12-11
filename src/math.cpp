#include <ghost_config.h>
#include <ghost_types.h>
#include <ghost_complex.h>
#include <ghost_math.h>

ghost_complex<double> conjugate(ghost_complex<double> * c) {
    return ghost_complex<double>(c->re,-c->im);
}
ghost_complex<float> conjugate(ghost_complex<float> * c) {
    return ghost_complex<float>(c->re,-c->im);
}
double conjugate(double * c) {return *c;}
float conjugate(float * c) {return *c;}
