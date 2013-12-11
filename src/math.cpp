#include <ghost_config.h>
#include <ghost_types.h>
#include <ghost_complex.h>
#include <ghost_math.h>

template <typename T> 
ghost_complex<T> conjugate(ghost_complex<T> * c) {
    return ghost_complex<T>(c->re,-c->im);
}
double conjugate(double * c) {return *c;}
float conjugate(float * c) {return *c;}
