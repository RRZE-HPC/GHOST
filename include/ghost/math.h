#ifndef GHOST_MATH_H
#define GHOST_MATH_H


#include "config.h"
#include "types.h"
#include "error.h"

#ifdef __cplusplus
#include "complex.h"

static inline ghost_complex<double> conjugate(ghost_complex<double> * c) {
    return ghost_complex<double>(c->re,-c->im);
}
static inline ghost_complex<float> conjugate(ghost_complex<float> * c) {
    return ghost_complex<float>(c->re,-c->im);
}
static inline double conjugate(double * c) {return *c;}
static inline float conjugate(float * c) {return *c;}
extern "C" {
#endif

void ghost_normalizeVec(ghost_vec_t *);
void ghost_dotProduct(ghost_vec_t *, ghost_vec_t *, void *);
ghost_error_t ghost_spmvm(ghost_context_t *context, ghost_vec_t *res, ghost_mat_t *mat, ghost_vec_t *invec, 
        int *spmvmOptions);
ghost_error_t ghost_gemm(char *, ghost_vec_t *,  ghost_vec_t *, ghost_vec_t *, void *, void *, int); 
void ghost_mpi_add_c(ghost_mpi_c *invec, ghost_mpi_c *inoutvec, int *len);
void ghost_mpi_add_z(ghost_mpi_z *invec, ghost_mpi_z *inoutvec, int *len);

#ifdef __cplusplus
} //extern "C"

#include "ghost/complex.h"

ghost_complex<float> conjugate(ghost_complex<float> * c);
ghost_complex<double> conjugate(ghost_complex<double> * c);
double conjugate(double * c);
float conjugate(float * c);
#endif

#endif

