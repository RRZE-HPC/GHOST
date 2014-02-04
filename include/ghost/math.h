#ifndef GHOST_MATH_H
#define GHOST_MATH_H

#include "config.h"
#include "types.h"
#include "context.h"
#include "mat.h"
#include "vec.h"
#include "error.h"

#define GHOST_GEMM_ALL_REDUCE -1
#define GHOST_GEMM_NO_REDUCE -2

typedef ghost_error_t (*ghost_spmvsolver_t)(ghost_context_t *, ghost_vec_t*, ghost_mat_t *, ghost_vec_t*, int);


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
/**
 * @ingroup globops
 *
 * @brief Compute the global dot product of two vectors.
 *
 * @param a The first vector.
 * @param b The second vector.
 * @param res Where to store the result.
 *
 * @return GHOST_SUCCESS on success or an error indicator.
 *
 * This function first computes the local dot product and then performs an allreduce on the result.
 */
ghost_error_t ghost_dotProduct(ghost_vec_t *a, ghost_vec_t *b, void *res);
ghost_error_t ghost_spmvm(ghost_context_t *context, ghost_vec_t *res, ghost_mat_t *mat, ghost_vec_t *invec, 
        int *spmvmOptions);
ghost_error_t ghost_spmv_vectormode(ghost_context_t *context, ghost_vec_t* res, ghost_mat_t* mat, ghost_vec_t* invec, int spmvmOptions);
ghost_error_t ghost_spmv_goodfaith(ghost_context_t *context, ghost_vec_t* res, ghost_mat_t* mat, ghost_vec_t* invec, int spmvmOptions);
ghost_error_t ghost_spmv_taskmode(ghost_context_t *context, ghost_vec_t* res, ghost_mat_t* mat, ghost_vec_t* invec, int spmvmOptions);
ghost_error_t ghost_spmv_nompi(ghost_context_t *context, ghost_vec_t* res, ghost_mat_t* mat, ghost_vec_t* invec, int spmvmOptions);
ghost_error_t ghost_referenceSolver(ghost_vec_t *, char *matrixPath, int datatype, ghost_vec_t *rhs, int nIter, int spmvmOptions);
void ghost_pickSpMVMMode(ghost_context_t * context, int *spmvmOptions);
ghost_error_t ghost_gemm(char *, ghost_vec_t *,  ghost_vec_t *, ghost_vec_t *, void *, void *, int); 
void ghost_mpi_add_c(ghost_mpi_c *invec, ghost_mpi_c *inoutvec, int *len);
void ghost_mpi_add_z(ghost_mpi_z *invec, ghost_mpi_z *inoutvec, int *len);
ghost_error_t ghost_mpi_createOperations();
ghost_error_t ghost_mpi_destroyOperations();
ghost_error_t ghost_mpi_op_sum(ghost_mpi_op_t * op, int datatype);
ghost_error_t ghost_flopsPerSpmvm(int *nFlops, int m_t, int v_t);

#ifdef __cplusplus
} //extern "C"

#include "ghost/complex.h"

ghost_complex<float> conjugate(ghost_complex<float> * c);
ghost_complex<double> conjugate(ghost_complex<double> * c);
double conjugate(double * c);
float conjugate(float * c);
#endif

#endif

