#ifndef GHOST_MATH_H
#define GHOST_MATH_H

#include "config.h"
#include "types.h"
#include "context.h"
#include "sparsemat.h"
#include "densemat.h"
#include "error.h"

#define GHOST_GEMM_ALL_REDUCE -1
#define GHOST_GEMM_NO_REDUCE -2

typedef ghost_error_t (*ghost_spmvsolver_t)(ghost_context_t *, ghost_densemat_t*, ghost_sparsemat_t *, ghost_densemat_t*, ghost_spmv_flags_t);



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

    /**
     *
     * @ingroup globops
     * @brief Normalize a dense matrix (interpreted as a multi-vector). 
     *
     * @param mat The dense matrix.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     *
     * This function normalizes every column of the matrix to have Euclidian norm 1.
     */
    ghost_error_t ghost_normalize(ghost_densemat_t *mat);
    /**
     * @ingroup globops
     *
     * @brief Compute the global dot product of two dense vectors/matrices.
     *
     * @param a The first vector/matrix.
     * @param b The second vector/matrix.
     * @param res Where to store the result.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     *
     * This function first computes the local dot product and then performs an allreduce on the result.
     */
    ghost_error_t ghost_dot(ghost_densemat_t *a, ghost_densemat_t *b, void *res);
    ghost_error_t ghost_spmv(ghost_context_t *ctx, ghost_densemat_t *res, ghost_sparsemat_t *mat, ghost_densemat_t *invec, ghost_spmv_flags_t *flags);
    ghost_error_t ghost_spmv_vectormode(ghost_context_t *ctx, ghost_densemat_t* res, ghost_sparsemat_t* mat, ghost_densemat_t* invec, ghost_spmv_flags_t flags);
    ghost_error_t ghost_spmv_goodfaith(ghost_context_t *ctx, ghost_densemat_t* res, ghost_sparsemat_t* mat, ghost_densemat_t* invec, ghost_spmv_flags_t flags);
    ghost_error_t ghost_spmv_taskmode(ghost_context_t *ctx, ghost_densemat_t* res, ghost_sparsemat_t* mat, ghost_densemat_t* invec, ghost_spmv_flags_t flags);
    ghost_error_t ghost_spmv_nompi(ghost_context_t *ctx, ghost_densemat_t* res, ghost_sparsemat_t* mat, ghost_densemat_t* invec, ghost_spmv_flags_t flags);
    ghost_error_t ghost_spmv_nflops(int *nFlops, int m_t, int v_t);
    void          ghost_spmv_selectMode(ghost_context_t * context, int *spmvmOptions);
    /**
     * @deprecated Construct a test case with a known result instead (e.g., same as HPCCG)
     */
    ghost_error_t ghost_referenceSolver(ghost_densemat_t *, char *matrixPath, int datatype, ghost_densemat_t *rhs, int nIter, ghost_spmv_flags_t flags);
    ghost_error_t ghost_gemm(char *, ghost_densemat_t *,  ghost_densemat_t *, ghost_densemat_t *, void *, void *, int); 
    ghost_error_t ghost_mpi_createOperations();
    ghost_error_t ghost_mpi_destroyOperations();
    ghost_error_t ghost_mpi_op_sum(ghost_mpi_op_t * op, int datatype);
    char * ghost_modeName(ghost_spmv_flags_t flags);

#ifdef __cplusplus
} //extern "C"

#include "ghost/complex.h"

ghost_complex<float> conjugate(ghost_complex<float> * c);
ghost_complex<double> conjugate(ghost_complex<double> * c);
double conjugate(double * c);
float conjugate(float * c);
#endif

#endif

