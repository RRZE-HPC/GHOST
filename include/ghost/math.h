/**
 * @file math.h
 * @brief Functions for global mathematical operations.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_MATH_H
#define GHOST_MATH_H

#include "config.h"
#include "types.h"
#include "context.h"
#include "sparsemat.h"
#include "densemat.h"
#include "error.h"

#include <stdarg.h>

#define GHOST_GEMM_ALL_REDUCE -1
#define GHOST_GEMM_NO_REDUCE -2


typedef ghost_error_t (*ghost_spmvsolver_t)(ghost_densemat_t*, ghost_sparsemat_t *, ghost_densemat_t*, ghost_spmv_flags_t, va_list argp);

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
     * @return ::GHOST_SUCCESS on success or an error indicator.
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
     * @return ::GHOST_SUCCESS on success or an error indicator.
     *
     * This function first computes the local dot product ghost_densemat_t::dot and then performs an allreduce on the result.
     */
    ghost_error_t ghost_dot(ghost_densemat_t *a, ghost_densemat_t *b, void *res);
    /**
     * @ingroup globops
     *
     * @brief Multiply a sparse matrix with a dense vector.
     *
     * @param res The result vector.
     * @param mat The sparse matrix.
     * @param invec The right hand side vector.
     * @param flags Configuration flags for the spMV operation.
     * @param ... Further arguments \f$\alpha\f$ , \f$\beta\f$, \f$\gamma\f$, and \a dot (in that exact order) if configured in the flags (cf. detailed description).
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     *
     * In the most general case, this function computes the operation \f$y = \alpha (A - \gamma I) x + \beta y\f$.
     * If required  by the operation, \f$\alpha\f$ , \f$\beta\f$, \f$\gamma\f$, and \a dot have 
     * to be given in the correct order as pointers to variables of the same type as the vector's data.
     *
     * Application of the scaling factor \f$\alpha\f$ can be switched on by setting ::GHOST_SPMV_APPLY_SCALE in the flags.
     * Otherwise, \f$\alpha=1\f$.
     *
     * The scaling factor \f$\beta\f$ can be enabled by setting GHOST_SPMV_AXPBY in the flags.
     * The flag ::GHOST_SPMV_AXPY sets \f$\beta\f$ to a fixed value of 1.
     * 
     * \f$\gamma\f$ will be evaluated if the flags contain ::GHOST_SPMV_APPLY_SHIFT.
     *
     * In case ::GHOST_SPMV_COMPUTE_LOCAL_DOTPRODUCT is set, \a dot has to point to a memory destination of the size
     * of three vector values.
     *
     * \warning If there is something wrong with the variadic arguments, i.e., if (following from the flags) more arguments are expected than present, random errors may occur. In order to avoid this, passing NULL as the last argument is a good practice.
     *
     */
    ghost_error_t ghost_spmv(ghost_densemat_t *res, ghost_sparsemat_t *mat, ghost_densemat_t *invec, ghost_spmv_flags_t *flags, ...);
    /**
     * @ingroup globops
     *
     * @brief Compute the general (dense) matrix-matrix product.
     *
     * @param transpose
     * @param v
     * @param w
     * @param x
     * @param alpha
     * @param beta
     * @param reduce
     *
     * @return 
     */
    ghost_error_t ghost_gemm(char * transpose, ghost_densemat_t *v,  ghost_densemat_t *w, ghost_densemat_t *x, void *alpha, void *beta, int reduce); 
    ghost_error_t ghost_mpi_createOperations();
    ghost_error_t ghost_mpi_destroyOperations();
    ghost_error_t ghost_mpi_op_sum(ghost_mpi_op_t * op, int datatype);
    
    ghost_error_t ghost_spmv_nflops(int *nFlops, ghost_datatype_t m_t, ghost_datatype_t v_t);
    char * ghost_spmv_modeString(ghost_spmv_flags_t flags);

#ifdef __cplusplus
} //extern "C"

#include "ghost/complex.h"

ghost_complex<float> conjugate(ghost_complex<float> * c);
ghost_complex<double> conjugate(ghost_complex<double> * c);
double conjugate(double * c);
float conjugate(float * c);
#endif

#endif

