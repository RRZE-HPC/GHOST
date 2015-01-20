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
#include "tsmm.h"
#include "tsmm_inplace.h"
#include "tsmttsm.h"

#include <stdarg.h>

typedef enum {
    GHOST_GEMM_DEFAULT = 0,
    /**
     * @brief Do _not_ look for special implementations!
     */
    GHOST_GEMM_NOT_SPECIAL = 1,
} ghost_gemm_flags_t;

#define GHOST_GEMM_ALL_REDUCE -1
#define GHOST_GEMM_NO_REDUCE -2

typedef struct {
    ghost_sparsemat_t *mat;
    ghost_densemat_t *rhs;
    ghost_spmv_flags_t flags;
}
ghost_spmv_perf_args_t;
#define GHOST_SPMV_PERF_UNIT "GF/s"
#define GHOST_SPMV_PERF_TAG "spmv"

typedef struct {
    ghost_densemat_t *vec1;
    ghost_densemat_t *vec2;
}
ghost_axpy_perf_args_t;
#define GHOST_AXPY_PERF_UNIT "GB/s"
#define GHOST_AXPY_PERF_TAG "axpy"

typedef struct {
    ghost_densemat_t *vec1;
    ghost_densemat_t *vec2;
}
ghost_dot_perf_args_t;
#define GHOST_DOT_PERF_UNIT "GB/s"
#define GHOST_DOT_PERF_TAG "dot"

typedef struct {
    ghost_densemat_t *vec;
}
ghost_scale_perf_args_t;
#define GHOST_SCALE_PERF_UNIT "GB/s"
#define GHOST_SCALE_PERF_TAG "scale"

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
     * @param res Where to store the result.
     * @param a The first vector/matrix.
     * @param b The second vector/matrix.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     *
     * This function first computes the local dot product ghost_densemat_t::dot and then performs an allreduce on the result.
     */
    ghost_error_t ghost_dot(void *res, ghost_densemat_t *a, ghost_densemat_t *b);
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
     * Application of the scaling factor \f$\alpha\f$ can be switched on by setting ::GHOST_SPMV_SCALE in the flags.
     * Otherwise, \f$\alpha=1\f$.
     *
     * The scaling factor \f$\beta\f$ can be enabled by setting ::GHOST_SPMV_AXPBY in the flags.
     * The flag ::GHOST_SPMV_AXPY sets \f$\beta\f$ to a fixed value of 1 which is a very common case.
     * 
     * \f$\gamma\f$ will be evaluated if the flags contain ::GHOST_SPMV_SHIFT or ::GHOST_SPMV_VSHIFT.
     *
     * In case ::GHOST_SPMV_DOT is set, \a dot has to point to a memory destination of the size
     * of three vector values.
     *
     * \warning If there is something wrong with the variadic arguments, i.e., if (following from the flags) more arguments are expected than present, random errors may occur. In order to avoid this, passing NULL as the last argument is a good practice.
     *
     */
    ghost_error_t ghost_spmv(ghost_densemat_t *res, ghost_sparsemat_t *mat, ghost_densemat_t *invec, ghost_spmv_flags_t *flags, ...);
    /**
     * @ingroup globops
     *
     * @brief Compute the general (dense) matrix-matrix product x = v*w
     *
     * @param x
     * @param v
     * @param transv
     * @param w
     * @param transw
     * @param alpha
     * @param beta
     * @param reduce
     * @param flags
     *
     * @return 
     */
    ghost_error_t ghost_gemm(ghost_densemat_t *x, ghost_densemat_t *v, const char *transv, ghost_densemat_t *w, const char * transw, void *alpha, void *beta, int reduce,ghost_gemm_flags_t flags); 
    ghost_error_t ghost_mpi_operations_create();
    ghost_error_t ghost_mpi_operations_destroy();
    ghost_error_t ghost_mpi_op_sum(ghost_mpi_op_t * op, int datatype);
    
    ghost_error_t ghost_spmv_nflops(int *nFlops, ghost_datatype_t m_t, ghost_datatype_t v_t);
    /**
     * @ingroup stringification
     *
     * @brief Get a string about the SpMV solver mode. 
     *
     * @param flags The SpMV flags.
     *
     * @return The string.
     */
    char * ghost_spmv_mode_string(ghost_spmv_flags_t flags);

    int ghost_spmv_perf(double *perf, double time, void *arg);
    int ghost_axpy_perf(double *perf, double time, void *arg);
    int ghost_scale_perf(double *perf, double time, void *arg);
    int ghost_dot_perf(double *perf, double time, void *arg);

#ifdef __cplusplus
} //extern "C"

#include "ghost/complex.h"

ghost_complex<float> conjugate(ghost_complex<float> * c);
ghost_complex<double> conjugate(ghost_complex<double> * c);
double conjugate(double * c);
float conjugate(float * c);
#endif

#endif

