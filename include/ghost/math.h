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
#include "dot.h"
#include "funcptr_wrappers.h"

#include <stdarg.h>

typedef enum {
    GHOST_GEMM_DEFAULT = 0,
    /**
     * @brief Do _not_ look for special implementations!
     */
    GHOST_GEMM_NOT_SPECIAL = 1,
    GHOST_GEMM_NOT_CLONE_ALIASED = 2,
    GHOST_GEMM_KAHAN = 4
} ghost_gemm_flags;

#ifdef __cplusplus
inline ghost_gemm_flags operator&=(ghost_gemm_flags &a, const ghost_gemm_flags &b) {
    a = static_cast<ghost_gemm_flags>(static_cast<int>(a) & static_cast<int>(b));
    return a;
}
inline ghost_gemm_flags operator~(const ghost_gemm_flags &a) {
    return static_cast<ghost_gemm_flags>(~static_cast<int>(a));
}
#endif

typedef struct {
    /* C = alpha(A*B) + beta(C)
     * C is mxn
     * A is mxk
     * B is kxn
     */ 
    ghost_gidx m,n,k;
    bool alphaisone;
    bool betaiszero;
    /**
     * @brief True if A is equal to C.
     */
    bool aisc;
    ghost_datatype dt;
}
ghost_gemm_perf_args;

#define GHOST_GEMM_ALL_REDUCE GHOST_ALLREDUCE
#define GHOST_GEMM_NO_REDUCE -2

typedef struct {
    ghost_lidx vecncols;
    ghost_lidx globalrows;
    ghost_gidx globalnnz;
    ghost_datatype dt;
}
ghost_kacz_perf_args;
#define GHOST_KACZ_PERF_UNIT "GF/s"
#define GHOST_KACZ_PERF_TAG "kacz"

typedef struct {
    ghost_lidx vecncols;
    ghost_lidx globalrows;
    ghost_gidx globalnnz;
    ghost_datatype dt;
    ghost_spmv_flags flags;
}
ghost_spmv_perf_args;
#define GHOST_SPMV_PERF_UNIT "GF/s"
#define GHOST_SPMV_PERF_TAG "spmv"

typedef struct {
    ghost_lidx ncols;
    ghost_gidx globnrows;
    ghost_datatype dt;
}
ghost_axpy_perf_args;
#define GHOST_AXPY_PERF_UNIT "GB/s"
#define GHOST_AXPY_PERF_TAG "axpy"

typedef struct {
    ghost_lidx ncols;
    ghost_gidx globnrows;
    ghost_datatype dt;
}
ghost_axpby_perf_args;
#define GHOST_AXPBY_PERF_UNIT "GB/s"
#define GHOST_AXPBY_PERF_TAG "axpby"

typedef struct {
    ghost_lidx ncols;
    ghost_gidx globnrows;
    ghost_datatype dt;
}
ghost_axpbypcz_perf_args;
#define GHOST_AXPBYPCZ_PERF_UNIT "GB/s"
#define GHOST_AXPBYPCZ_PERF_TAG "axpbypcz"

typedef struct {
    ghost_lidx ncols;
    ghost_gidx globnrows;
    ghost_datatype dt;
    bool samevec;
}
ghost_dot_perf_args;
#define GHOST_DOT_PERF_UNIT "GB/s"
#define GHOST_DOT_PERF_TAG "dot"

typedef struct {
    ghost_lidx ncols;
    ghost_gidx globnrows;
    ghost_datatype dt;
}
ghost_scale_perf_args;
#define GHOST_SCALE_PERF_UNIT "GB/s"
#define GHOST_SCALE_PERF_TAG "scale"


#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @ingroup globops
     *
     * @brief Multiply a sparse matrix with a dense vector.
     *
     * @param res The result vector.
     * @param mat The sparse matrix.
     * @param invec The right hand side vector.
     * @param opts Configuration options for the spMV operation.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     *
     * In the most general case, this function computes the operation \f$y = \alpha (A - \gamma I) x + \beta y\f$.
     * If required  by the operation, \f$\alpha\f$ , \f$\beta\f$, \f$\gamma\f$, \f$\delta\f$, \f$\eta\f$, \a dot, and \a z have 
     * to be given in the \a traits struct (z has to be pointer to a ghost_densemat, everyhing else are pointers to variables of the same type as the vector's data).
     *
     * Application of the scaling factor \f$\alpha\f$ can be switched on by setting ::GHOST_SPMV_SCALE in the flags.
     * Otherwise, \f$\alpha=1\f$.
     *
     * The scaling factor \f$\beta\f$ can be enabled by setting ::GHOST_SPMV_AXPBY in the flags.
     * The flag ::GHOST_SPMV_AXPY sets \f$\beta\f$ to a fixed value of 1 which is a very common case.
     * 
     * \f$\gamma\f$ will be evaluated if the flags contain ::GHOST_SPMV_SHIFT or ::GHOST_SPMV_VSHIFT.
     *
     * In case ::GHOST_SPMV_DOT, ::GHOST_SPMV_DOT_YY, ::GHOST_SPMV_DOT_XY, or ::GHOST_SPMV_DOT_XX are set, 
     * \a dot has to point to a memory destination with the size (3 * "number of vector columns" * "sizeof(vector entry))".
     * Column-wise dot products \f$y^Hy, x^Hy, x^Hx\f$ will be computed and stored to this location.
     *
     * This operation maybe changed with an additional AXPBY operation on the vector z: \f$z = \delta z + \eta y\f$
     * If this should be done, ::GHOST_SPMV_CHAIN_AXPBY has to be set in the flags.
     */
    ghost_error ghost_spmv(ghost_densemat *res, ghost_sparsemat *mat, ghost_densemat *invec, ghost_spmv_opts opts);
    ghost_error ghost_spmv_nocomm(ghost_densemat *res, ghost_sparsemat *mat, ghost_densemat *invec, ghost_spmv_opts opts);
    ghost_error ghost_gemm_valid(ghost_densemat *x, ghost_densemat *v, const char * transv, 
    ghost_densemat *w, const char *transw, void *alpha, void *beta, int reduce,ghost_context *ctx,ghost_gemm_flags flags, int printerror); 
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
     * @param ctx The context in which the reduction should be done if it is enabled.
     * @param flags
     *
     * @return 
     */
    ghost_error ghost_gemm(ghost_densemat *x, ghost_densemat *v, const char *transv, ghost_densemat *w, const char * transw, void *alpha, void *beta, int reduce,ghost_context *ctx,ghost_gemm_flags flags); 
    ghost_error ghost_mpi_operations_create();
    ghost_error ghost_mpi_operations_destroy();
    ghost_error ghost_mpi_op_densemat_sum(ghost_mpi_op * op, ghost_datatype datatype);
    ghost_error ghost_mpi_op_sum(ghost_mpi_op * op, ghost_datatype datatype);
    
    ghost_error ghost_spmv_nflops(int *nFlops, ghost_datatype m_t, ghost_datatype v_t);


    /**
     * @ingroup locops
     * @brief Computes <em>y := a*x + y</em> with scalar a
     * @param y The in-/output densemat
     * @param x The input densemat
     * @param a Points to the scale factor.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_axpy(ghost_densemat *y, ghost_densemat *x, void *a);
    /**
     * @ingroup locops
     * @brief Computes column-wise <em>y := a_i*x + y</em> with separate scalar a_i
     * @param y The in-/output densemat
     * @param x The input densemat
     * @param a Points to the scale factors a. Length must be number of densemat columns.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_vaxpy(ghost_densemat *y, ghost_densemat *x, void *a);
    /**
     * @ingroup locops
     * @brief Computes <em>y := a*x + b*y</em> with scalar a and b
     * @param y The in-/output densemat.
     * @param x The input densemat
     * @param a Points to the scale factor a.
     * @param b Points to the scale factor b.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_axpby(ghost_densemat *y, ghost_densemat *x, void *a, void *b);
    /**
     * @ingroup locops
     * @brief Computes column-wise <em>y := a_i*x + b_i*y</em> with separate scalar a_i and b_i
     * @param y The in-/output densemat.
     * @param x The input densemat
     * @param a Points to the scale factors a. Length must be number of densemat columns.
     * @param b Points to the scale factors b. Length must be number of densemat columns.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_vaxpby(ghost_densemat *y, ghost_densemat *x, void *a, void *b);
    /**
     * @ingroup locops
     * @brief Computes <em>y := a*x + b*y + c*z</em> with scalar a, b, and c
     * @param y The in-/output densemat.
     * @param x The input densemat x
     * @param z The input densemat z
     * @param a Points to the scale factor a.
     * @param b Points to the scale factor b.
     * @param c Points to the scale factor c.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_axpbypcz(ghost_densemat *y, ghost_densemat *x, void *a, void *b, ghost_densemat *z, void *c);
    /**
     * @ingroup locops
     * @brief Computes column-wise <em>y := a_i*x + b_i*y + c_i*z</em> with separate scalars a_i, b_i, and c_i
     * @param y The in-/output densemat.
     * @param x The input densemat x
     * @param z The input densemat z
     * @param a Points to the scale factors a. Length must be number of densemat columns.
     * @param b Points to the scale factors b. Length must be number of densemat columns.
     * @param c Points to the scale factors c. Length must be number of densemat columns.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_vaxpbypcz(ghost_densemat *y, ghost_densemat *x, void *a, void *b, ghost_densemat *z, void *c); 

    /**
     * @ingroup locops
     * @brief Computes <em>x := s*x</em> with scalar s
     * @param x The densemat.
     * @param s The scale factor.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_scale(ghost_densemat *x, void *s);
    /**
     * @ingroup locops
     * @brief Computes column-wise <em>x := s_i*x</em> with separate scalars s_i
     * @param x The densemat.
     * @param s The scale factors. Length must be number of densemat columns.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_vscale(ghost_densemat *x, void *s);
    /**
     * @ingroup globops
     * @brief Normalizes a densemat (interpreted as a block vector).
     * @param x The densemat.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     *
     * This function normalizes every column of the matrix to have Euclidian norm 1.
     */
    ghost_error ghost_normalize(ghost_densemat *x);
    /**
     * @ingroup locops
     * @brief Conjugates a densemat.
     * @param x The densemat.
     * @return ::GHOST_SUCCESS on success or an error indicator.
     *
     * This function does nothing for real-valued densemats.
     */
    ghost_error ghost_conj(ghost_densemat *x);
    
    /**
     * @brief Compute the norm of a densemat: sum_i [conj(vec_i) * vec_i]^pow
     *
     * @param norm Where to store the norm. Must be a pointer to the densemat's data type.
     * @param vec The densemat.
     * @param pow The power. Must be a pointer to the densemat's data type.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_norm(void *norm, ghost_densemat *vec, void *pow);

    int ghost_kacz_perf(double *perf, double time, void *arg);
    int ghost_spmv_perf(double *perf, double time, void *arg);
    int ghost_axpy_perf(double *perf, double time, void *arg);
    int ghost_axpby_perf(double *perf, double time, void *arg);
    int ghost_scale_perf(double *perf, double time, void *arg);
    int ghost_dot_perf(double *perf, double time, void *arg);
    int ghost_axpbypcz_perf(double *perf, double time, void *arg);
    int ghost_gemm_perf_GFs(double *perf, double time, void *arg);
    int ghost_gemm_perf_GBs(double *perf, double time, void *arg);

    bool ghost_iszero(void *number, ghost_datatype dt);
    bool ghost_isone(void *vnumber, ghost_datatype dt);

#ifdef __cplusplus
} //extern "C"

#endif

#endif

