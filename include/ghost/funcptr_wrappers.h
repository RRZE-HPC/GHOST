/**
 * @file funcptr_wrappers.h
 * @brief Wrappers for functions which are stored as pointers in structs 
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 *
 * The only purpose of those functions is to enable nicer looking application code.
 */

#ifndef GHOST_FUNCPTR_WRAPPERS_H
#define GHOST_FUNCPTR_WRAPPERS_H

#include "ghost/densemat.h"
#include "ghost/sparsemat.h"

/**
 * @ingroup locops
 * @brief Computes <em>y := a*x + y</em> with scalar a
 * @param y The in-/output densemat
 * @param x The input densemat
 * @param a Points to the scale factor.
 * @return ::GHOST_SUCCESS on success or an error indicator.
 *
 * This function is just a wrapper to ghost_densemat::axpy.
 */
static inline ghost_error ghost_axpy(ghost_densemat *y, ghost_densemat *x, void *a) 
{ 
    return y->axpy(y,x,a); 
}

/**
 * @ingroup locops
 * @brief Computes <em>y := a*x + b*y</em> with scalar a and b
 * @param y The in-/output densemat.
 * @param x The input densemat
 * @param a Points to the scale factor a.
 * @param b Points to the scale factor b.
 * @return ::GHOST_SUCCESS on success or an error indicator.
 *
 * This function is just a wrapper to ghost_densemat::axpby.
 */
static inline ghost_error ghost_axpby(ghost_densemat *y, ghost_densemat *x, void *a, void *b) 
{ 
    return y->axpby(y,x,a,b); 
}

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
 *
 * This function is just a wrapper to ghost_densemat::axpbypcz.
 */
static inline ghost_error ghost_axpbypcz(ghost_densemat *y, ghost_densemat *x, void *a, void *b, ghost_densemat *z, void *c) 
{ 
    return y->axpbypcz(y,x,a,b,z,c); 
}

/**
 * @ingroup locops
 * @brief Computes column-wise <em>y := a_i*x + y</em> with separate scalar a_i
 * @param y The in-/output densemat
 * @param x The input densemat
 * @param a Points to the scale factors a. Length must be number of densemat columns.
 * @return ::GHOST_SUCCESS on success or an error indicator.
 *
 * This function is just a wrapper to ghost_densemat::vaxpy.
 */
static inline ghost_error ghost_vaxpy(ghost_densemat *y, ghost_densemat *x, void *a) 
{ 
    return y->vaxpy(y,x,a); 
}

/**
 * @ingroup locops
 * @brief Computes column-wise <em>y := a_i*x + b_i*y</em> with separate scalar a_i and b_i
 * @param y The in-/output densemat.
 * @param x The input densemat
 * @param a Points to the scale factors a. Length must be number of densemat columns.
 * @param b Points to the scale factors b. Length must be number of densemat columns.
 * @return ::GHOST_SUCCESS on success or an error indicator.
 *
 * This function is just a wrapper to ghost_densemat::vaxpby.
 */
static inline ghost_error ghost_vaxpby(ghost_densemat *y, ghost_densemat *x, void *a, void *b) 
{ 
    return y->vaxpby(y,x,a,b); 
}

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
 *
 * This function is just a wrapper to ghost_densemat::vaxpbypcz.
 */
static inline ghost_error ghost_vaxpbypcz(ghost_densemat *y, ghost_densemat *x, void *a, void *b, ghost_densemat *z, void *c) 
{ 
    return y->vaxpbypcz(y,x,a,b,z,c); 
}

/**
 * @ingroup locops
 * @brief Computes <em>x := s*x</em> with scalar s
 * @param x The densemat.
 * @param s The scale factor.
 * @return ::GHOST_SUCCESS on success or an error indicator.
 *
 * This function is just a wrapper to ghost_densemat::scale.
 */
static inline ghost_error ghost_scale(ghost_densemat *x, void *s)
{ 
    return x->scale(x,s); 
}

/**
 * @ingroup locops
 * @brief Computes column-wise <em>x := s_i*x</em> with separate scalars s_i
 * @param x The densemat.
 * @param s The scale factors. Length must be number of densemat columns.
 * @return ::GHOST_SUCCESS on success or an error indicator.
 *
 * This function is just a wrapper to ghost_densemat::vscale.
 */
static inline ghost_error ghost_vscale(ghost_densemat *x, void *s)
{ 
    return x->vscale(x,s); 
}

/**
 * @ingroup globops
 * @brief Normalizes a densemat (interpreted as a block vector).
 * @param x The densemat.
 * @param mpicomm The communicator if a global normalization should be done.
 * @return ::GHOST_SUCCESS on success or an error indicator.
 *
 * This function normalizes every column of the matrix to have Euclidian norm 1.
 * This function is just a wrapper to ghost_densemat::normalize.
 */
static inline ghost_error ghost_normalize(ghost_densemat *x, ghost_mpi_comm mpicomm)
{ 
    return x->normalize(x,mpicomm); 
}

/**
 * @ingroup locops
 * @brief Conjugates a densemat.
 * @param x The densemat.
 * @return ::GHOST_SUCCESS on success or an error indicator.
 *
 * This function does nothing for real-valued densemats.
 * This function is just a wrapper to ghost_densemat::normalize.
 */
static inline ghost_error ghost_conj(ghost_densemat *x)
{ 
    return x->conj(x); 
}

/**
 * @ingroup denseinit
 * @brief Initializes a densemat from random values.
 * @param x The densemat.
 * @return ::GHOST_SUCCESS on success or an error indicator.
 */
static inline ghost_error ghost_densemat_init_rand(ghost_densemat *x)
{
    return x->fromRand(x);
}

/**
 * @ingroup denseinit
 * @brief Initializes a densemat from a scalar value.
 * @param x The densemat.
 * @return ::GHOST_SUCCESS on success or an error indicator.
 */
static inline ghost_error ghost_densemat_init_val(ghost_densemat *x, void *v)
{
    return x->fromScalar(x,v);
}

/**
 * @ingroup denseinit
 * @brief Initializes a densemat from a given callback function.
 * @param x The densemat.
 * @param func The callback function pointer. 
 * @param arg The argument which should be forwarded to the callback.
 * @return ::GHOST_SUCCESS on success or an error indicator.
 */
static inline ghost_error ghost_densemat_init_func(ghost_densemat *x, ghost_densemat_srcfunc func, void *arg)
{
    return x->fromFunc(x,func,arg);
}

/**
 * @ingroup denseinit
 * @brief Initializes a densemat from another densemat at a given column and row offset.
 * @param x The densemat.
 * @param y The source.
 * @param roffs The first row to clone.
 * @param coffs The first column to clone.
 * @return ::GHOST_SUCCESS on success or an error indicator.
 */
static inline ghost_error ghost_densemat_init_densemat(ghost_densemat *x, ghost_densemat *y, ghost_lidx roffs, ghost_lidx coffs)
{
    return x->fromVec(x,y,roffs,coffs);
}

/**
 * @ingroup denseinit
 * @brief Initializes a densemat from a file.
 * @param x The densemat.
 * @param path Path to the file.
 * @param mpicomm If equal to MPI_COMM_SELF, each process will read from a separate file.
 * Else, a combined file will be read with MPI I/O.
 * @return ::GHOST_SUCCESS on success or an error indicator.
 */
static inline ghost_error ghost_densemat_init_file(ghost_densemat *x, char *path, ghost_mpi_comm mpicomm)
{
    return x->fromFile(x,path,mpicomm);
}

/**
 * @ingroup denseinit
 * @brief Initializes a complex densemat from two real ones (one holding the real, the other one the imaginary part).
 * @param vec The densemat.
 * @param re The real source densemat.
 * @param im The imaginary source densemat.
 * @return ::GHOST_SUCCESS on success or an error indicator.
 */
static inline ghost_error ghost_densemat_init_real(ghost_densemat *vec, ghost_densemat *re, ghost_densemat *im)
{
    return vec->fromReal(vec,re,im);
}

/**
 * @ingroup denseinit
 * @brief Initializes two real densemats from a complex one.
 * @param re The resulting real densemat holding the real part of the source.
 * @param im The resulting real densemat holding the imaginary part of the source.
 * @param src The complex source densemat.
 * @return ::GHOST_SUCCESS on success or an error indicator.
 */
static inline ghost_error ghost_densemat_init_complex(ghost_densemat *re, ghost_densemat *im, ghost_densemat *src)
{
    return re->fromComplex(re,im,src);
}

/**
 * @ingroup denseinit
 * @ingroup denseview
 * @brief View plain data which is stored with a given stride 
 * @param x The densemat.
 * @param data Memory location of the data.
 * @param stride Stride of the data.
 * @return ::GHOST_SUCCESS on success or an error indicator.
 */
static inline ghost_error ghost_densemat_view_plain(ghost_densemat *x, void *data, ghost_lidx stride)
{
    return x->viewPlain(x,data,stride);
}

/**
 * @ingroup denseinit
 * @ingroup denseview
 * @brief Create a ghost_densemat as a view of compact data of another ghost_densemat
 * @param x The resulting scattered view.
 * @param src The source densemat with the data to be viewed.
 * @param nr The number of rows of the new densemat.
 * @param roffs The row offset into the source densemat.
 * @param nc The number of columsn of the new densemat.
 * @param coffs The column offset into the source densemat.
 * @return ::GHOST_SUCCESS on success or an error indicator.
 */
static inline ghost_error ghost_densemat_create_and_view_densemat(ghost_densemat **x, ghost_densemat *src, ghost_lidx nr, ghost_lidx roffs, ghost_lidx nc, ghost_lidx coffs)
{
    return src->viewVec(src,x,nr,roffs,nc,coffs);
}

/**
 * @ingroup denseinit
 * @ingroup denseview
 * @brief Create a ghost_densemat as a view of arbitrarily scattered data of another ghost_densemat
 * @param x The resulting scattered view.
 * @param src The source densemat with the data to be viewed.
 * @param nr The number of rows of the new densemat.
 * @param ridx The row indices to be viewed.
 * @param nc The number of columsn of the new densemat.
 * @param cidx The column indices to be viewed.
 * @return ::GHOST_SUCCESS on success or an error indicator.
 */
static inline ghost_error ghost_densemat_create_and_view_densemat_scattered(ghost_densemat **x, ghost_densemat *src, ghost_lidx nr, ghost_lidx *ridx, ghost_lidx nc, ghost_lidx *cidx)
{
    return src->viewScatteredVec(src,x,nr,ridx,nc,cidx);
}

/**
 * @ingroup denseinit
 * @ingroup denseview
 * @brief Create a ghost_densemat as a view of compact columns of another ghost_densemat
 * @param x The resulting scattered view.
 * @param src The source densemat with the data to be viewed.
 * @param nc The number of columsn of the new densemat.
 * @param coffs The column offset into the source densemat.
 * @return ::GHOST_SUCCESS on success or an error indicator.
 */
static inline ghost_error ghost_densemat_create_and_view_densemat_cols(ghost_densemat **x, ghost_densemat *src, ghost_lidx nc, ghost_lidx coffs)
{
    return src->viewCols(src,x,nc,coffs);
}

/**
 * @ingroup denseinit
 * @ingroup denseview
 * @brief Create a ghost_densemat as a view of full but scattered columns of another ghost_densemat
 * @param x The resulting scattered view.
 * @param src The source densemat with the data to be viewed.
 * @param nc The number of columsn of the new densemat.
 * @param cidx The column indices to be viewed.
 * @return ::GHOST_SUCCESS on success or an error indicator.
 */
static inline ghost_error ghost_densemat_create_and_view_densemat_cols_scattered(ghost_densemat **x, ghost_densemat *src, ghost_lidx nc, ghost_lidx *cidx)
{
    return src->viewScatteredCols(src,x,nc,cidx);
}

/**
 * @ingroup denseinit
 * @ingroup denseview
 * @brief Create a ghost_densemat as a clone of another ghost_densemat at a given offset
 * @param x The clone.
 * @param src The source densemat.
 * @param nr The number of rows of the new densemat.
 * @param roffs The row offset into the source densemat.
 * @param nc The number of columsn of the new densemat.
 * @param coffs The column offset into the source densemat.
 * @return ::GHOST_SUCCESS on success or an error indicator.
 */
static inline ghost_error ghost_densemat_clone(ghost_densemat **x, ghost_densemat *src, ghost_lidx nr, ghost_lidx roffs, ghost_lidx nc, ghost_lidx coffs)
{
    return src->clone(src,x,nr,roffs,nc,coffs);
}

/**
 * @ingroup stringification
 * @brief Creates a string of the densemat's contents.
 * @param x The densemat.
 * @param str Where to store the string.
 * @return ::GHOST_SUCCESS on success or an error indicator.
 *
 * The string has to be freed by the caller.
 */
static inline ghost_error ghost_densemat_string(char **str, ghost_densemat *x)
{
    return x->string(x,str);
}

/**
 * @brief Permute a densemat in a given direction.
 * @param x The densemat.
 * @param ctx The context if a global permutation is present.
 * @param dir The permutation direction.
 * @return ::GHOST_SUCCESS on success or an error indicator.
 */
static inline ghost_error ghost_densemat_permute(ghost_densemat *x, ghost_context *ctx, ghost_permutation_direction dir)
{
    return x->permute(x,ctx,dir);
}



#endif
