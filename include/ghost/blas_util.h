/**
 * @file blas_util.h
 * @brief Util function for BLAS calls, especially error for error handling.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_BLAS_UTIL_H
#define GHOST_BLAS_UTIL_H

#include "ghost/config.h"

int ghost_blas_err_pop();

#ifdef GHOST_HAVE_MKL
void cblas_xerbla(const char *name, const int num);
#endif

#ifdef GHOST_HAVE_GSL
void cblas_xerbla (int p, const char *rout, const char *form, ...);
#endif

#endif
