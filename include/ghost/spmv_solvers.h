#ifndef __GHOST_SPMV_SOLVERS_H__
#define __GHOST_SPMV_SOLVERS_H__

#ifdef __cpluscplus
extern "C" {
#endif

    ghost_error_t ghost_spmv_vectormode(ghost_densemat_t* res, ghost_sparsemat_t* mat, ghost_densemat_t* invec, ghost_spmv_flags_t flags, va_list argp);
    ghost_error_t ghost_spmv_goodfaith(ghost_densemat_t* res, ghost_sparsemat_t* mat, ghost_densemat_t* invec, ghost_spmv_flags_t flags, va_list argp);
    ghost_error_t ghost_spmv_taskmode(ghost_densemat_t* res, ghost_sparsemat_t* mat, ghost_densemat_t* invec, ghost_spmv_flags_t flags, va_list argp);
    ghost_error_t ghost_spmv_nompi(ghost_densemat_t* res, ghost_sparsemat_t* mat, ghost_densemat_t* invec, ghost_spmv_flags_t flags, va_list argp);

#ifdef __cpluscplus
}
#endif
#endif
