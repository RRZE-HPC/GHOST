#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/log.h"
#include "ghost/instr.h"
#include "ghost/sparsemat.h"
#include "ghost/spmv_solvers.h"

ghost_error_t ghost_spmv_nompi(ghost_densemat_t* res, ghost_sparsemat_t* mat, ghost_densemat_t* invec, ghost_spmv_flags_t flags, va_list argp)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t err = mat->spmv(mat,res,invec,flags,argp);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return err;
}
