#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/log.h"
#include "ghost/instr.h"
#include "ghost/sparsemat.h"
#include "ghost/spmv_solvers.h"

ghost_error ghost_spmv_nompi(ghost_densemat* res, ghost_sparsemat* mat, ghost_densemat* invec, ghost_spmv_flags flags, va_list argp)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error err = mat->spmv(mat,res,invec,flags,argp);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return err;
}
