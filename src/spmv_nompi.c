#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/log.h"
#include "ghost/instr.h"
#include "ghost/sparsemat.h"

ghost_error_t ghost_spmv_nompi(ghost_context_t *context, ghost_densemat_t* res, ghost_sparsemat_t* mat, ghost_densemat_t* invec, int spmvmOptions)
{
    if (context == NULL) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }

    GHOST_INSTR_START(spmv_nompi);
    
    ghost_error_t err = mat->spmv(mat,res,invec,spmvmOptions);
    
    GHOST_INSTR_STOP(spmv_nompi);

    return err;
}
