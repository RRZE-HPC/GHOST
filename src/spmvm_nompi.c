#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/log.h"
#include "ghost/instr.h"
#include "ghost/mat.h"

ghost_error_t ghost_spmv_nompi(ghost_context_t *context, ghost_vec_t* res, ghost_mat_t* mat, ghost_vec_t* invec, int spmvmOptions)
{
    if (context == NULL) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }

    GHOST_INSTR_START(spmvm_nompi);
    
    ghost_error_t err = mat->spmv(mat,res,invec,spmvmOptions);
    
    GHOST_INSTR_STOP(spmvm_nompi);

    return err;
}
