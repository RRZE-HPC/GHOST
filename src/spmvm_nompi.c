#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/log.h"
#include "ghost/instr.h"

void ghost_solver_nompi(ghost_context_t *context, ghost_vec_t* res, ghost_mat_t* mat, ghost_vec_t* invec, int spmvmOptions)
{
    if (context == NULL)
        return;

    GHOST_INSTR_START(spmvm_nompi);
    
    mat->spmv(mat,res,invec,spmvmOptions);
    
    GHOST_INSTR_STOP(spmvm_nompi);
}
