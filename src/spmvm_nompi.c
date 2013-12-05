#include <ghost_config.h>
#include <ghost_types.h>
#include <ghost_util.h>

void ghost_solver_nompi(ghost_context_t *context, ghost_vec_t* res, ghost_mat_t* mat, ghost_vec_t* invec, int spmvmOptions)
{
    if (context == NULL)
        return;

    GHOST_INSTR_START(spmvm_nompi);
    
    mat->kernel(mat,res,invec,spmvmOptions);
    
    GHOST_INSTR_STOP(spmvm_nompi);
}
