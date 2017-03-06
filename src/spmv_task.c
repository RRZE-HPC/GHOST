#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/locality.h"
#include "ghost/densemat.h"
#include "ghost/task.h"
#include "ghost/taskq.h"
#include "ghost/pumap.h"
#include "ghost/machine.h"
#include "ghost/util.h"
#include "ghost/instr.h"
#include "ghost/sparsemat.h"
#include "ghost/spmv_solvers.h"
#include "ghost/math.h"

#ifdef GHOST_HAVE_MPI
#include <mpi.h>
#endif

#include <sys/types.h>
#include <string.h>

#ifdef LIKWID
#include <likwid.h>
#endif

#ifdef GHOST_HAVE_OPENMP
#include <omp.h>
#endif

#ifdef GHOST_HAVE_MPI
typedef struct {
    ghost_densemat *rhs;
    ghost_densemat_halo_comm *comm;
    ghost_context *ctx;
} commArgs;

static void *communicate(void *vargs)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
    commArgs *args = (commArgs *)vargs;
    ghost_error *ret = NULL;
    ghost_malloc((void **)&ret,sizeof(ghost_error)); // don't use macro because it would read *ret
    if (!ret) {
        goto err;
    }
    *ret = GHOST_SUCCESS;
    GHOST_CALL_GOTO(ghost_densemat_halocomm_start(args->rhs,args->ctx,args->comm),err,*ret);
    GHOST_CALL_GOTO(ghost_densemat_halocomm_finalize(args->rhs,args->ctx,args->comm),err,*ret);

    goto out;
err:

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_COMMUNICATION);
    return ret;
}

typedef struct {
    ghost_sparsemat *mat;
    ghost_densemat *res;
    ghost_densemat *invec;
    ghost_spmv_opts spmvtraits;
} compArgs;

static void *computeLocal(void *vargs)
{
//#pragma omp parallel
//    INFO_LOG("comp local t %d running @ core %d",ghost_ompGetThreadNum(),ghost_getCore());
    //GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error *ret = NULL;
    ghost_malloc((void **)&ret,sizeof(ghost_error)); // don't use macro because it would read *ret
    if (!ret) {
        goto err;
    }
    *ret = GHOST_SUCCESS;

    compArgs *args = (compArgs *)vargs;
    GHOST_CALL_GOTO(ghost_spmv_nocomm(args->res,args->mat,args->invec,args->spmvtraits),err,*ret);

    goto out;
err:
out:
    //GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);
    return ret;
}
#endif

ghost_error ghost_spmv_taskmode(ghost_densemat* res, ghost_sparsemat* mat, ghost_densemat* invec, ghost_spmv_opts traits)
{
#ifndef GHOST_HAVE_MPI
    UNUSED(res);
    UNUSED(mat);
    UNUSED(invec);
    UNUSED(traits);
    ERROR_LOG("Cannot execute this spMV solver without MPI");
    return GHOST_ERR_UNKNOWN;
#else
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    GHOST_INSTR_START("prepare");
    ghost_error ret = GHOST_SUCCESS;

    ghost_spmv_opts localtraits = traits;
    ghost_spmv_opts remotetraits = traits;

/*    int remoteExists;
    ghost_nrank(&remoteExists,mat->context->mpicomm);
    remoteExists -= 1;*/
    int remoteExists = mat->remotePart->nEnts > 0;
    //MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE,&remoteExists,1,MPI_INT,MPI_MAX,mat->context->mpicomm));
   
    if (remoteExists) {
        localtraits.flags |= (ghost_spmv_flags)GHOST_SPMV_LOCAL;
        remotetraits.flags |= (ghost_spmv_flags)GHOST_SPMV_REMOTE;
    }

    ghost_densemat_halo_comm comm = GHOST_DENSEMAT_HALO_COMM_INITIALIZER;
    commArgs cargs;
    compArgs cplargs;
    ghost_task *commTask;
    ghost_task *compTask;

    ghost_task_flags taskflags = GHOST_TASK_DEFAULT;
    ghost_task *parent = NULL;
    GHOST_CALL_RETURN(ghost_task_cur(&parent));
    if (parent) {
        DEBUG_LOG(1,"using the parent's cores for the task mode spmv solver");
        ghost_task_create(&compTask, parent->nThreads - remoteExists, 0, &computeLocal, &cplargs, taskflags, NULL, 0);
        ghost_task_create(&commTask, remoteExists, 0, &communicate, &cargs, taskflags, NULL, 0);


    } else {
        DEBUG_LOG(1,"No parent task in task mode spmv solver");

        int nIdleCores;
        ghost_pumap_nidle(&nIdleCores,GHOST_NUMANODE_ANY);
        ghost_task_create(&compTask, 2/*nIdleCores-remoteExists*/, 0, &computeLocal, &cplargs, taskflags, NULL, 0);
        ghost_task_create(&commTask, 2/*remoteExists*/, 0, &communicate, &cargs, taskflags, NULL, 0);
    }

    cargs.rhs = invec;
    cargs.comm = &comm;
    cargs.ctx = mat->context;
    cplargs.mat = mat->localPart;
    cplargs.invec = invec;
    cplargs.res = res;
    cplargs.spmvtraits = localtraits;

    GHOST_INSTR_STOP("prepare");
    
    GHOST_INSTR_START("haloassembly");
    
    GHOST_CALL_GOTO(ghost_densemat_halocomm_init(invec,mat->context,&comm),err,ret);
    
    GHOST_INSTR_STOP("haloassembly");

    GHOST_INSTR_START("both_tasks");
    if (remoteExists) {
        ghost_task_enqueue(commTask);
    }
    ghost_task_enqueue(compTask);
    ghost_task_wait(compTask);
    if ((ret = *((ghost_error *)(compTask->ret))) != GHOST_SUCCESS) {
        goto err;
    }
    if (remoteExists) {
        ghost_task_wait(commTask);
        if ((ret = *((ghost_error *)(commTask->ret))) != GHOST_SUCCESS) {
            goto err;
        }
    }
    GHOST_INSTR_STOP("both_tasks");

    GHOST_INSTR_START("remote");
    if (remoteExists) {
        ghost_spmv_nocomm(res,mat->remotePart,invec,remotetraits);
    }
    GHOST_INSTR_STOP("remote");
       
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    goto out;
err:

out:
    free(compTask->ret); compTask->ret = NULL;
    free(commTask->ret); commTask->ret = NULL;

    return ret;
#endif
}
