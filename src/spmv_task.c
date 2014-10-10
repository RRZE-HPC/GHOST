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
    ghost_densemat_t *rhs;
    ghost_permutation_t *perm;
} commArgs;

static void *communicate(void *vargs)
{
    GHOST_FUNC_ENTRY(GHOST_FUNCTYPE_COMMUNICATION);
    commArgs *args = (commArgs *)vargs;
    ghost_error_t *ret = NULL;
    GHOST_CALL_GOTO(ghost_malloc((void **)&ret,sizeof(ghost_error_t)),err,*ret);
    *ret = GHOST_SUCCESS;
    GHOST_CALL_GOTO(ghost_spmv_haloexchange_initiate(args->rhs,args->perm,true),err,*ret);
    GHOST_CALL_GOTO(ghost_spmv_haloexchange_finalize(args->rhs),err,*ret);

    goto out;
err:

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_COMMUNICATION);
    return ret;
}

typedef struct {
    ghost_sparsemat_t *mat;
    ghost_densemat_t *res;
    ghost_densemat_t *invec;
    int spmvOptions;
    va_list argp;
} compArgs;

static void *computeLocal(void *vargs)
{
//#pragma omp parallel
//    INFO_LOG("comp local t %d running @ core %d",ghost_ompGetThreadNum(),ghost_getCore());
    GHOST_FUNC_ENTRY(GHOST_FUNCTYPE_MATH);
    ghost_error_t *ret = NULL;
    GHOST_CALL_GOTO(ghost_malloc((void **)&ret,sizeof(ghost_error_t)),err,*ret);
    *ret = GHOST_SUCCESS;

    compArgs *args = (compArgs *)vargs;
    GHOST_CALL_GOTO(args->mat->spmv(args->mat,args->res,args->invec,args->spmvOptions,args->argp),err,*ret);

    goto out;
err:
out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);
    return ret;
}
#endif

ghost_error_t ghost_spmv_taskmode(ghost_densemat_t* res, ghost_sparsemat_t* mat, ghost_densemat_t* invec, ghost_spmv_flags_t spmvOptions, va_list argp)
{
#ifndef GHOST_HAVE_MPI
    UNUSED(res);
    UNUSED(mat);
    UNUSED(invec);
    UNUSED(spmvOptions);
    UNUSED(argp);
    ERROR_LOG("Cannot execute this spMV solver without MPI");
    return GHOST_ERR_UNKNOWN;
#else
    GHOST_FUNC_ENTRY(GHOST_FUNCTYPE_MATH);
    GHOST_INSTR_START("prepare");
    ghost_error_t ret = GHOST_SUCCESS;

    ghost_spmv_flags_t localopts = spmvOptions;
    ghost_spmv_flags_t remoteopts = spmvOptions;

/*    int remoteExists;
    ghost_nrank(&remoteExists,mat->context->mpicomm);
    remoteExists -= 1;*/
    int remoteExists = mat->remotePart->nnz > 0;
    MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE,&remoteExists,1,MPI_INT,MPI_MAX,mat->context->mpicomm));
   
    if (remoteExists) {
        localopts |= GHOST_SPMV_LOCAL;
        remoteopts |= GHOST_SPMV_REMOTE;
    }

    commArgs cargs;
    compArgs cplargs;
    ghost_task_t *commTask;
    ghost_task_t *compTask;

    int taskflags = GHOST_TASK_DEFAULT;
    ghost_task_t *parent = NULL;
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
    cargs.perm = mat->permutation;
    cplargs.mat = mat->localPart;
    cplargs.invec = invec;
    cplargs.res = res;
    cplargs.spmvOptions = localopts;
    va_copy(cplargs.argp,argp);

    GHOST_INSTR_STOP("prepare");
    
    GHOST_INSTR_START("haloassembly");
    
    GHOST_CALL_GOTO(ghost_spmv_haloexchange_assemble(invec, mat->permutation),err,ret);
    
    GHOST_INSTR_STOP("haloassembly");

    GHOST_INSTR_START("both_tasks");
    if (remoteExists) {
        ghost_task_enqueue(commTask);
    }
    ghost_task_enqueue(compTask);
    ghost_task_wait(compTask);
    if ((ret = *((ghost_error_t *)(compTask->ret))) != GHOST_SUCCESS) {
        goto err;
    }
    if (remoteExists) {
        ghost_task_wait(commTask);
        if ((ret = *((ghost_error_t *)(commTask->ret))) != GHOST_SUCCESS) {
            goto err;
        }
    }
    GHOST_INSTR_STOP("both_tasks");

    GHOST_INSTR_START("remote");
    if (remoteExists) {
        mat->remotePart->spmv(mat->remotePart,res,invec,remoteopts,argp);
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
