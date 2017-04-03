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
#include "ghost/omp.h"

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

//#define USE_GHOST_TASKS

#ifdef GHOST_HAVE_MPI
#ifdef USE_GHOST_TASKS
typedef struct {
    ghost_densemat *rhs;
    ghost_densemat_halo_comm *comm;
    ghost_context *ctx;
} commArgs;

static void *communicate(void *vargs)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
    DEBUG_LOG(1,"Communication task started");
    commArgs *args = (commArgs *)vargs;
    ghost_error *ret = NULL;
    ghost_malloc((void **)&ret,sizeof(ghost_error)); // don't use macro because it would read *ret
    if (!ret) {
        goto err;
    }
    *ret = GHOST_SUCCESS;
    GHOST_CALL_GOTO(ghost_densemat_halocomm_init(args->rhs,args->ctx,args->comm),err,*ret);
    GHOST_CALL_GOTO(ghost_densemat_halocomm_start(args->rhs,args->ctx,args->comm),err,*ret);
    GHOST_CALL_GOTO(ghost_densemat_halocomm_finalize(args->rhs,args->ctx,args->comm),err,*ret);

    goto out;
err:

out:
    DEBUG_LOG(1,"Communication task finished");
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_COMMUNICATION);
    return ret;
}

typedef struct {
    ghost_sparsemat *mat;
    ghost_densemat *lhs;
    ghost_densemat *rhs;
    ghost_spmv_opts spmvtraits;
} compArgs;

static void *compute(void *vargs)
{
    DEBUG_LOG(1,"Computation task started");
    ghost_error *ret = NULL;
    ghost_malloc((void **)&ret,sizeof(ghost_error)); // don't use macro because it would read *ret
    if (!ret) {
        goto err;
    }
    *ret = GHOST_SUCCESS;

    compArgs *args = (compArgs *)vargs;
    GHOST_CALL_GOTO(ghost_spmv_nocomm(args->lhs,args->mat,args->rhs,args->spmvtraits),err,*ret);

    goto out;
err:
out:
    DEBUG_LOG(1,"Computation task finished");
    return ret;
}
#endif

#endif

ghost_error ghost_spmv_pipelined(ghost_densemat* lhs, ghost_sparsemat* mat, ghost_densemat* rhs, ghost_spmv_opts traits)
{
#ifndef GHOST_HAVE_MPI
    UNUSED(lhs);
    UNUSED(mat);
    UNUSED(rhs);
    UNUSED(traits);
    ERROR_LOG("Cannot execute this spMV solver without MPI");
    return GHOST_ERR_UNKNOWN;
#else
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    WARNING_LOG("Experimental function!");
    GHOST_INSTR_START("prepare");
    ghost_error ret = GHOST_SUCCESS;

    ghost_densemat_halo_comm comm = GHOST_DENSEMAT_HALO_COMM_INITIALIZER;
#ifdef USE_GHOST_TASKS
    commArgs communiArgs;
    compArgs computeArgs;
    ghost_task *commTask;
    ghost_task *compTask;

    ghost_task_flags taskflags = GHOST_TASK_DEFAULT;
    ghost_task *parent = NULL;
    GHOST_CALL_RETURN(ghost_task_cur(&parent));
    if (parent) {
        DEBUG_LOG(1,"using the parent's cores for the task mode spmv solver");
        ghost_task_create(&compTask, parent->nThreads - 1, 0, &compute, &computeArgs, taskflags, NULL, 0);
        ghost_task_create(&commTask, 1, 0, &communicate, &communiArgs, taskflags, NULL, 0);


    } else {
        DEBUG_LOG(1,"No parent task in task mode spmv solver");

        int nIdleCores;
        ghost_pumap_nidle(&nIdleCores,GHOST_NUMANODE_ANY);
        ghost_task_create(&compTask, nIdleCores - 1, 0, &compute, &computeArgs, taskflags, NULL, 0);
        ghost_task_create(&commTask, 1, 0, &communicate, &communiArgs, taskflags, NULL, 0);
    }

    communiArgs.comm = &comm;
    communiArgs.ctx = mat->context;
    computeArgs.mat = mat;
    computeArgs.spmvtraits = traits;
#endif
    
    GHOST_INSTR_STOP("prepare");
    GHOST_INSTR_START("comm_first");

    GHOST_CALL_GOTO(ghost_densemat_halocomm_init(rhs->subdm[0],mat->context,&comm),err,ret);
    GHOST_CALL_GOTO(ghost_densemat_halocomm_start(rhs->subdm[0],mat->context,&comm),err,ret);
    GHOST_CALL_GOTO(ghost_densemat_halocomm_finalize(rhs->subdm[0],mat->context,&comm),err,ret);
    
    GHOST_INSTR_STOP("comm_first");
    
    for (ghost_lidx s=0; s<rhs->nsub-1; s++) { 
        GHOST_INSTR_START("pipelined_comm+comp");
#ifdef USE_GHOST_TASKS
        communiArgs.rhs = rhs->subdm[s+1];
        computeArgs.rhs = rhs->subdm[s];
        computeArgs.lhs = lhs->subdm[s];
        
        ghost_task_enqueue(commTask);

        ghost_task_enqueue(compTask);
        ghost_task_wait(compTask);
        if ((ret = *((ghost_error *)(compTask->ret))) != GHOST_SUCCESS) {
            goto err;
        }
        ghost_task_wait(commTask);
        if ((ret = *((ghost_error *)(commTask->ret))) != GHOST_SUCCESS) {
            goto err;
        }
#else
        GHOST_CALL_GOTO(ghost_densemat_halocomm_init(rhs->subdm[s+1],mat->context,&comm),err,ret);
        GHOST_CALL_GOTO(ghost_densemat_halocomm_start(rhs->subdm[s+1],mat->context,&comm),err,ret);
        GHOST_CALL_GOTO(ghost_spmv_nocomm(lhs->subdm[s],mat,rhs->subdm[s],traits),err,ret);
        GHOST_CALL_GOTO(ghost_densemat_halocomm_finalize(rhs->subdm[s+1],mat->context,&comm),err,ret);
#if 0
#pragma omp parallel
        {
            int core;
            ghost_cpu(&core);
        printf("thread %d @ core %d\n",ghost_omp_threadnum(),core);
        }
#endif
#endif
        GHOST_INSTR_STOP("pipelined_comm+comp");
    }
    
    GHOST_INSTR_START("comp_final");
    ghost_spmv_nocomm(lhs->subdm[lhs->nsub-1],mat,rhs->subdm[rhs->nsub-1],traits);
    GHOST_INSTR_STOP("comp_final");
       
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    goto out;
err:

out:
#ifdef USE_GHOST_TASKS
    free(compTask->ret); compTask->ret = NULL;
    free(commTask->ret); commTask->ret = NULL;
#endif

    return ret;
#endif
}
