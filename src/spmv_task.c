#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/locality.h"
#include "ghost/densemat.h"
#include "ghost/task.h"
#include "ghost/taskq.h"
#include "ghost/pumap.h"
#include "ghost/machine.h"
#include "ghost/constants.h"
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
    ghost_context_t *context;
    int nprocs,me,msgcount;
    char *work;
    MPI_Request *request;
    MPI_Status *status;
    ghost_idx_t max_dues;
} commArgs;

static void *communicate(void *vargs)
{
//#pragma omp parallel
//    INFO_LOG("comm t %d running @ core %d",ghost_ompGetThreadNum(),ghost_getCore());

    GHOST_INSTR_START(spmv_task_communicate);
    ghost_error_t *ret = NULL;
    GHOST_CALL_GOTO(ghost_malloc((void **)&ret,sizeof(ghost_error_t)),err,*ret);
    *ret = GHOST_SUCCESS;

    int to_PE, from_PE;
    ghost_idx_t c;
    commArgs *args = (commArgs *)vargs;
#ifdef GHOST_HAVE_INSTR_TIMING
    size_t recvBytes = 0, sendBytes = 0;
    size_t recvMsgs = 0, sendMsgs = 0;
#endif

    for (from_PE=0; from_PE<args->nprocs; from_PE++){
#ifdef GHOST_HAVE_INSTR_TIMING
            INFO_LOG("from %d: %zu bytes",from_PE,args->mat->context->wishes[from_PE]*args->rhs->traits->elSize);
#endif
        if (args->context->wishes[from_PE]>0){
            for (c=0; c<args->rhs->traits->ncols; c++) {
                MPI_CALL_GOTO(MPI_Irecv(VECVAL(args->rhs,args->rhs->val,c,args->context->hput_pos[from_PE]), args->context->wishes[from_PE]*args->rhs->traits->elSize,MPI_CHAR, from_PE, from_PE, args->context->mpicomm,&args->request[args->msgcount]),err,*ret);
                args->msgcount++;
#ifdef GHOST_HAVE_INSTR_TIMING
                recvBytes += args->context->wishes[from_PE]*args->rhs->traits->elSize;
                recvMsgs++;
#endif
            }
        }
    }
    
    for (to_PE=0 ; to_PE<args->nprocs ; to_PE++){
        if (args->context->dues[to_PE]>0){
            for (c=0; c<args->rhs->traits->ncols; c++) {
                MPI_CALL_GOTO(MPI_Isend( args->work + c*args->nprocs*args->max_dues*args->rhs->traits->elSize + to_PE*args->max_dues*args->rhs->traits->elSize, args->context->dues[to_PE]*args->rhs->traits->elSize, MPI_CHAR, to_PE, args->me, args->context->mpicomm, &args->request[args->msgcount]),err,*ret);
                args->msgcount++;
#ifdef GHOST_HAVE_INSTR_TIMING
                sendBytes += args->context->dues[to_PE]*args->rhs->traits->elSize;
                sendMsgs++;
#endif
            }
        }
    }
    


    MPI_CALL_GOTO(MPI_Waitall(args->msgcount, args->request, args->status),err,*ret);
    GHOST_CALL_GOTO(args->rhs->uploadHalo(args->rhs),err,*ret);
    
    GHOST_INSTR_STOP(spmv_task_communicate);

#ifdef GHOST_HAVE_INSTR_TIMING
    INFO_LOG("sendbytes: %zu",sendBytes);
    INFO_LOG("recvbytes: %zu",recvBytes);
    INFO_LOG("sendmsgs : %zu",sendMsgs);
    INFO_LOG("recvmsgs : %zu",recvMsgs);
#endif

    goto out;
err:

out:
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
    ghost_error_t *ret = NULL;
    GHOST_CALL_GOTO(ghost_malloc((void **)&ret,sizeof(ghost_error_t)),err,*ret);
    *ret = GHOST_SUCCESS;

    GHOST_INSTR_START(spmv_task_computeLocal);
    compArgs *args = (compArgs *)vargs;
    GHOST_CALL_GOTO(args->mat->spmv(args->mat,args->res,args->invec,args->spmvOptions,args->argp),err,*ret);
    GHOST_INSTR_STOP(spmv_task_computeLocal);

    goto out;
err:
out:
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
    ERROR_LOG("Cannot execute this spMV solver without MPI");
    return GHOST_ERR_UNKNOWN;
#else
    GHOST_INSTR_START(spmv_task_entiresolver)
    ghost_nnz_t max_dues;
    char *work = NULL;
    ghost_error_t ret = GHOST_SUCCESS;
    int nprocs;

    int me; 
    int i;


    ghost_spmv_flags_t localopts = spmvOptions;
    ghost_spmv_flags_t remoteopts = spmvOptions;

    int remoteExists = mat->remotePart->nnz > 0;
   
    if (remoteExists) {
        localopts &= ~GHOST_SPMV_COMPUTE_LOCAL_DOTPRODUCT;

        remoteopts &= ~GHOST_SPMV_AXPBY;
        remoteopts &= ~GHOST_SPMV_APPLY_SHIFT;
        remoteopts |= GHOST_SPMV_AXPY;
    }


    commArgs cargs;
    compArgs cplargs;
    ghost_task_t *commTask;
    ghost_task_t *compTask;
    GHOST_INSTR_START(spmv_task_prepare);

    DEBUG_LOG(1,"In task mode spmv solver");
    GHOST_CALL_RETURN(ghost_getRank(mat->context->mpicomm,&me));
    GHOST_CALL_RETURN(ghost_getNumberOfRanks(mat->context->mpicomm,&nprocs));
    MPI_Request request[invec->traits->ncols*2*nprocs];
    MPI_Status  status[invec->traits->ncols*2*nprocs];

    max_dues = 0;
    for (i=0;i<nprocs;i++)
        if (mat->context->dues[i]>max_dues) 
            max_dues = mat->context->dues[i];

    GHOST_CALL_RETURN(ghost_malloc((void **)&work,invec->traits->ncols*max_dues*nprocs * invec->traits->elSize));

    int taskflags = GHOST_TASK_DEFAULT;
    if (pthread_getspecific(ghost_thread_key) != NULL) {
        DEBUG_LOG(1,"using the parent's cores for the task mode spmv solver");
        taskflags |= GHOST_TASK_USE_PARENTS;
        ghost_task_t *parent = pthread_getspecific(ghost_thread_key);
        ghost_task_create(&compTask, parent->nThreads - remoteExists, 0, &computeLocal, &cplargs, taskflags);
        ghost_task_create(&commTask, remoteExists, 0, &communicate, &cargs, taskflags);


    } else {
        DEBUG_LOG(1,"No parent task in task mode spmv solver");

        int nIdleCores;
        ghost_pumap_getNumberOfIdlePUs(&nIdleCores,GHOST_NUMANODE_ANY);
        ghost_task_create(&compTask, nIdleCores-remoteExists, 0, &computeLocal, &cplargs, taskflags);
        ghost_task_create(&commTask, remoteExists, 0, &communicate, &cargs, taskflags);
    }



    cargs.context = mat->context;
    cargs.nprocs = nprocs;
    cargs.me = me;
    cargs.work = work;
    cargs.rhs = invec;
    cargs.max_dues = max_dues;
    cargs.msgcount = 0;
    cargs.request = request;
    cargs.status = status;
    cplargs.mat = mat->localPart;
    cplargs.invec = invec;
    cplargs.res = res;
    cplargs.spmvOptions = localopts;
    va_copy(cplargs.argp,argp);

    for (i=0;i<invec->traits->ncols*2*nprocs;i++) {
        request[i] = MPI_REQUEST_NULL;
    }

#ifdef __INTEL_COMPILER
  //  kmp_set_blocktime(1);
#endif
    
    int to_PE;
    ghost_idx_t c;
    
    GHOST_INSTR_STOP(spmv_task_prepare);
    GHOST_INSTR_START(spmv_task_assemblebuffer);
    invec->downloadNonHalo(invec);

    if ((mat->traits->flags & GHOST_SPARSEMAT_PERMUTE) && 
            (mat->permutation->scope == GHOST_PERMUTATION_LOCAL)) {
#pragma omp parallel private(to_PE,i,c)
        for (to_PE=0 ; to_PE<nprocs ; to_PE++){
            for (c=0; c<invec->traits->ncols; c++) {
#pragma omp for 
                for (i=0; i<mat->context->dues[to_PE]; i++){
                    memcpy(work + c*nprocs*max_dues*invec->traits->elSize + (to_PE*max_dues+i)*invec->traits->elSize,VECVAL(invec,invec->val,c,mat->permutation->perm[mat->context->duelist[to_PE][i]]),invec->traits->elSize);
                }
            }
        }
    } else {
#pragma omp parallel private(to_PE,i,c)
        for (to_PE=0 ; to_PE<nprocs ; to_PE++){
            for (c=0; c<invec->traits->ncols; c++) {
#pragma omp for 
                for (i=0; i<mat->context->dues[to_PE]; i++){
                    memcpy(work + c*nprocs*max_dues*invec->traits->elSize + (to_PE*max_dues+i)*invec->traits->elSize,VECVAL(invec,invec->val,c,mat->context->duelist[to_PE][i]),invec->traits->elSize);
                }
            }
        }
    }
    GHOST_INSTR_STOP(spmv_task_assemblebuffer);

    GHOST_INSTR_START(spmv_task_both_tasks);
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
    GHOST_INSTR_STOP(spmv_task_both_tasks);

    GHOST_INSTR_START(spmv_task_computeRemote);
    if (remoteExists) {
        mat->remotePart->spmv(mat->remotePart,res,invec,remoteopts,argp);
    }
    GHOST_INSTR_STOP(spmv_task_computeRemote);
       
    GHOST_INSTR_STOP(spmv_task_entiresolver)

    goto out;
err:

out:
    free(work); work = NULL;
    free(compTask->ret); compTask->ret = NULL;
    free(commTask->ret); commTask->ret = NULL;

    return ret;
#endif
}
