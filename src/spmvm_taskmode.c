#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/affinity.h"
#include "ghost/vec.h"
#include "ghost/taskq.h"
#include "ghost/constants.h"
#include "ghost/util.h"

#include <mpi.h>
#include <sys/types.h>
#include <string.h>

#ifdef LIKWID
#include <likwid.h>
#endif

#if GHOST_HAVE_OPENMP
#include <omp.h>
#endif

typedef struct {
    ghost_vec_t *rhs;
    ghost_context_t *context;
    int nprocs,me,msgcount;
    char *work;
    size_t sizeofRHS;
    MPI_Request *request;
    MPI_Status *status;
    ghost_midx_t max_dues;
} commArgs;

void *prepare(void *vargs)
{
    int to_PE, i;
    ghost_vidx_t c;
    commArgs *args = (commArgs *)vargs;
    args->rhs->downloadNonHalo(args->rhs);
#pragma omp parallel private(to_PE,i,c)
    for (to_PE=0 ; to_PE<args->nprocs ; to_PE++){
        for (c=0; c<args->rhs->traits->nvecs; c++) {
#pragma omp for 
            for (i=0; i<args->context->dues[to_PE]; i++){
                memcpy(args->work + c*args->nprocs*args->max_dues*args->sizeofRHS + (to_PE*args->max_dues+i)*args->sizeofRHS,VECVAL(args->rhs,args->rhs->val,c,args->context->duelist[to_PE][i]),args->sizeofRHS);
            }
        }
    }
    return NULL;

}

void *communicate(void *vargs)
{
    commArgs *args = (commArgs *)vargs;
    int to_PE,from_PE;
    ghost_vidx_t c;

    for (from_PE=0; from_PE<args->nprocs; from_PE++){
        if (args->context->wishes[from_PE]>0){
            for (c=0; c<args->rhs->traits->nvecs; c++) {
                MPI_safecall(MPI_Irecv(VECVAL(args->rhs,args->rhs->val,c,args->context->hput_pos[from_PE]), args->context->wishes[from_PE]*args->sizeofRHS,MPI_CHAR, from_PE, from_PE, args->context->mpicomm,&args->request[args->msgcount] ));
                args->msgcount++;
            }
        }
    }

    for (to_PE=0 ; to_PE<args->nprocs ; to_PE++){
        if (args->context->dues[to_PE]>0){
            for (c=0; c<args->rhs->traits->nvecs; c++) {
                MPI_safecall(MPI_Isend( args->work + c*args->nprocs*args->max_dues*args->sizeofRHS + to_PE*args->max_dues*args->sizeofRHS, args->context->dues[to_PE]*args->sizeofRHS, MPI_CHAR, to_PE, args->me, args->context->mpicomm, &args->request[args->msgcount] ));
                args->msgcount++;
            }
        }
    }

    MPI_safecall(MPI_Waitall(args->msgcount, args->request, args->status));

    args->rhs->uploadHalo(args->rhs);
    return NULL;
}

typedef struct {
    ghost_mat_t *mat;
    ghost_vec_t *res;
    ghost_vec_t *invec;
    int spmvmOptions;
} compArgs;

void *computeLocal(void *vargs)
{
    compArgs *args = (compArgs *)vargs;

    args->mat->localPart->spmv(args->mat->localPart,args->res,args->invec,args->spmvmOptions);

    return NULL;
}

void *computeRemote(void *vargs)
{
    compArgs *args = (compArgs *)vargs;
    args->mat->remotePart->spmv(args->mat->remotePart,args->res,args->invec,args->spmvmOptions);

    return NULL;
}

void hybrid_kernel_III(ghost_context_t *context, ghost_vec_t* res, ghost_mat_t* mat, ghost_vec_t* invec, int spmvmOptions)
{
    ghost_mnnz_t max_dues;
    char *work;
    int nprocs;

    int me; 
    int i;

    MPI_Request *request;
    MPI_Status  *status;

    int localopts = spmvmOptions;
    localopts &= ~GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT;

    int remoteopts = spmvmOptions;
    remoteopts &= ~GHOST_SPMVM_AXPBY;
    remoteopts &= ~GHOST_SPMVM_APPLY_SHIFT;
    remoteopts |= GHOST_SPMVM_AXPY;

    size_t sizeofRHS;

    commArgs cargs;
    compArgs cplargs;
    compArgs cprargs;
    ghost_task_t *commTask;// = ghost_task_init(1, GHOST_TASK_LD_ANY, &communicate, &cargs, GHOST_TASK_DEFAULT);
    ghost_task_t *compTask;// = ghost_task_init(ghost_thpool->nThreads-1, GHOST_TASK_LD_ANY, &computeLocal, &cpargs, GHOST_TASK_DEFAULT);
    ghost_task_t *compRTask;// = ghost_task_init(ghost_thpool->nThreads-1, GHOST_TASK_LD_ANY, &computeLocal, &cpargs, GHOST_TASK_DEFAULT);
    ghost_task_t *prepareTask;

    DEBUG_LOG(1,"In task mode spMVM solver");
    me = ghost_getRank(context->mpicomm);
    nprocs = ghost_getNumberOfRanks(context->mpicomm);
    sizeofRHS = ghost_sizeofDataType(invec->traits->datatype);

    max_dues = 0;
    for (i=0;i<nprocs;i++)
        if (context->dues[i]>max_dues) 
            max_dues = context->dues[i];

    work = (char *)ghost_malloc(invec->traits->nvecs*max_dues*nprocs * ghost_sizeofDataType(invec->traits->datatype));
    request = (MPI_Request*) ghost_malloc(invec->traits->nvecs*2*nprocs*sizeof(MPI_Request));
    status  = (MPI_Status*)  ghost_malloc(invec->traits->nvecs*2*nprocs*sizeof(MPI_Status));

    int taskflags = GHOST_TASK_DEFAULT;
    if (pthread_getspecific(ghost_thread_key) != NULL) {
        DEBUG_LOG(1,"using the parent's cores for the task mode spmvm solver");
        taskflags |= GHOST_TASK_USE_PARENTS;
        ghost_task_t *parent = pthread_getspecific(ghost_thread_key);
        ghost_task_init(&compTask, parent->nThreads/ghost_getSMTlevel(), 0, &computeLocal, &cplargs, taskflags|GHOST_TASK_NO_HYPERTHREADS);
        ghost_task_init(&compRTask, parent->nThreads/ghost_getSMTlevel(), 0, &computeRemote, &cprargs, taskflags|GHOST_TASK_NO_HYPERTHREADS);
        ghost_task_init(&commTask, 1, 0, &communicate, &cargs, taskflags|GHOST_TASK_ONLY_HYPERTHREADS);
        ghost_task_init(&prepareTask, parent->nThreads/ghost_getSMTlevel(), 0, &prepare, &cargs, taskflags|GHOST_TASK_NO_HYPERTHREADS);
    } else {
        DEBUG_LOG(1,"No parent task in task mode spMVM solver");
        ghost_task_init(&compTask, ghost_thpool->nThreads-1, 0, &computeLocal, &cplargs, taskflags);
        ghost_task_init(&compRTask, ghost_thpool->nThreads, 0, &computeRemote, &cprargs, taskflags);
        ghost_task_init(&commTask, 1, ghost_thpool->nLDs-1, &communicate, &cargs, taskflags);
        ghost_task_init(&prepareTask, ghost_thpool->nThreads, 0, &prepare, &cargs, taskflags);
    }



    cargs.context = context;
    cargs.nprocs = nprocs;
    cargs.me = me;
    cargs.work = work;
    cargs.rhs = invec;
    cargs.sizeofRHS = sizeofRHS;
    cargs.max_dues = max_dues;
    cargs.msgcount = 0;
    cargs.request = request;
    cargs.status = status;
    cplargs.mat = mat;
    cplargs.invec = invec;
    cplargs.res = res;
    cplargs.spmvmOptions = localopts;
    cprargs.mat = mat;
    cprargs.invec = invec;
    cprargs.res = res;
    cprargs.spmvmOptions = remoteopts;


    for (i=0;i<invec->traits->nvecs*2*nprocs;i++) {
        request[i] = MPI_REQUEST_NULL;
    }

#ifdef __INTEL_COMPILER
    kmp_set_blocktime(1);
#endif

    prepare(&cargs);
    ghost_task_add(commTask);
    ghost_task_add(compTask);
    ghost_task_wait(commTask);
    ghost_task_wait(compTask);
    ghost_task_add(compRTask);
    ghost_task_wait(compRTask);

    free(work);
    free(request);
    free(status);
}
