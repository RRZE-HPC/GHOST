#include <ghost_types.h>
#include <ghost_affinity.h>
#include <ghost_vec.h>
#include <ghost_taskq.h>
#include <ghost_constants.h>
#include <ghost_util.h>

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

static void *prepare(void *vargs)
{
    int to_PE, i;
    ghost_vidx_t c;
    commArgs *args = (commArgs *)vargs;
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

static void *communicate(void *vargs)
{
//    ghost_setCore(23);
//#pragma omp parallel
//    {
//#pragma omp single
//    printf("    ######### communicate: numThreads: %d\n",omp_get_num_threads());
//    printf("    ######### communicate: thread %d (%llu) running @ core %d\n",ghost_ompGetThreadNum(), (unsigned long)pthread_self(), ghost_getCore());
//    }
//    WARNING_LOG("Sleeping thread %lu",(unsigned long)pthread_self());
//    usleep(3000000);
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

static void *computeLocal(void *vargs)
{
//#pragma omp parallel
//    {
//#pragma omp single
//    printf("    ######### compute: numThreads: %d\n",omp_get_num_threads());
//    printf("    ######### compute: thread %d (%llu) running @ core %d\n",ghost_ompGetThreadNum(), (unsigned long)pthread_self(), ghost_getCore());
//    }
    compArgs *args = (compArgs *)vargs;
//    args->invec->uploadNonHalo(args->invec);

    args->mat->localPart->spmv(args->mat->localPart,args->res,args->invec,args->spmvmOptions);

    return NULL;
}

static void *computeRemote(void *vargs)
{
    compArgs *args = (compArgs *)vargs;
    args->mat->remotePart->spmv(args->mat->remotePart,args->res,args->invec,args->spmvmOptions|GHOST_SPMVM_AXPY);

    return NULL;
}

// if called with context==NULL: clean up variables
void hybrid_kernel_III(ghost_context_t *context, ghost_vec_t* res, ghost_mat_t* mat, ghost_vec_t* invec, int spmvmOptions)
{

    /*****************************************************************************
     ********               Kernel ir -- lc|csw  -- nl                    ********
     ********     Expliziter Ueberlapp von Rechnung und Kommunikation     ********
     ********        durch Abspalten eines Kommunikationsthreads          ********
     ********     - Umkopieren durch Kommunikationsthread simultan mit    ********
     ********       lokaler Rechnung im overlap-Block                     ********
     ********     - nichtlokalen Rechnung mit allen threads               ********
     ********     - naive Reihenfolge der MPI_ISend                       ********
     ****************************************************************************/

    static int init_kernel=1; 
    static ghost_mnnz_t max_dues;
    static char *work;
    static int nprocs;

    static int me; 
    int i;

    static MPI_Request *request;
    static MPI_Status  *status;
    
    int localopts = spmvmOptions;
    localopts &= ~GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT;
    
    int remoteopts = spmvmOptions;
    remoteopts &= ~GHOST_SPMVM_AXPBY;
    remoteopts &= ~GHOST_SPMVM_APPLY_SHIFT;
    remoteopts |= GHOST_SPMVM_AXPY;

    static size_t sizeofRHS;

    static commArgs cargs;
    static compArgs cplargs;
    static compArgs cprargs;
    static ghost_task_t *commTask;// = ghost_task_init(1, GHOST_TASK_LD_ANY, &communicate, &cargs, GHOST_TASK_DEFAULT);
    static ghost_task_t *compTask;// = ghost_task_init(ghost_thpool->nThreads-1, GHOST_TASK_LD_ANY, &computeLocal, &cpargs, GHOST_TASK_DEFAULT);
    static ghost_task_t *compRTask;// = ghost_task_init(ghost_thpool->nThreads-1, GHOST_TASK_LD_ANY, &computeLocal, &cpargs, GHOST_TASK_DEFAULT);
    static ghost_task_t *prepareTask;

    if (init_kernel==1){
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
            ghost_task_init(&compTask, parent->nThreads, 0, &computeLocal, &cplargs, taskflags|GHOST_TASK_NO_HYPERTHREADS);
            ghost_task_init(&compRTask, parent->nThreads, 0, &computeRemote, &cprargs, taskflags|GHOST_TASK_NO_HYPERTHREADS);
            ghost_task_init(&commTask, 1, 0, &communicate, &cargs, taskflags|GHOST_TASK_ONLY_HYPERTHREADS);
            ghost_task_init(&prepareTask, parent->nThreads, 0, &prepare, &cargs, taskflags|GHOST_TASK_NO_HYPERTHREADS);
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
        cplargs.mat = mat;
        cplargs.invec = invec;
          cplargs.res = res;
        cplargs.spmvmOptions = localopts;
        cprargs.mat = mat;
        cprargs.invec = invec;
          cprargs.res = res;
        cprargs.spmvmOptions = remoteopts;

        init_kernel = 0;
    }
    if (context == NULL) {
        free(work);
        free(request);
        free(status);
    //    kmp_set_blocktime(200);
        return;
    }


    for (i=0;i<invec->traits->nvecs*2*nprocs;i++) {
        request[i] = MPI_REQUEST_NULL;
    }

#ifdef LIKWID_MARKER
#pragma omp parallel
    likwid_markerStartRegion("Kernel 3");
#endif
#ifdef LIKWID_MARKER_FINE
#pragma omp parallel
    likwid_markerStartRegion("Kernel 3 -- communication");
#endif
    cargs.msgcount = 0;
    cargs.request = request;
    cargs.status = status;

//    prepare(&cargs);
//    communicate(&cargs);
//    computeLocal(&cpargs);
//    computeRemote(&cpargs);
//#pragma omp parallel
#ifdef __INTEL_COMPILER
    kmp_set_blocktime(1);
#endif

    prepare(&cargs);


//    ghost_task_add(prepareTask);
//    ghost_task_wait(prepareTask);
//    double start = ghost_wctime();
//    communicate(&cargs);
//    WARNING_LOG("comm took %f sec",ghost_wctime()-start);
//    start = ghost_wctime();
//    computeLocal(&cpargs);
//    WARNING_LOG("comm...loc took %f sec",ghost_wctime()-start);

/*
#pragma omp parallel num_threads(2)
    {
        if (omp_get_thread_num() == 0) {
            ghost_setCore(11);
            communicate(&cargs);
        } else {
            omp_set_num_threads(11);
            ghost_pinThreads(GHOST_PIN_MANUAL,"0,1,2,3,4,5,6,7,8,9,10");
            computeLocal(&cpargs);
        }
    }

*/

//    start = ghost_wctime();
    ghost_task_add(commTask);
    ghost_task_add(compTask);
//    communicate(&cargs);
    ghost_task_wait(commTask);
    ghost_task_wait(compTask);
//    double start = ghost_wctime();
//    computeLocal(&cpargs);
//    WARNING_LOG("comploc took %f sec",ghost_wctime()-start);
//    WARNING_LOG("comm+loc took %f sec",ghost_wctime()-start);

    computeRemote(&cprargs);
//    ghost_task_add(compRTask);
//    ghost_task_wait(compRTask);
}
