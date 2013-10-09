#include <mpi.h>
#include <omp.h>
#include <sys/types.h>
#include <string.h>

#include "ghost_util.h"
#include "ghost_taskq.h"

#ifdef LIKWID
#include <likwid.h>
#endif


typedef struct {
	ghost_vec_t *rhs;
	ghost_context_t *context;
	int nprocs,me,send_messages,recv_messages;
	char *work;
	size_t sizeofRHS;
	MPI_Request *request;
	MPI_Status *status;
	ghost_midx_t max_dues;
} commArgs;

static void *prepare(void *vargs)
{
	int to_PE, i;
	commArgs *args = (commArgs *)vargs;
	for (to_PE=0 ; to_PE<args->nprocs ; to_PE++){
#pragma omp parallel for 
		for (i=0; i<args->context->communicator->dues[to_PE]; i++){
			memcpy(args->work+(to_PE*args->max_dues+i)*args->sizeofRHS,&((char *)(args->rhs->val))[args->context->communicator->duelist[to_PE][i]*args->sizeofRHS],args->sizeofRHS);
		}
	}
	return NULL;

}

static void *communicate(void *vargs)
{
//	ghost_setCore(23);
//#pragma omp parallel
//	{
//#pragma omp single
//	printf("    ######### communicate: numThreads: %d\n",omp_get_num_threads());
//	printf("    ######### communicate: thread %d running @ core %d\n",ghost_ompGetThreadNum(), ghost_getCore());
//	}
//	WARNING_LOG("Sleeping thread %lu",(unsigned long)pthread_self());
//	usleep(3000000);
	commArgs *args = (commArgs *)vargs;
	int to_PE,i,from_PE;

	for (from_PE=0; from_PE<args->nprocs; from_PE++){
		if (args->context->communicator->wishes[from_PE]>0){
			MPI_safecall(MPI_Irecv(&((char *)(args->rhs->val))[args->context->communicator->hput_pos[from_PE]*args->sizeofRHS], args->context->communicator->wishes[from_PE]*args->sizeofRHS,MPI_CHAR, from_PE, from_PE, args->context->mpicomm,&args->request[args->recv_messages] ));
			args->recv_messages++;
		}
	}

	for (to_PE=0 ; to_PE<args->nprocs ; to_PE++){

		if (args->context->communicator->dues[to_PE]>0){
			MPI_safecall(MPI_Isend( args->work+to_PE*args->max_dues*args->sizeofRHS, args->context->communicator->dues[to_PE]*args->sizeofRHS, MPI_CHAR, to_PE, args->me, args->context->mpicomm, &args->request[args->recv_messages+args->send_messages] ));
			args->send_messages++;
		}
	}

	MPI_safecall(MPI_Waitall(args->send_messages+args->recv_messages, args->request, args->status));

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
//	{
//#pragma omp single
//	printf("    ######### compute: numThreads: %d\n",omp_get_num_threads());
//	printf("    ######### compute: thread %d running @ core %d\n",ghost_ompGetThreadNum(), ghost_getCore());
//	}
	compArgs *args = (compArgs *)vargs;
	args->invec->uploadNonHalo(args->invec);

	args->mat->localPart->kernel(args->mat->localPart,args->res,args->invec,args->spmvmOptions);

	return NULL;
}

static void *computeRemote(void *vargs)
{
	compArgs *args = (compArgs *)vargs;
	args->mat->remotePart->kernel(args->mat->remotePart,args->res,args->invec,args->spmvmOptions|GHOST_SPMVM_AXPY);

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
	int i, from_PE, to_PE;
	int send_messages, recv_messages;

	static MPI_Request *request;
	static MPI_Status  *status;

	static size_t sizeofRHS;

	static commArgs cargs;
	static compArgs cpargs;
	static ghost_task_t *commTask;// = ghost_task_init(1, GHOST_TASK_LD_ANY, &communicate, &cargs, GHOST_TASK_DEFAULT);
	static ghost_task_t *compTask;// = ghost_task_init(ghost_thpool->nThreads-1, GHOST_TASK_LD_ANY, &computeLocal, &cpargs, GHOST_TASK_DEFAULT);
	static ghost_task_t *compRTask;// = ghost_task_init(ghost_thpool->nThreads-1, GHOST_TASK_LD_ANY, &computeLocal, &cpargs, GHOST_TASK_DEFAULT);
	static ghost_task_t *prepareTask;

	if (init_kernel==1){
		me = ghost_getRank(context->mpicomm);
		nprocs = ghost_getNumberOfRanks(context->mpicomm);
		sizeofRHS = ghost_sizeofDataType(invec->traits->datatype);

		max_dues = 0;
		for (i=0;i<nprocs;i++)
			if (context->communicator->dues[i]>max_dues) 
				max_dues = context->communicator->dues[i];

		work = (char *)ghost_malloc(max_dues*nprocs * ghost_sizeofDataType(invec->traits->datatype));
		request = (MPI_Request*) ghost_malloc( 2*nprocs*sizeof(MPI_Request));
		status  = (MPI_Status*)  ghost_malloc( 2*nprocs*sizeof(MPI_Status));

		int taskflags = GHOST_TASK_DEFAULT;
		if (pthread_getspecific(ghost_thread_key) != NULL) {
			DEBUG_LOG(1,"using the parent's cores for the task mode spmvm solver");
			taskflags |= GHOST_TASK_USE_PARENTS;
			ghost_task_t *parent = pthread_getspecific(ghost_thread_key);
			compTask = ghost_task_init(parent->nThreads-1, 0, &computeLocal, &cpargs, taskflags|GHOST_TASK_NO_HYPERTHREADS);
			compRTask = ghost_task_init(parent->nThreads, 0, &computeRemote, &cpargs, taskflags|GHOST_TASK_NO_HYPERTHREADS);
			commTask = ghost_task_init(1, ghost_thpool->nLDs-1, &communicate, &cargs, taskflags|GHOST_TASK_NO_HYPERTHREADS);
			prepareTask = ghost_task_init(parent->nThreads, 0, &prepare, &cargs, taskflags|GHOST_TASK_NO_HYPERTHREADS);
		} else {
			DEBUG_LOG(1,"No parent task in task mode spMVM solver");
			compTask = ghost_task_init(ghost_thpool->nThreads-1, 0, &computeLocal, &cpargs, taskflags);
			compRTask = ghost_task_init(ghost_thpool->nThreads, 0, &computeRemote, &cpargs, taskflags);
			commTask = ghost_task_init(1, ghost_thpool->nLDs-1, &communicate, &cargs, taskflags);
			prepareTask = ghost_task_init(ghost_thpool->nThreads, 0, &prepare, &cargs, taskflags);
		}


		
		cargs.context = context;
		cargs.nprocs = nprocs;
	   	cargs.me = me;
	   	cargs.work = work;
		cargs.rhs = invec;
	   	cargs.sizeofRHS = sizeofRHS;
	   	cargs.max_dues = max_dues;
		cpargs.mat = mat;
		cpargs.invec = invec;
	  	cpargs.res = res;
		cpargs.spmvmOptions = spmvmOptions;

		init_kernel = 0;
	}
	if (context == NULL) {
		free(work);
		free(request);
		free(status);
		return;
	}


	send_messages=0;
	recv_messages = 0;
	for (i=0;i<nprocs;i++) request[i] = MPI_REQUEST_NULL;

#ifdef LIKWID_MARKER
#pragma omp parallel
	likwid_markerStartRegion("Kernel 3");
#endif
#ifdef LIKWID_MARKER_FINE
#pragma omp parallel
	likwid_markerStartRegion("Kernel 3 -- communication");
#endif
	cargs.send_messages = send_messages;
	cargs.recv_messages = recv_messages;
	cargs.request = request;
	cargs.status = status;

//	prepare(&cargs);
//	communicate(&cargs);
//	computeLocal(&cpargs);
//	computeRemote(&cpargs);

	prepare(&cargs);
//	ghost_task_add(prepareTask);
//	ghost_task_wait(prepareTask);
//	double start = ghost_wctime();
//	communicate(&cargs);
//	WARNING_LOG("comm took %f sec",ghost_wctime()-start);
//	start = ghost_wctime();
//	computeLocal(&cpargs);
//	WARNING_LOG("comm...loc took %f sec",ghost_wctime()-start);

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

//	start = ghost_wctime();
	ghost_task_add(compTask);
//	communicate(&cargs);
	ghost_task_add(commTask);
	ghost_task_wait(commTask);
	ghost_task_wait(compTask);
//	WARNING_LOG("comm+loc took %f sec",ghost_wctime()-start);

	computeRemote(&cpargs);
//	ghost_task_add(compRTask);
//	ghost_task_wait(compRTask);
}
