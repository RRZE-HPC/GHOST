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
	ghost_context_t *context;
	int nprocs,me,send_messages,recv_messages;
	char *work;
	size_t sizeofRHS;
	MPI_Request *request;
	MPI_Status *status;
	ghost_midx_t max_dues;
} commArgs;

static void *communicate(void *vargs)
{
//#pragma omp parallel
//	{
//#pragma omp single
//	printf("    ######### communicate: numThreads: %d\n",omp_get_num_threads());
//	printf("    ######### communicate: thread %d running @ core %d\n",ghost_ompGetThreadNum(), ghost_getCore());
//	}
	commArgs *args = (commArgs *)vargs;
	int to_PE;

	for (to_PE=0 ; to_PE<args->nprocs ; to_PE++){

		if (args->context->communicator->dues[to_PE]>0){
			MPI_safecall(MPI_Isend( args->work+to_PE*args->max_dues*args->sizeofRHS, args->context->communicator->dues[to_PE]*args->sizeofRHS, MPI_CHAR, to_PE, args->me, args->context->communicator->mpicomm, &args->request[args->recv_messages+args->send_messages] ));
			args->send_messages++;
		}
	}

	MPI_safecall(MPI_Waitall(args->send_messages+args->recv_messages, args->request, args->status));

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

	args->mat->localPart->kernel(args->mat->localPart,args->res,args->invec,args->spmvmOptions);

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

	if (init_kernel==1){
		me = ghost_getRank(context->communicator->mpicomm);
		nprocs = ghost_getNumberOfRanks(context->communicator->mpicomm);
		sizeofRHS = ghost_sizeofDataType(invec->traits->datatype);

		max_dues = 0;
		for (i=0;i<nprocs;i++)
			if (context->communicator->dues[i]>max_dues) 
				max_dues = context->communicator->dues[i];

		work = (char *)ghost_malloc(max_dues*nprocs * ghost_sizeofDataType(invec->traits->datatype));
		request = (MPI_Request*) ghost_malloc( 2*nprocs*sizeof(MPI_Request));
		status  = (MPI_Status*)  ghost_malloc( 2*nprocs*sizeof(MPI_Status));

		int taskflags = GHOST_TASK_DEFAULT;
		if (pthread_getspecific(ghost_thread_key) != NULL)
			taskflags |= GHOST_TASK_USE_PARENTS;

		compTask = ghost_task_init(ghost_thpool->nThreads-1, 0, &computeLocal, &cpargs, taskflags);
		commTask = ghost_task_init(1, 1, &communicate, &cargs, taskflags);
		
		cargs.context = context;
		cargs.nprocs = nprocs;
	   	cargs.me = me;
	   	cargs.work = work;
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

	for (from_PE=0; from_PE<nprocs; from_PE++){
		if (context->communicator->wishes[from_PE]>0){
			MPI_safecall(MPI_Irecv(&((char *)(invec->val))[context->communicator->hput_pos[from_PE]*sizeofRHS], context->communicator->wishes[from_PE]*sizeofRHS,MPI_CHAR, from_PE, from_PE, context->communicator->mpicomm,&request[recv_messages] ));
			recv_messages++;
		}
	}

#pragma omp parallel private(to_PE,i) 
	{
//#pragma omp single
//		printf("assembly is done with %d threads\n",omp_get_num_threads());
	for (to_PE=0 ; to_PE<nprocs ; to_PE++){
#pragma omp for 
		for (i=0; i<context->communicator->dues[to_PE]; i++){
			memcpy(work+(to_PE*max_dues+i)*sizeofRHS,&((char *)(invec->val))[context->communicator->duelist[to_PE][i]*sizeofRHS],sizeofRHS);
		}
	}
	}

//	for (to_PE=0 ; to_PE<nprocs ; to_PE++){
//		if (context->communicator->dues[to_PE]>0){
//			MPI_safecall(MPI_Isend( work+to_PE*max_dues*sizeofRHS, context->communicator->dues[to_PE]*sizeofRHS, MPI_CHAR, to_PE, me, context->communicator->mpicomm, &request[recv_messages+send_messages] ));
//			send_messages++;
//		}
//	}

//	double start = ghost_wctime();
	cargs.send_messages = send_messages;
	cargs.recv_messages = recv_messages;
	cargs.request = request;
	cargs.status = status;
	ghost_task_add(commTask);
//	double time = ghost_wctime()-start;
	//ghost_task_destroy(&commTask);
//	printf("spawning async comm task took %f seconds\n",time);
//	communicate(&cargs);
//	commArgs *args = &cargs;

	/****************************************************************************
	 *******       Calculation of SpMVM for local entries of invec->val        *******
	 ***************************************************************************/

#ifdef LIKWID_MARKER_FINE
#pragma omp parallel
	likwid_markerStartRegion("Kernel 3 -- local computation");
#endif

#ifdef OPENCL
	CL_copyHostToDevice(invec->CL_val_gpu, invec->val, mat->nrows(mat)*sizeofRHS);
#endif
#ifdef CUDA
	CU_copyHostToDevice(invec->CU_val, invec->val, mat->nrows(mat)*sizeofRHS);
#endif


//	start = ghost_wctime();
	ghost_task_add(compTask);
	ghost_task_wait(compTask);
//	ghost_task_destroy(compTask);
//	time = ghost_wctime()-start;
	//printf("local computation took %f seconds\n",time);
//	ghost_spawnTask(&computeLocal, &cpargs, 12, compThreads, "local compute", GHOST_TASK_SYNC/*|GHOST_TASK_EXCLUSIVE*/);

//	mat->localPart->kernel(mat->localPart,res,invec,spmvmOptions);

#ifdef LIKWID_MARKER_FINE
#pragma omp parallel
	likwid_markerStopRegion("Kernel 3 -- local computation");
#endif

	/****************************************************************************
	 *******       Finishing communication: MPI_Waitall                   *******
	 ***************************************************************************/

	ghost_task_wait(commTask);
//	ghost_task_destroy(commTask);
//	start = ghost_wctime();
//	time = ghost_wctime()-start;
//	printf("had to wait for comm to finish for %f seconds\n",time);
//	MPI_safecall(MPI_Waitall(send_messages+recv_messages, request, status));

#ifdef LIKWID_MARKER_FINE
#pragma omp parallel
	{
		likwid_markerStopRegion("Kernel 3 -- communication");
		likwid_markerStartRegion("Kernel 3 -- remote computation");
	}
#endif

	/****************************************************************************
	 *******     Calculation of SpMVM for non-local entries of invec->val      *******
	 ***************************************************************************/
#ifdef OPENCL
	CL_copyHostToDeviceOffset(invec->CL_val_gpu, 
			&((char *)(invec->val))[mat->nrows(mat)*sizeofRHS], context->communicator->halo_elements*sizeofRHS,
			mat->nrows(mat)*sizeofRHS);
#endif
#ifdef CUDA
	CU_copyHostToDevice(&((char *)(invec->CU_val))[mat->nrows(mat)*sizeofRHS], 
			&((char *)(invec->val))[mat->nrows(mat)*sizeofRHS], context->communicator->halo_elements*sizeofRHS);
#endif

	mat->remotePart->kernel(mat->remotePart,res,invec,spmvmOptions|GHOST_SPMVM_AXPY);

#ifdef LIKWID_MARKER_FINE
#pragma omp parallel
	likwid_markerStopRegion("Kernel 3 -- remote computation");
#endif

#ifdef LIKWID_MARKER
#pragma omp parallel
	likwid_markerStopRegion("Kernel 3");
#endif
	
	/*static int init_kernel=1;
	static int nprocs;

	static ghost_midx_t max_dues;
	static char *work;
	static double hlp_sent;
	static double hlp_recv;

	static int me; 
	int i, from_PE, to_PE;
	int send_messages, recv_messages;

	static MPI_Request *request;
	static MPI_Status  *status;
	//static CR_TYPE *localCR;


	size_t sizeofRHS;



	if (init_kernel==1){
		MPI_safecall(MPI_Comm_rank(context->communicator->mpicomm, &me));
		nprocs = ghost_getNumberOfProcesses();
		sizeofRHS = ghost_sizeofDataType(invec->traits->datatype);

		//	localCR = (CR_TYPE *)(context->localMatrix->data);

		max_dues = 0;
		for (i=0;i<nprocs;i++)
			if (context->communicator->dues[i]>max_dues) 
				max_dues = context->communicator->dues[i];

		hlp_sent = 0.0;
		hlp_recv = 0.0;
		for (i=0;i<nprocs; i++){
			hlp_sent += context->communicator->dues[i];
			hlp_recv += context->communicator->wishes[i];
		}

		work = (char *)allocateMemory(max_dues*nprocs * ghost_sizeofDataType(invec->traits->datatype), "work");
		request = (MPI_Request*) allocateMemory( 2*nprocs*sizeof(MPI_Request), "request" );
		status  = (MPI_Status*)  allocateMemory( 2*nprocs*sizeof(MPI_Status),  "status" );

		init_kernel = 0;

#pragma omp parallel 
		{
			if (omp_get_num_threads() < 2) {
#pragma omp single
				ABORT("Cannot execute task mode kernel with less than two OpenMP threads (%d)",omp_get_num_threads());
			}
		}
	}


	send_messages=0;
	recv_messages = 0;

	for (i=0;i<nprocs;i++) request[i] = MPI_REQUEST_NULL; // TODO 2*?

#ifdef LIKWID_MARKER
#pragma omp parallel
	likwid_markerStartRegion("Kernel 3");
#endif
#ifdef LIKWID_MARKER_FINE
#pragma omp parallel
	likwid_markerStartRegion("Kernel 3 -- communication (last thread) & local computation (others)");
#endif

	for (from_PE=0; from_PE<nprocs; from_PE++){
		if (context->communicator->wishes[from_PE]>0){
			MPI_safecall(MPI_Irecv(&((char *)(invec->val))[context->communicator->hput_pos[from_PE]*sizeofRHS], context->communicator->wishes[from_PE]*sizeofRHS,MPI_CHAR, from_PE, from_PE, context->communicator->mpicomm,&request[recv_messages] ));
			recv_messages++;
		}
	}

#pragma omp parallel private(to_PE,i)
	for (to_PE=0 ; to_PE<nprocs ; to_PE++){
#pragma omp for 
		for (i=0; i<context->communicator->dues[to_PE]; i++){
			memcpy(work+(to_PE*max_dues+i)*sizeofRHS,&((char *)(invec->val))[context->communicator->duelist[to_PE][i]*sizeofRHS],sizeofRHS);
		}
	}

//	commArgs cargs = {.context = context, .nprocs = nprocs, .me = me, .send_messages = send_messages, .recv_messages = recv_messages, .work = work, .sizeofRHS = sizeofRHS, .request = request, .status = status, .max_dues = max_dues};

//	ghost_task_t commTask = ghost_spawnTask(&communicate, &cargs, 1, NULL, "communicate", GHOST_TASK_ASYNC);
//	communicate(&cargs);
//	commArgs *args = &cargs;

	for (to_PE=0 ; to_PE<nprocs ; to_PE++){
		if (context->communicator->dues[to_PE]>0){
			MPI_safecall(MPI_Isend( work+to_PE*max_dues*sizeofRHS, context->communicator->dues[to_PE]*sizeofRHS, MPI_CHAR, to_PE, me, context->communicator->mpicomm, &request[recv_messages+send_messages] ));
			send_messages++;
		}
	}

	MPI_safecall(MPI_Waitall(send_messages+recv_messages, request, status));

	mat->localPart->kernel(mat->localPart,res,invec,spmvmOptions);

//	ghost_waitTask(&commTask);

	#ifdef OPENCL
	  UNUSED(localCR);
	  if( tid == nthreads-2 ) {
	  CL_copyHostToDevice(invec->CL_val_gpu, invec->val, context->lnrows(context)*sizeof(ghost_vdat_t));
	  context->localMatrix->kernel(context->localMatrix,res,invec,spmvmOptions);
	  }
#elif defined(CUDA)
UNUSED(localCR);
if( tid == nthreads-2 ) {
CU_copyHostToDevice(invec->CU_val, invec->val, context->lnrows(context)*sizeof(ghost_vdat_t));
context->localMatrix->kernel(context->localMatrix,res,invec,spmvmOptions);
}
#else
ghost_vdat_t hlp1;
int n_per_thread, n_local;
n_per_thread = context->communicator->lnrows[me]/(nthreads-1);

if (tid < nthreads-2)  
n_local = n_per_thread;
else
n_local = context->communicator->lnrows[me]-(nthreads-2)*n_per_thread;

for (i=tid*n_per_thread; i<tid*n_per_thread+n_local; i++){
hlp1 = 0.0;
for (j=localCR->rpt[i]; j<localCR->rpt[i+1]; j++){
hlp1 = hlp1 + (ghost_vdat_t)localCR->val[j] * invec->val[localCR->col[j]]; 
}

if (spmvmOptions & GHOST_SPMVM_AXPY) 
res->val[i] += hlp1;
else
res->val[i] = hlp1;
}

#endif


#ifdef LIKWID_MARKER_FINE
#pragma omp parallel
{
	likwid_markerStopRegion("Kernel 3 -- communication (last thread) & local computation (others)");
	likwid_markerStartRegion("Kernel 3 -- remote computation");
}
#endif

#ifdef OPENCL
CL_copyHostToDeviceOffset(invec->CL_val_gpu, 
		invec->val+context->lnrows(context), context->communicator->halo_elements*sizeof(ghost_vdat_t),
		context->lnrows(context)*sizeof(ghost_vdat_t));
#endif
#ifdef CUDA
CU_copyHostToDevice(&invec->CU_val[context->lnrows(context)], 
		&invec->val[context->lnrows(context)], context->communicator->halo_elements*sizeof(ghost_vdat_t));
#endif
mat->remotePart->kernel(mat->remotePart,res,invec,spmvmOptions|GHOST_SPMVM_AXPY);

#ifdef LIKWID_MARKER_FINE
#pragma omp parallel
likwid_markerStopRegion("Kernel 3 -- remote computation");
#endif

#ifdef LIKWID_MARKER
#pragma omp parallel
likwid_markerStopRegion("Kernel 3");
#endif

*/
}
