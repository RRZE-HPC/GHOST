#include <mpi.h>
#include <stdio.h>
#include <string.h>


#include "ghost_util.h"

// if called with context==NULL: clean up variables
void hybrid_kernel_I(ghost_context_t *context, ghost_vec_t* res, ghost_mat_t* mat, ghost_vec_t* invec, int spmvmOptions)
{

	/*****************************************************************************
	 ********                  Kernel ir -- cs -- wa -- ca                ********   
	 ********          Kommunikation mittels MPI_ISend, MPI_IRecv         ********
	 ********                serielles Umkopieren und Senden              ********
	 ****************************************************************************/

	static int init_kernel=1; 
	static ghost_mnnz_t max_dues;
	static int nprocs;

	static int me; 
	int i, from_PE, to_PE;
	int send_messages, recv_messages;

	static char *work;
	static MPI_Request *request;
	static MPI_Status  *status;

	static size_t sizeofRHS;

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
	likwid_markerStartRegion("Kernel 1");
#endif
#ifdef LIKWID_MARKER_FINE
#pragma omp parallel
	likwid_markerStartRegion("Kernel 1 -- communication");
#endif

	for (from_PE=0; from_PE<nprocs; from_PE++){
		if (context->communicator->wishes[from_PE]>0){
			MPI_safecall(MPI_Irecv(&((char *)(invec->val))[context->communicator->hput_pos[from_PE]*sizeofRHS], context->communicator->wishes[from_PE]*sizeofRHS,MPI_CHAR, from_PE, from_PE, context->mpicomm,&request[recv_messages] ));
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

	for (to_PE=0 ; to_PE<nprocs ; to_PE++){
		if (context->communicator->dues[to_PE]>0){
			MPI_safecall(MPI_Isend( work+to_PE*max_dues*sizeofRHS, context->communicator->dues[to_PE]*sizeofRHS, MPI_CHAR, to_PE, me, context->mpicomm, &request[recv_messages+send_messages] ));
			send_messages++;
		}
	}

	MPI_safecall(MPI_Waitall(send_messages+recv_messages, request, status));

	double start;
#ifdef LIKWID_MARKER_FINE
#pragma omp parallel
	{
	likwid_markerStopRegion("Kernel 1 -- communication");
	likwid_markerStartRegion("Kernel 1 -- computation");
	}
#endif

	DEBUG_LOG(1,"Vector mode kernel: Upload RHS to device");
//	start = ghost_wctime();
	invec->upload(invec);
//	WARNING_LOG("comm: %f",ghost_wctime()-start);


//	start = ghost_wctime();
	mat->kernel(mat,res,invec,spmvmOptions);	
//	WARNING_LOG("comp: %f",ghost_wctime()-start);

#ifdef LIKWID_MARKER_FINE
#pragma omp parallel
	likwid_markerStartRegion("Kernel 1 -- computation");
#endif
#ifdef LIKWID_MARKER
#pragma omp parallel
	likwid_markerStartRegion("Kernel 1");
#endif

}

