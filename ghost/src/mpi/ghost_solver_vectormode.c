#include <mpi.h>
#include <stdio.h>
#include <string.h>


#include "ghost_util.h"

void hybrid_kernel_I(ghost_context_t *context, ghost_vec_t* res, ghost_mat_t* mat, ghost_vec_t* invec, int spmvmOptions)
{

	/*****************************************************************************
	 ********                  Kernel ir -- cs -- wa -- ca                ********   
	 ********          Kommunikation mittels MPI_ISend, MPI_IRecv         ********
	 ********                serielles Umkopieren und Senden              ********
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

	size_t sizeofRHS;

	if (init_kernel==1){
		me = ghost_getRank();
		nprocs = ghost_getNumberOfProcesses();
		sizeofRHS = ghost_sizeofDataType(invec->traits->datatype);

		max_dues = 0;
		for (i=0;i<nprocs;i++)
			if (context->communicator->dues[i]>max_dues) 
				max_dues = context->communicator->dues[i];

		work = (char *)allocateMemory(max_dues*nprocs * ghost_sizeofDataType(invec->traits->datatype), "work");
		request = (MPI_Request*) allocateMemory( 2*nprocs*sizeof(MPI_Request), "request" );
		status  = (MPI_Status*)  allocateMemory( 2*nprocs*sizeof(MPI_Status),  "status" );

		init_kernel = 0;
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
			MPI_safecall(MPI_Irecv(&((char *)(invec->val))[context->communicator->hput_pos[from_PE]*sizeofRHS], context->communicator->wishes[from_PE]*sizeofRHS,MPI_CHAR, from_PE, from_PE, MPI_COMM_WORLD,&request[recv_messages] ));
			recv_messages++;
		}
	}

//#pragma omp parallel private(to_PE,i)
	for (to_PE=0 ; to_PE<nprocs ; to_PE++){
//#pragma omp for 
		for (i=0; i<context->communicator->dues[to_PE]; i++){
			memcpy(work+to_PE*max_dues*sizeofRHS+i*sizeofRHS,&((char *)(invec->val))[context->communicator->duelist[to_PE][i]*sizeofRHS],sizeofRHS);
		}
	}

	for (to_PE=0 ; to_PE<nprocs ; to_PE++){
		if (context->communicator->dues[to_PE]>0){
			MPI_safecall(MPI_Isend( work+to_PE*max_dues*sizeofRHS, context->communicator->dues[to_PE]*sizeofRHS, MPI_CHAR, to_PE, me, MPI_COMM_WORLD, &request[recv_messages+send_messages] ));
			send_messages++;
		}
	}


	MPI_safecall(MPI_Waitall(send_messages+recv_messages, request, status));

#ifdef LIKWID_MARKER_FINE
#pragma omp parallel
	{
	likwid_markerStopRegion("Kernel 1 -- communication");
	likwid_markerStartRegion("Kernel 1 -- computation");
	}
#endif

#ifdef OPENCL
	DEBUG_LOG(1,"Vector mode kernel: Upload RHS to OpenCL device");
	CL_uploadVector(invec);
#endif
#ifdef CUDA
	DEBUG_LOG(1,"Vector mode kernel: Upload RHS to CUDA device");
	CU_uploadVector(invec);
#endif


	mat->kernel(mat,res,invec,spmvmOptions);	

#ifdef LIKWID_MARKER_FINE
#pragma omp parallel
	likwid_markerStartRegion("Kernel 1 -- computation");
#endif
#ifdef LIKWID_MARKER
#pragma omp parallel
	likwid_markerStartRegion("Kernel 1");
#endif

}

