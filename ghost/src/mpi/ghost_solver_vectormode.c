#include <mpi.h>
#include <stdio.h>

#include "ghost_util.h"

void hybrid_kernel_I(ghost_vec_t* res, ghost_context_t* context, ghost_vec_t* invec, int spmvmOptions)
{

	/*****************************************************************************
	 ********                  Kernel ir -- cs -- wa -- ca                ********   
	 ********          Kommunikation mittels MPI_ISend, MPI_IRecv         ********
	 ********                serielles Umkopieren und Senden              ********
	 ****************************************************************************/

	static int init_kernel=1; 
	static ghost_mnnz_t max_dues;
	//static ghost_vdat_t *work_mem, **work;
	static char *work_mem, **work;
	static double hlp_sent;
	static double hlp_recv;
	static unsigned int nprocs;

	static int me; 
	ghost_mnnz_t j;
	unsigned int i, from_PE, to_PE;
	int send_messages, recv_messages;

	static MPI_Request *request;
	static MPI_Status  *status;
	//static MPI_Request *send_request, *recv_request;
	//static MPI_Status  *send_status,  *recv_status;

	size_t sizeofRHS = ghost_sizeofDataType(invec->traits->datatype);

	size_t size_request, size_status, size_work, size_mem;

	if (init_kernel==1){
		
		MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD, &me));
		nprocs = ghost_getNumberOfProcesses();

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



		//size_mem     = (size_t)( max_dues*nprocs * sizeof( ghost_vdat_t  ) );
		//size_work    = (size_t)( nprocs          * sizeof( ghost_vdat_t* ) );
		size_mem     = (size_t)( max_dues*nprocs * ghost_sizeofDataType(invec->traits->datatype) );
		size_work    = (size_t)( nprocs          * sizeof( void * ) );
		size_request = (size_t)( 2*nprocs          * sizeof( MPI_Request ) );
		size_status  = (size_t)( 2*nprocs          * sizeof( MPI_Status ) );

		work_mem = (char*)  allocateMemory( size_mem,  "work_mem" );
		work     = (char**) allocateMemory( size_work, "work" );
	//	work_mem = (ghost_vdat_t*)  allocateMemory( size_mem,  "work_mem" );
	//	work     = (ghost_vdat_t**) allocateMemory( size_work, "work" );

		for (i=0; i<nprocs; i++) work[i] = &work_mem[context->communicator->due_displ[i]];

		/*send_request = (MPI_Request*) allocateMemory( size_request, "send_request" );
		recv_request = (MPI_Request*) allocateMemory( size_request, "recv_request" );
		send_status  = (MPI_Status*)  allocateMemory( size_status,  "send_status" );
		recv_status  = (MPI_Status*)  allocateMemory( size_status,  "recv_status" );*/
		request = (MPI_Request*) allocateMemory( size_request, "request" );
		status  = (MPI_Status*)  allocateMemory( size_status,  "status" );

		init_kernel = 0;
	}

	send_messages = 0;
	recv_messages = 0;

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
			//MPI_safecall(MPI_Irecv(&invec->val[context->communicator->hput_pos[from_PE]], context->communicator->wishes[from_PE], 
			//		ghost_mpi_dt_vdat, from_PE, from_PE, MPI_COMM_WORLD, 
			//		&request[recv_messages] ));
			recv_messages++;
		}
	}

	for (to_PE=0 ; to_PE<nprocs ; to_PE++){
#pragma omp parallel for private(j)
		for (j=0; j<context->communicator->dues[to_PE]*sizeofRHS; j++){
			work[to_PE][j] = ((char *)(invec->val))[context->communicator->duelist[to_PE][j/sizeofRHS]];
			//work[to_PE][j] = invec->val[context->communicator->duelist[to_PE][j]];
		}
		if (context->communicator->dues[to_PE]>0){
			MPI_safecall(MPI_Isend( &work[to_PE][0], context->communicator->dues[to_PE]*sizeofRHS, 
					MPI_CHAR, to_PE, me, MPI_COMM_WORLD, 
					&request[recv_messages+send_messages] ));
			//MPI_safecall(MPI_Isend( &work[to_PE][0], context->communicator->dues[to_PE], 
			//		ghost_mpi_dt_vdat, to_PE, me, MPI_COMM_WORLD, 
			//		&request[recv_messages+send_messages] ));
			send_messages++;
		}
	}

	MPI_safecall(MPI_Waitall(send_messages+recv_messages, request, status));
	//MPI_safecall(MPI_Waitall(send_messages, send_request, send_status));
	//MPI_safecall(MPI_Waitall(recv_messages, recv_request, recv_status));

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


	context->fullMatrix->kernel(context->fullMatrix,res,invec,spmvmOptions);	

#ifdef LIKWID_MARKER_FINE
#pragma omp parallel
	likwid_markerStartRegion("Kernel 1 -- computation");
#endif
#ifdef LIKWID_MARKER
#pragma omp parallel
	likwid_markerStartRegion("Kernel 1");
#endif

}

