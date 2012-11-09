#include <matricks.h>
#include <mpi.h>
#include <omp.h>
#include <sys/types.h>
#include "kernel_helper.h"
#include "kernel.h"

void hybrid_kernel_II(VECTOR_TYPE* res, SETUP_TYPE* setup, VECTOR_TYPE* invec, int spmvmOptions){

	/*****************************************************************************
	 ********              Kernel ir -- cs -- lc -- wa -- nl              ********
	 ********   'Good faith'- Ueberlapp von Rechnung und Kommunikation    ********
	 ********     - das was andere als 'hybrid' bezeichnen                ********
	 ********     - ob es klappt oder nicht haengt vom MPI ab...          ********
	 ****************************************************************************/

	static int init_kernel=1; 
	static int max_dues;
	static mat_data_t *work_mem, **work;
	static double hlp_sent;
	static double hlp_recv;

	static int me; 
	int i, j;
	int from_PE, to_PE;
	int send_messages, recv_messages;

	//static MPI_Request *send_request, *recv_request;
	//static MPI_Status  *send_status,  *recv_status;
	static MPI_Request *request;
	static MPI_Status  *status;

	size_t size_request, size_status, size_work, size_mem;


	if (init_kernel==1){
		MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD, &me));

		max_dues = 0;
		for (i=0;i<setup->communicator->nodes;i++)
			if (setup->communicator->dues[i]>max_dues) 
				max_dues = setup->communicator->dues[i];

		hlp_sent = 0.0;
		hlp_recv = 0.0;
		for (i=0;i<setup->communicator->nodes; i++){
			hlp_sent += setup->communicator->dues[i];
			hlp_recv += setup->communicator->wishes[i];
		}


		size_mem     = (size_t)( max_dues*setup->communicator->nodes * sizeof( mat_data_t  ) );
		size_work    = (size_t)( setup->communicator->nodes          * sizeof( mat_data_t* ) );
		size_request = (size_t)( 2*setup->communicator->nodes          * sizeof( MPI_Request ) );
		size_status  = (size_t)( 2*setup->communicator->nodes          * sizeof( MPI_Status ) );

		work_mem = (mat_data_t*)  allocateMemory( size_mem,  "work_mem" );
		work     = (mat_data_t**) allocateMemory( size_work, "work" );

		for (i=0; i<setup->communicator->nodes; i++) work[i] = &work_mem[setup->communicator->due_displ[i]];

		/*send_request = (MPI_Request*) allocateMemory( size_request, "send_request" );
		recv_request = (MPI_Request*) allocateMemory( size_request, "recv_request" );
		send_status  = (MPI_Status*)  allocateMemory( size_status,  "send_status" );
		recv_status  = (MPI_Status*)  allocateMemory( size_status,  "recv_status" );*/
		request = (MPI_Request*) allocateMemory( size_request, "request" );
		status  = (MPI_Status*)  allocateMemory( size_status,  "status" );

		init_kernel = 0;
	}


	send_messages=0;
	recv_messages = 0;
	for (i=0;i<setup->communicator->nodes;i++) request[i] = MPI_REQUEST_NULL;

#ifdef LIKWID_MARKER
#pragma omp parallel
	likwid_markerStartRegion("Kernel 2");
#endif
#ifdef LIKWID_MARKER_FINE
#pragma omp parallel
	likwid_markerStartRegion("Kernel 2 -- communication");
#endif

	for (from_PE=0; from_PE<setup->communicator->nodes; from_PE++){
		if (setup->communicator->wishes[from_PE]>0){
			MPI_safecall(MPI_Irecv( &invec->val[setup->communicator->hput_pos[from_PE]], setup->communicator->wishes[from_PE], 
					MPI_MYDATATYPE, from_PE, from_PE, MPI_COMM_WORLD, 
					&request[recv_messages] ));
			recv_messages++;
		}
	}

	/*****************************************************************************
	 *******       Local assembly of halo-elements  & Communication       ********
	 ****************************************************************************/

	for (to_PE=0 ; to_PE<setup->communicator->nodes ; to_PE++){
#pragma omp parallel for private(j) 
		for (j=0; j<setup->communicator->dues[to_PE]; j++){
			work[to_PE][j] = invec->val[setup->communicator->duelist[to_PE][j]];
		}
		if (setup->communicator->dues[to_PE]>0){
			MPI_safecall(MPI_Isend( &work[to_PE][0], setup->communicator->dues[to_PE], 
					MPI_MYDATATYPE, to_PE, me, MPI_COMM_WORLD, 
					&request[recv_messages+send_messages] ));
			send_messages++;
		}
	}

	/****************************************************************************
	 *******       Calculation of SpMVM for local entries of invec->val        *******
	 ***************************************************************************/

#ifdef LIKWID_MARKER_FINE
#pragma omp parallel
	likwid_markerStartRegion("Kernel 2 -- local computation");
#endif

	spmvmKernAll( setup->localMatrix->data, invec, res, spmvmOptions);

#ifdef LIKWID_MARKER_FINE
#pragma omp parallel
	likwid_markerStopRegion("Kernel 2 -- local computation");
#endif

	/****************************************************************************
	 *******       Finishing communication: MPI_Waitall                   *******
	 ***************************************************************************/

	//MPI_safecall(MPI_Waitall(send_messages, send_request, send_status));
	//MPI_safecall(MPI_Waitall(recv_messages, recv_request, recv_status));
	MPI_safecall(MPI_Waitall(send_messages+recv_messages, request, status));

#ifdef LIKWID_MARKER_FINE
#pragma omp parallel
	{
	likwid_markerStopRegion("Kernel 2 -- communication");
	likwid_markerStartRegion("Kernel 2 -- remote computation");
	}
#endif

	/****************************************************************************
	 *******     Calculation of SpMVM for non-local entries of invec->val      *******
	 ***************************************************************************/

	spmvmKernAll( setup->remoteMatrix->data, invec, res, spmvmOptions|SPMVM_OPTION_AXPY );

#ifdef LIKWID_MARKER_FINE
#pragma omp parallel
	likwid_markerStopRegion("Kernel 2 -- remote computation");
#endif

#ifdef LIKWID_MARKER
#pragma omp parallel
	likwid_markerStopRegion("Kernel 2");
#endif


}
