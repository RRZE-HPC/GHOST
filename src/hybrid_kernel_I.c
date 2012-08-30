#include <matricks.h>
#include <mpi.h>
#include "kernel_helper.h"
#include "kernel.h"

void hybrid_kernel_I(VECTOR_TYPE* res, LCRP_TYPE* lcrp, VECTOR_TYPE* invec)
{

	/*****************************************************************************
	 ********                  Kernel ir -- cs -- wa -- ca                ********   
	 ********          Kommunikation mittels MPI_ISend, MPI_IRecv         ********
	 ********                serielles Umkopieren und Senden              ********
	 ****************************************************************************/

	static int init_kernel=1; 
	static int max_dues;
	static real *work_mem, **work;
	static double hlp_sent;
	static double hlp_recv;

	int me; 
	int i, j, from_PE, to_PE;
	int send_messages, recv_messages;

	static MPI_Request *send_request, *recv_request;
	static MPI_Status  *send_status,  *recv_status;


	size_t size_request, size_status, size_work, size_mem;

	MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD, &me));

	if (init_kernel==1){

		max_dues = 0;
		for (i=0;i<lcrp->nodes;i++)
			if (lcrp->dues[i]>max_dues) 
				max_dues = lcrp->dues[i];

		hlp_sent = 0.0;
		hlp_recv = 0.0;
		for (i=0;i<lcrp->nodes; i++){
			hlp_sent += lcrp->dues[i];
			hlp_recv += lcrp->wishes[i];
		}



		size_mem     = (size_t)( max_dues*lcrp->nodes * sizeof( real  ) );
		size_work    = (size_t)( lcrp->nodes          * sizeof( real* ) );
		size_request = (size_t)( lcrp->nodes          * sizeof( MPI_Request ) );
		size_status  = (size_t)( lcrp->nodes          * sizeof( MPI_Status ) );

		work_mem = (real*)  allocateMemory( size_mem,  "work_mem" );
		work     = (real**) allocateMemory( size_work, "work" );

		for (i=0; i<lcrp->nodes; i++) work[i] = &work_mem[lcrp->due_displ[i]];

		send_request = (MPI_Request*) allocateMemory( size_request, "send_request" );
		recv_request = (MPI_Request*) allocateMemory( size_request, "recv_request" );
		send_status  = (MPI_Status*)  allocateMemory( size_status,  "send_status" );
		recv_status  = (MPI_Status*)  allocateMemory( size_status,  "recv_status" );

		init_kernel = 0;
	}

	send_messages = 0;
	recv_messages = 0;


	for (from_PE=0; from_PE<lcrp->nodes; from_PE++){
		if (lcrp->wishes[from_PE]>0){
			MPI_safecall(MPI_Irecv(&invec->val[lcrp->hput_pos[from_PE]], lcrp->wishes[from_PE], 
					MPI_MYDATATYPE, from_PE, from_PE, MPI_COMM_WORLD, 
					&recv_request[recv_messages] ));
			recv_messages++;
		}
	}

	for (to_PE=0 ; to_PE<lcrp->nodes ; to_PE++){
		for (j=0; j<lcrp->dues[to_PE]; j++){
			work[to_PE][j] = invec->val[lcrp->duelist[to_PE][j]];
		}
		if (lcrp->dues[to_PE]>0){
			MPI_safecall(MPI_Isend( &work[to_PE][0], lcrp->dues[to_PE], 
					MPI_MYDATATYPE, to_PE, me, MPI_COMM_WORLD, 
					&send_request[send_messages] ));
			send_messages++;
		}
	}

	MPI_safecall(MPI_Waitall(send_messages, send_request, send_status));
	MPI_safecall(MPI_Waitall(recv_messages, recv_request, recv_status));

	spmvmKernAll(lcrp, invec, res, &me);

}

