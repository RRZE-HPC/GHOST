#include <matricks.h>
#include <mpi.h>
#include <omp.h>
#include <sys/types.h>
#include <likwid.h>
#include "kernel_helper.h"

void hybrid_kernel_III(VECTOR_TYPE* res, LCRP_TYPE* lcrp, VECTOR_TYPE* invec){

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
	static int max_dues;
	static real *work_mem, **work;
	static double hlp_sent;
	static double hlp_recv;

	int me; 
	int i, j;
	int from_PE, to_PE;
	int send_messages, recv_messages;

	static MPI_Request *send_request, *recv_request;
	static MPI_Status  *send_status,  *recv_status;


	/* Thread-ID */
	int tid;

	size_t size_request, size_status, size_work, size_mem;

	/*****************************************************************************
	 *******            ........ Executable statements ........           ********
	 ****************************************************************************/

	MPI_Comm_rank(MPI_COMM_WORLD, &me);

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


	send_messages=0;
	recv_messages = 0;
	for (i=0;i<lcrp->nodes;i++) send_request[i] = MPI_REQUEST_NULL;

	/*****************************************************************************
	 *******        Post of Irecv to ensure that we are prepared...       ********
	 ****************************************************************************/

	for (from_PE=0; from_PE<lcrp->nodes; from_PE++){
		if (lcrp->wishes[from_PE]>0){
			MPI_Irecv( &invec->val[lcrp->hput_pos[from_PE]], lcrp->wishes[from_PE], 
					MPI_MYDATATYPE, from_PE, from_PE, MPI_COMM_WORLD, 
					&recv_request[recv_messages] );
			recv_messages++;
		}
	}

	/*****************************************************************************
	 *******                          Overlap region                       *******
	 ****************************************************************************/
#ifdef OPEN_MPI
#pragma omp parallel                                                            \
	default   (none)                                                             \
	private   (i, j, to_PE, hlp1, tid, n_local)                            \
	shared    (MPI_MYDATATYPE, ompi_mpi_real, ompi_mpi_comm_world, lcrp, me, work, invec, send_request, res, n_per_thread,           \
			send_status, recv_status, recv_request, recv_messages,                 \
			SPMVM_OPTIONS)                                                  \
	reduction (+:send_messages) 
#else
#ifdef COMPLEX // MPI_MYDATATYPE is _only_ a variable in the complex case (otherwise it's a #define) 
#pragma omp parallel                                                            \
	default   (none)                                                             \
	private   (i, j, ierr, to_PE, tid)                            \
	shared    (MPI_MYDATATYPE, \
			lcrp, me, work, invec, send_request, res,           \
			send_status, recv_status, recv_request, recv_messages,                 \
			SPMVM_OPTIONS)                                                  \
	reduction (+:send_messages)
#else
#pragma omp parallel                                                            \
	default   (none)                                                             \
	private   (i, j, to_PE, tid)                            \
	shared    (\
			lcrp, me, work, invec, send_request, res,           \
			send_status, recv_status, recv_request, recv_messages,                 \
			SPMVM_OPTIONS)                                                  \
	reduction (+:send_messages)
#endif
#endif 
	{

#ifdef _OPENMP
		tid = omp_get_thread_num();
#endif

#ifdef LIKWID_MARKER_FINE
		likwid_markerStartRegion("task mode comm (= last core) + local comp");
#endif
		if (tid == lcrp->threads-1){ /* Kommunikations-thread */
			/***********************************************************************
			 *******  Local gather of data in work array & communication    ********
			 **********************************************************************/
			for (to_PE=0 ; to_PE<lcrp->nodes ; to_PE++){

				for (j=0; j<lcrp->dues[to_PE]; j++){
					work[to_PE][j] = invec->val[lcrp->duelist[to_PE][j]];
				}

				if (lcrp->dues[to_PE]>0){
					MPI_Isend( &work[to_PE][0], lcrp->dues[to_PE], MPI_MYDATATYPE,
							to_PE, me, MPI_COMM_WORLD, &send_request[to_PE] );
					send_messages++;
				}
			}

			MPI_Waitall(lcrp->nodes, send_request, send_status);
			MPI_Waitall(recv_messages, recv_request, recv_status);
		} else { /* Rechen-threads */

			/***********************************************************************
			 *******     Calculation of SpMVM for local entries of invec->val     *******
			 **********************************************************************/
#ifdef OPENCL

			if( tid == lcrp->threads-2 ) {
				spmvmKernLocalXThread( lcrp, invec, res, &me);
			}

#else
			real hlp1;
			int n_per_thread, n_local;
			n_per_thread = lcrp->lnRows[me]/(lcrp->threads-1);

			/* Alle threads gleichviel; letzter evtl. mehr */
			if (tid < lcrp->threads-2)  n_local = n_per_thread;
			else                        n_local = lcrp->lnRows[me]-(lcrp->threads-2)*n_per_thread;



			for (i=tid*n_per_thread; i<tid*n_per_thread+n_local; i++){
				hlp1 = 0.0;
				for (j=lcrp->lrow_ptr_l[i]; j<lcrp->lrow_ptr_l[i+1]; j++){
					hlp1 = hlp1 + lcrp->lval[j] * invec->val[lcrp->lcol[j]]; 
				}

				if (SPMVM_OPTIONS & SPMVM_OPTION_AXPY) 
					res->val[i] += hlp1;
				else
					res->val[i] = hlp1;
			}

#endif

		}

#ifdef LIKWID_MARKER_FINE
		likwid_markerStopRegion("task mode comm (= last core) + local comp");
#endif
	}
	/**************************************************************************
	 *******    Calculation of SpMVM for non-local entries of invec->val     *******
	 *************************************************************************/

	spmvmKernRemote( lcrp, invec, res, &me );


}
