#include <matricks.h>
#include <mpi.h>
#include <omp.h>
#include <sys/types.h>

#ifdef LIKWID
#include <likwid.h>
#endif

#include "kernel_helper.h"
#include "kernel.h"

void hybrid_kernel_III(ghost_vec_t* res, ghost_setup_t* setup, ghost_vec_t* invec, int spmvmOptions){

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
	static unsigned int nthreads;
	static unsigned int nprocs;

	static mat_idx_t max_dues;
	static mat_data_t *work_mem, **work;
	static double hlp_sent;
	static double hlp_recv;

	static int me; 
	unsigned int i, from_PE, to_PE;
	mat_idx_t j;
	int send_messages, recv_messages;

//	static MPI_Request *send_request, *recv_request;
//	static MPI_Status  *send_status,  *recv_status;
	static MPI_Request *request;
	static MPI_Status  *status;
	static CR_TYPE *localCR;


	/* Thread-ID */
	unsigned int tid;

	size_t size_request, size_status, size_work, size_mem;

	/*****************************************************************************
	 *******            ........ Executable statements ........           ********
	 ****************************************************************************/


	if (init_kernel==1){
		MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD, &me));
		nthreads = SpMVM_getNumberOfThreads();
		nprocs = SpMVM_getNumberOfProcesses();

		localCR = (CR_TYPE *)(setup->localMatrix->data);

		max_dues = 0;
		for (i=0;i<nprocs;i++)
			if (setup->communicator->dues[i]>max_dues) 
				max_dues = setup->communicator->dues[i];

		hlp_sent = 0.0;
		hlp_recv = 0.0;
		for (i=0;i<nprocs; i++){
			hlp_sent += setup->communicator->dues[i];
			hlp_recv += setup->communicator->wishes[i];
		}




		size_mem     = (size_t)( max_dues*nprocs * sizeof( mat_data_t  ) );
		size_work    = (size_t)( nprocs          * sizeof( mat_data_t* ) );
		size_request = (size_t)( 2*nprocs          * sizeof( MPI_Request ) );
		size_status  = (size_t)( 2*nprocs          * sizeof( MPI_Status ) );

		work_mem = (mat_data_t*)  allocateMemory( size_mem,  "work_mem" );
		work     = (mat_data_t**) allocateMemory( size_work, "work" );

		for (i=0; i<nprocs; i++) work[i] = &work_mem[setup->communicator->due_displ[i]];

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

	for (i=0;i<nprocs;i++) request[i] = MPI_REQUEST_NULL;

#ifdef LIKWID_MARKER
#pragma omp parallel
	likwid_markerStartRegion("Kernel 3");
#endif
#ifdef LIKWID_MARKER_FINE
#pragma omp parallel
	likwid_markerStartRegion("Kernel 3 -- communication (last thread) & local computation (others)");
#endif

	/*****************************************************************************
	 *******        Post of Irecv to ensure that we are prepared...       ********
	 ****************************************************************************/

	for (from_PE=0; from_PE<nprocs; from_PE++){
		if (setup->communicator->wishes[from_PE]>0){
			MPI_safecall(MPI_Irecv( &invec->val[setup->communicator->hput_pos[from_PE]], setup->communicator->wishes[from_PE], 
						MPI_MYDATATYPE, from_PE, from_PE, MPI_COMM_WORLD, 
						&request[recv_messages] ));
			recv_messages++;
		}
	}

	/*****************************************************************************
	 *******                          Overlap region                       *******
	 ****************************************************************************/
#ifdef OPEN_MPI
#ifdef COMPLEX // MPI_MYDATATYPE is _only_ a variable in the complex case (otherwise it's a #define) 
#pragma omp parallel                                                            \
	default   (none)                                                             \
	private   (i, j, to_PE, tid)                            \
	shared    (MPI_MYDATATYPE, ompi_mpi_double, ompi_mpi_comm_world,\
			setup, me, work, invec, res,  localCR,          \
			status, request, recv_messages,                 \
			spmvmOptions,stderr, nthreads, nprocs)                                                  \
	reduction (+:send_messages)
#else
#pragma omp parallel                                                            \
	default   (none)                                                             \
	private   (i, j, to_PE, tid)                            \
	shared    (ompi_mpi_double, ompi_mpi_comm_world,\
			setup, me, work, invec, res,  localCR,          \
			status, request, recv_messages,                 \
			spmvmOptions,stderr, nthreads, nprocs)                                                  \
	reduction (+:send_messages)
#endif
#else
#ifdef COMPLEX // MPI_MYDATATYPE is _only_ a variable in the complex case (otherwise it's a #define) 
#pragma omp parallel                                                            \
	default   (none)                                                             \
	private   (i, j, to_PE, tid)                            \
	shared    (MPI_MYDATATYPE, \
			setup, me, work, invec, res, localCR,           \
			status, request, recv_messages,                 \
			spmvmOptions,stderr, nthreads, nprocs)                                                  \
	reduction (+:send_messages)
#else
#pragma omp parallel                                                            \
	default   (none)                                                             \
	private   (i, j, to_PE, tid)                            \
	shared    (\
			setup, me, work, invec, request, res,  localCR,          \
			status, recv_messages,                 \
			spmvmOptions,stderr, nthreads, nprocs)                                                  \
	reduction (+:send_messages)
#endif
#endif 
	{

#ifdef _OPENMP
		tid = omp_get_thread_num();
#endif
		for (to_PE=0 ; to_PE<nprocs ; to_PE++){
#pragma omp for
				for (j=0; j<setup->communicator->dues[to_PE]; j++){
					work[to_PE][j] = invec->val[setup->communicator->duelist[to_PE][j]];
				}
		}

		if (tid == nthreads-1){ /* Kommunikations-thread */
			/***********************************************************************
			 *******  Local gather of data in work array & communication    ********
			 **********************************************************************/
			for (to_PE=0 ; to_PE<nprocs ; to_PE++){

				if (setup->communicator->dues[to_PE]>0){
					MPI_safecall(MPI_Isend( &work[to_PE][0], setup->communicator->dues[to_PE], MPI_MYDATATYPE,
								to_PE, me, MPI_COMM_WORLD, &request[recv_messages+send_messages] ));
					send_messages++;
				}
			}

			//MPI_safecall(MPI_Waitall(nprocs, send_request, send_status));
			//MPI_safecall(MPI_Waitall(recv_messages, recv_request, recv_status));
			MPI_safecall(MPI_Waitall(send_messages+recv_messages, request, status));
		} else { /* Rechen-threads */

			/***********************************************************************
			 *******     Calculation of SpMVM for local entries of invec->val     *******
			 **********************************************************************/
#ifdef OPENCL

			if( tid == nthreads-2 ) {
				spmvmKernLocalXThread( setup->communicator, invec, res, &me, spmvmOptions);
			}

#else
			mat_data_t hlp1;
			int n_per_thread, n_local;
			n_per_thread = setup->communicator->lnrows[me]/(nthreads-1);

			/* Alle threads gleichviel; letzter evtl. mehr */
			if (tid < nthreads-2)  n_local = n_per_thread;
			else                        n_local = setup->communicator->lnrows[me]-(nthreads-2)*n_per_thread;

			for (i=tid*n_per_thread; i<tid*n_per_thread+n_local; i++){
				hlp1 = 0.0;
				for (j=localCR->rpt[i]; j<localCR->rpt[i+1]; j++){
					hlp1 = hlp1 + localCR->val[j] * invec->val[localCR->col[j]]; 
				}

				if (spmvmOptions & GHOST_OPTION_AXPY) 
					res->val[i] += hlp1;
				else
					res->val[i] = hlp1;
			}

#endif

		}

	}
#ifdef LIKWID_MARKER_FINE
#pragma omp parallel
	{
		likwid_markerStopRegion("Kernel 3 -- communication (last thread) & local computation (others)");
		likwid_markerStartRegion("Kernel 3 -- remote computation");
	}
#endif
	/**************************************************************************
	 *******    Calculation of SpMVM for non-local entries of invec->val     *******
	 *************************************************************************/

	spmvmKernAll( setup->remoteMatrix->data, invec, res, spmvmOptions|GHOST_OPTION_AXPY );

#ifdef LIKWID_MARKER_FINE
#pragma omp parallel
	likwid_markerStopRegion("Kernel 3 -- remote computation");
#endif

#ifdef LIKWID_MARKER
#pragma omp parallel
	likwid_markerStopRegion("Kernel 3");
#endif


}
