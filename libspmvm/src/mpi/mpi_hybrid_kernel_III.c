#include <matricks.h>
#include <mpi.h>
#include <omp.h>
#include <sys/types.h>

#ifdef LIKWID
#include <likwid.h>
#endif

#include "kernel_helper.h"
#include "kernel.h"

void hybrid_kernel_III(ghost_vec_t* res, ghost_context_t* context, ghost_vec_t* invec, int spmvmOptions){

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

	static ghost_midx_t max_dues;
	static ghost_vdat_t *work_mem, **work;
	static double hlp_sent;
	static double hlp_recv;

	static int me; 
	unsigned int i, from_PE, to_PE;
	ghost_midx_t j;
	int send_messages, recv_messages;

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
		nthreads = ghost_getNumberOfThreads();
		nprocs = ghost_getNumberOfProcesses();

		localCR = (CR_TYPE *)(context->localMatrix->data);

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




		size_mem     = (size_t)( max_dues*nprocs * sizeof( ghost_vdat_t  ) );
		size_work    = (size_t)( nprocs          * sizeof( ghost_vdat_t* ) );
		size_request = (size_t)( 2*nprocs          * sizeof( MPI_Request ) );
		size_status  = (size_t)( 2*nprocs          * sizeof( MPI_Status ) );

		work_mem = (ghost_vdat_t*)  allocateMemory( size_mem,  "work_mem" );
		work     = (ghost_vdat_t**) allocateMemory( size_work, "work" );

		for (i=0; i<nprocs; i++) work[i] = &work_mem[context->communicator->due_displ[i]];

		request = (MPI_Request*) allocateMemory( size_request, "request" );
		status  = (MPI_Status*)  allocateMemory( size_status,  "status" );

		init_kernel = 0;
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

	/*****************************************************************************
	 *******        Post of Irecv to ensure that we are prepared...       ********
	 ****************************************************************************/

	for (from_PE=0; from_PE<nprocs; from_PE++){
		if (context->communicator->wishes[from_PE]>0){
			MPI_safecall(MPI_Irecv( &invec->val[context->communicator->hput_pos[from_PE]], context->communicator->wishes[from_PE], 
						ghost_mpi_dt_vdat, from_PE, from_PE, MPI_COMM_WORLD, 
						&request[recv_messages] ));
			recv_messages++;
		}
	}

	/*****************************************************************************
	 *******                          Overlap region                       *******
	 ****************************************************************************/
#pragma omp parallel                                                            \
	default   (shared)                                                             \
	private   (i, j, to_PE, tid)                            \
	reduction (+:send_messages)
	{

#ifdef _OPENMP
		tid = omp_get_thread_num();
#endif
		for (to_PE=0 ; to_PE<nprocs ; to_PE++){
#pragma omp for
				for (j=0; j<context->communicator->dues[to_PE]; j++){
					work[to_PE][j] = invec->val[context->communicator->duelist[to_PE][j]];
				}
		}

		if (tid == nthreads-1){ /* Kommunikations-thread */
			/***********************************************************************
			 *******  Local gather of data in work array & communication    ********
			 **********************************************************************/
			for (to_PE=0 ; to_PE<nprocs ; to_PE++){

				if (context->communicator->dues[to_PE]>0){
					MPI_safecall(MPI_Isend( &work[to_PE][0], context->communicator->dues[to_PE], ghost_mpi_dt_vdat,
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
			UNUSED(localCR);
			if( tid == nthreads-2 ) {
				CL_copyHostToDevice(invec->CL_val_gpu, invec->val, context->lnrows(context)*sizeof(ghost_vdat_t));
				context->localMatrix->kernel(context->localMatrix,res,invec,spmvmOptions);
			}

#else
			ghost_vdat_t hlp1;
			int n_per_thread, n_local;
			n_per_thread = context->communicator->lnrows[me]/(nthreads-1);

			/* Alle threads gleichviel; letzter evtl. mehr */
			if (tid < nthreads-2)  
				n_local = n_per_thread;
			else
				n_local = context->communicator->lnrows[me]-(nthreads-2)*n_per_thread;

			for (i=tid*n_per_thread; i<tid*n_per_thread+n_local; i++){
				hlp1 = 0.0;
				for (j=localCR->rpt[i]; j<localCR->rpt[i+1]; j++){
					hlp1 = hlp1 + (ghost_vdat_t)localCR->val[j] * invec->val[localCR->col[j]]; 
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

#ifdef OPENCL
	CL_copyHostToDeviceOffset(invec->CL_val_gpu, 
			invec->val+context->lnrows(context), context->communicator->halo_elements*sizeof(ghost_vdat_t),
			context->lnrows(context)*sizeof(ghost_vdat_t));
#endif
	context->remoteMatrix->kernel(context->remoteMatrix,res,invec,spmvmOptions|GHOST_OPTION_AXPY);

#ifdef LIKWID_MARKER_FINE
#pragma omp parallel
	likwid_markerStopRegion("Kernel 3 -- remote computation");
#endif

#ifdef LIKWID_MARKER
#pragma omp parallel
	likwid_markerStopRegion("Kernel 3");
#endif


}
