#include <mpi.h>
#include <omp.h>
#include <sys/types.h>
#include <string.h>

#include "ghost_util.h"

// if called with context==NULL: clean up variables
void hybrid_kernel_II(ghost_context_t *context, ghost_vec_t* res, ghost_mat_t* mat, ghost_vec_t* invec, int spmvmOptions)
{

	/*****************************************************************************
	 ********              Kernel ir -- cs -- lc -- wa -- nl              ********
	 ********   'Good faith'- Ueberlapp von Rechnung und Kommunikation    ********
	 ********     - das was andere als 'hybrid' bezeichnen                ********
	 ********     - ob es klappt oder nicht haengt vom MPI ab...          ********
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

	if (init_kernel==1){
		me = ghost_getRank();
		nprocs = ghost_getNumberOfProcesses();
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
	likwid_markerStartRegion("Kernel 2");
#endif
#ifdef LIKWID_MARKER_FINE
#pragma omp parallel
	likwid_markerStartRegion("Kernel 2 -- communication");
#endif

	for (from_PE=0; from_PE<nprocs; from_PE++){
		if (context->communicator->wishes[from_PE]>0){
			MPI_safecall(MPI_Irecv(&((char *)(invec->val))[context->communicator->hput_pos[from_PE]*sizeofRHS], context->communicator->wishes[from_PE]*sizeofRHS,MPI_CHAR, from_PE, from_PE, MPI_COMM_WORLD,&request[recv_messages] ));
			recv_messages++;
		}
	}

	/*****************************************************************************
	 *******       Local assembly of halo-elements  & Communication       ********
	 ****************************************************************************/

#pragma omp parallel private(to_PE,i)
	for (to_PE=0 ; to_PE<nprocs ; to_PE++){
#pragma omp for 
		for (i=0; i<context->communicator->dues[to_PE]; i++){
			memcpy(work+(to_PE*max_dues+i)*sizeofRHS,&((char *)(invec->val))[context->communicator->duelist[to_PE][i]*sizeofRHS],sizeofRHS);
		}
	}

	for (to_PE=0 ; to_PE<nprocs ; to_PE++){
		if (context->communicator->dues[to_PE]>0){
			MPI_safecall(MPI_Isend( work+to_PE*max_dues*sizeofRHS, context->communicator->dues[to_PE]*sizeofRHS, MPI_CHAR, to_PE, me, MPI_COMM_WORLD, &request[recv_messages+send_messages] ));
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

#ifdef OPENCL
	CL_copyHostToDevice(invec->CL_val_gpu, invec->val, mat->nrows(mat)*sizeofRHS);
#endif
#ifdef CUDA
	CU_copyHostToDevice(invec->CU_val, invec->val, mat->nrows(mat)*sizeofRHS);
#endif

	mat->localPart->kernel(mat->localPart,res,invec,spmvmOptions);

#ifdef LIKWID_MARKER_FINE
#pragma omp parallel
	likwid_markerStopRegion("Kernel 2 -- local computation");
#endif

	/****************************************************************************
	 *******       Finishing communication: MPI_Waitall                   *******
	 ***************************************************************************/

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
	likwid_markerStopRegion("Kernel 2 -- remote computation");
#endif

#ifdef LIKWID_MARKER
#pragma omp parallel
	likwid_markerStopRegion("Kernel 2");
#endif


}
