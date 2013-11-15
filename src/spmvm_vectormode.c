#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#ifdef LIKWID_PERFMON
#include <likwid.h>
#endif

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
	int msgcount;
	ghost_vidx_t c;

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

		work = (char *)ghost_malloc(invec->traits->nvecs*max_dues*nprocs * ghost_sizeofDataType(invec->traits->datatype));

		request = (MPI_Request*) ghost_malloc(invec->traits->nvecs*2*nprocs*sizeof(MPI_Request));
		status  = (MPI_Status*)  ghost_malloc(invec->traits->nvecs*2*nprocs*sizeof(MPI_Status));

		init_kernel = 0;
	}

	if (context == NULL) {
		free(work);
		free(request);
		free(status);
		return;
	}
#ifdef __INTEL_COMPILER
	kmp_set_blocktime(1);
#endif

	msgcount=0;
	for (i=0;i<invec->traits->nvecs*2*nprocs;i++) { 
		request[i] = MPI_REQUEST_NULL;
	}

	for (from_PE=0; from_PE<nprocs; from_PE++){
		if (context->communicator->wishes[from_PE]>0){
			for (c=0; c<invec->traits->nvecs; c++) {
				MPI_safecall(MPI_Irecv(VECVAL(invec,invec->val,c,context->communicator->hput_pos[from_PE]), context->communicator->wishes[from_PE]*sizeofRHS,MPI_CHAR, from_PE, from_PE, context->mpicomm,&request[msgcount] ));
				msgcount++;
			}
		}
	}

#pragma omp parallel private(to_PE,i,c)
	for (to_PE=0 ; to_PE<nprocs ; to_PE++){
		for (c=0; c<invec->traits->nvecs; c++) {
#pragma omp for 
			for (i=0; i<context->communicator->dues[to_PE]; i++){
				memcpy(work + c*nprocs*max_dues*sizeofRHS + (to_PE*max_dues+i)*sizeofRHS,VECVAL(invec,invec->val,c,context->communicator->duelist[to_PE][i]),sizeofRHS);
			}
		}
	}

	for (to_PE=0 ; to_PE<nprocs ; to_PE++){
		if (context->communicator->dues[to_PE]>0){
			for (c=0; c<invec->traits->nvecs; c++) {
				MPI_safecall(MPI_Isend( work + c*nprocs*max_dues*sizeofRHS + to_PE*max_dues*sizeofRHS, context->communicator->dues[to_PE]*sizeofRHS, MPI_CHAR, to_PE, me, context->mpicomm, &request[msgcount] ));
				msgcount++;
			}
		}
	}

	MPI_safecall(MPI_Waitall(msgcount, request, status));

	invec->uploadHalo(invec);
	mat->kernel(mat,res,invec,spmvmOptions);	
}

