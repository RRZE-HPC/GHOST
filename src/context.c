#include <ghost_config.h>
#include <ghost_types.h>
#include <ghost_util.h>
#include <ghost_context.h>
#include <ghost_constants.h>
#include <ghost_affinity.h>
#include <ghost_crs.h>


ghost_context_t *ghost_createContext(int64_t gnrows, int64_t gncols, int context_flags, char *matrixPath, MPI_Comm comm, double weight) 
{
	ghost_context_t *context;
	int i;

	context = (ghost_context_t *)ghost_malloc(sizeof(ghost_context_t));
	context->flags = context_flags;
	context->rowPerm = NULL;
	context->invRowPerm = NULL;
	context->mpicomm = comm;

	if ((gnrows == GHOST_GET_DIM_FROM_MATRIX) || (gncols == GHOST_GET_DIM_FROM_MATRIX)) {
		ghost_matfile_header_t fileheader;
		ghost_readMatFileHeader(matrixPath,&fileheader);
#ifndef LONGIDX
		if ((fileheader.nrows >= (int64_t)INT_MAX) || (fileheader.ncols >= (int64_t)INT_MAX)) {
			ABORT("The matrix is too big for 32-bit indices. Recompile with LONGIDX!");
		}
#endif
		if (gnrows == GHOST_GET_DIM_FROM_MATRIX)
			context->gnrows = (ghost_midx_t)fileheader.nrows;
		if (gncols == GHOST_GET_DIM_FROM_MATRIX)
			context->gncols = (ghost_midx_t)fileheader.ncols;

	} else {
#ifndef LONGIDX
		if ((gnrows >= (int64_t)INT_MAX) || (gncols >= (int64_t)INT_MAX)) {
			ABORT("The matrix is too big for 32-bit indices. Recompile with LONGIDX!");
		}
#endif
		context->gnrows = (ghost_midx_t)gnrows;
		context->gncols = (ghost_midx_t)gncols;
	}
	DEBUG_LOG(1,"Creating context with dimension %"PRmatIDX"x%"PRmatIDX,context->gnrows,context->gncols);

#ifdef GHOST_HAVE_MPI
	if (!(context->flags & GHOST_CONTEXT_DISTRIBUTED) && !(context->flags & GHOST_CONTEXT_GLOBAL)) {
		DEBUG_LOG(1,"Context is set to be distributed");
		context->flags |= GHOST_CONTEXT_DISTRIBUTED;
	}
#else
	if (context->flags & GHOST_CONTEXT_DISTRIBUTED) {
		ABORT("Creating a distributed matrix without MPI is not possible");
	} else if (!(context->flags & GHOST_CONTEXT_GLOBAL)) {
		DEBUG_LOG(1,"Context is set to be global");
		context->flags |= GHOST_CONTEXT_GLOBAL;
	}
#endif

	context->solvers = (ghost_solver_t *)ghost_malloc(sizeof(ghost_solver_t)*GHOST_NUM_MODES);
	for (i=0; i<GHOST_NUM_MODES; i++) context->solvers[i] = NULL;
#ifdef GHOST_HAVE_MPI
	context->solvers[GHOST_SPMVM_MODE_VECTORMODE_IDX] = &hybrid_kernel_I;
	context->solvers[GHOST_SPMVM_MODE_GOODFAITH_IDX] = &hybrid_kernel_II;
	context->solvers[GHOST_SPMVM_MODE_TASKMODE_IDX] = &hybrid_kernel_III;
#else
	context->solvers[GHOST_SPMVM_MODE_NOMPI_IDX] = &ghost_solver_nompi;
#endif

#ifdef GHOST_HAVE_MPI
	if (context->flags & GHOST_CONTEXT_DISTRIBUTED) {
		context->communicator = (ghost_comm_t*) ghost_malloc( sizeof(ghost_comm_t));
		context->communicator->halo_elements = -1;

		int nprocs = ghost_getNumberOfRanks(context->mpicomm);

		context->communicator->lnEnts   = (ghost_mnnz_t*)       ghost_malloc( nprocs*sizeof(ghost_mnnz_t)); 
		context->communicator->lfEnt    = (ghost_mnnz_t*)       ghost_malloc( nprocs*sizeof(ghost_mnnz_t)); 
		context->communicator->lnrows   = (ghost_midx_t*)       ghost_malloc( nprocs*sizeof(ghost_midx_t)); 
		context->communicator->lfRow    = (ghost_midx_t*)       ghost_malloc( nprocs*sizeof(ghost_midx_t)); 

		if (context->flags & GHOST_CONTEXT_WORKDIST_NZE)
		{ // read rpt and fill lfrow, lnrows, lfent, lnents
			ghost_midx_t *rpt = NULL;
			ghost_mnnz_t gnnz;

			if (ghost_getRank(context->mpicomm) == 0) {
				rpt = CRS_readRpt(context->gnrows+1,matrixPath);
				context->rpt = rpt;

				gnnz = rpt[context->gnrows];
				ghost_mnnz_t target_nnz;
				target_nnz = (gnnz/nprocs)+1; /* sonst bleiben welche uebrig! */

				context->communicator->lfRow[0]  = 0;
				context->communicator->lfEnt[0] = 0;
				int j = 1;

				for (i=0;i<context->gnrows;i++){
					if (rpt[i] >= j*target_nnz){
						context->communicator->lfRow[j] = i;
						context->communicator->lfEnt[j] = rpt[i];
						j = j+1;
					}
				}
				for (i=0; i<nprocs-1; i++){
					context->communicator->lnrows[i] = context->communicator->lfRow[i+1] - context->communicator->lfRow[i] ;
					context->communicator->lnEnts[i] = context->communicator->lfEnt[i+1] - context->communicator->lfEnt[i] ;
				}

				context->communicator->lnrows[nprocs-1] = context->gnrows - context->communicator->lfRow[nprocs-1] ;
				context->communicator->lnEnts[nprocs-1] = gnnz - context->communicator->lfEnt[nprocs-1];

				//fclose(filed);
			}
			MPI_safecall(MPI_Bcast(context->communicator->lfRow,  nprocs, ghost_mpi_dt_midx, 0, context->mpicomm));
			MPI_safecall(MPI_Bcast(context->communicator->lfEnt,  nprocs, ghost_mpi_dt_midx, 0, context->mpicomm));
			MPI_safecall(MPI_Bcast(context->communicator->lnrows, nprocs, ghost_mpi_dt_midx, 0, context->mpicomm));
			MPI_safecall(MPI_Bcast(context->communicator->lnEnts, nprocs, ghost_mpi_dt_midx, 0, context->mpicomm));


		} else
		{ // don't read rpt, only fill lfrow, lnrows
			UNUSED(matrixPath);
			int me = ghost_getRank(context->mpicomm);
			double allweights;
			MPI_safecall(MPI_Allreduce(&weight,&allweights,1,MPI_DOUBLE,MPI_SUM,context->mpicomm))

				ghost_midx_t my_target_rows = (ghost_midx_t)(context->gnrows*((double)weight/(double)allweights));
			ghost_midx_t target_rows[nprocs];

			MPI_safecall(MPI_Allgather(&my_target_rows,1,ghost_mpi_dt_midx,&target_rows[me],1,ghost_mpi_dt_midx,context->mpicomm));

			context->rpt = NULL;
			context->communicator->lfRow[0] = 0;

			for (i=1; i<nprocs; i++){
				context->communicator->lfRow[i] = context->communicator->lfRow[i-1]+target_rows[i-1];
			}
			for (i=0; i<nprocs-1; i++){
				context->communicator->lnrows[i] = context->communicator->lfRow[i+1] - context->communicator->lfRow[i] ;
			}
			context->communicator->lnrows[nprocs-1] = context->gnrows - context->communicator->lfRow[nprocs-1] ;
			MPI_safecall(MPI_Bcast(context->communicator->lfRow,  nprocs, ghost_mpi_dt_midx, 0, context->mpicomm));
			MPI_safecall(MPI_Bcast(context->communicator->lnrows, nprocs, ghost_mpi_dt_midx, 0, context->mpicomm));
		}

	} else {
		context->communicator = NULL;
	}

#else
	UNUSED(weight);
	context->communicator = NULL;
#endif

	DEBUG_LOG(1,"Context created successfully");
	return context;
}


void ghost_freeContext(ghost_context_t *context)
{
	DEBUG_LOG(1,"Freeing context");
	if (context != NULL) {
		free(context->solvers);
		free(context->rowPerm);
		free(context->invRowPerm);
		ghost_freeCommunicator(context->communicator);

		free(context);
	}
	DEBUG_LOG(1,"Context freed successfully");
}
