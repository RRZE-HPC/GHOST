#define _GNU_SOURCE
#include "ghost_mpi_util.h"
#include "ghost.h"
#include "ghost_util.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/param.h>
#include <sys/syscall.h>
#include <stdlib.h>
#include <sched.h>

#include <math.h>
#include <omp.h>
#include <complex.h>
#include <dlfcn.h>

#define MAX_NUM_THREADS 128

static MPI_Comm single_node_comm;

static int getProcessorId() 
{

	cpu_set_t  cpu_set;
	int processorId;

	CPU_ZERO(&cpu_set);
	sched_getaffinity((pid_t)0,sizeof(cpu_set_t), &cpu_set);

	for (processorId=0;processorId<MAX_NUM_THREADS;processorId++){
		if (CPU_ISSET(processorId,&cpu_set))
		{  
			break;
		}
	}
	return processorId;
}

MPI_Comm getSingleNodeComm()
{

	return single_node_comm;
}

void setupSingleNodeComm() 
{

	/* return MPI communicator between nodal MPI processes single_node_comm
	 * and process rank me_node on local node */

	int i, coreId, me, n_nodes, me_node;
	char **all_hostnames;
	char *all_hn_mem;
	char hostname[MAXHOSTNAMELEN];
	gethostname(hostname,MAXHOSTNAMELEN);

	size_t size_ahnm, size_ahn, size_nint;
	int *mymate, *acc_mates;

	MPI_safecall(MPI_Comm_size ( MPI_COMM_WORLD, &n_nodes ));
	MPI_safecall(MPI_Comm_rank ( MPI_COMM_WORLD, &me ));

	coreId = getProcessorId();

	size_ahnm = (size_t)( MAXHOSTNAMELEN*n_nodes * sizeof(char) );
	size_ahn  = (size_t)( n_nodes    * sizeof(char*) );
	size_nint = (size_t)( n_nodes    * sizeof(int) );

	mymate        = (int*)      malloc( size_nint);
	acc_mates     = (int*)      malloc( size_nint );
	all_hn_mem    = (char*)     malloc( size_ahnm );
	all_hostnames = (char**)    malloc( size_ahn );

	for (i=0; i<n_nodes; i++){
		all_hostnames[i] = &all_hn_mem[i*MAXHOSTNAMELEN];
		mymate[i] = 0;
	}

	/* write local hostname to all_hostnames and share */
	MPI_safecall(MPI_Allgather ( hostname, MAXHOSTNAMELEN, MPI_CHAR, 
				&all_hostnames[0][0], MAXHOSTNAMELEN, MPI_CHAR, MPI_COMM_WORLD ));

	/* one process per node writes its global id to all its mates' fields */ 
	if (coreId==0){
		for (i=0; i<n_nodes; i++){
			if ( strcmp (hostname, all_hostnames[i]) == 0) mymate[i]=me;
		}
	}  

	MPI_safecall(MPI_Allreduce( mymate, acc_mates, n_nodes, MPI_INT, MPI_SUM, MPI_COMM_WORLD)); 
	/* all processes should now have the rank of their coreId 0 process in their acc_mate field;
	 * split into comm groups with this rank as communicator ID */
	MPI_safecall(MPI_Comm_split ( MPI_COMM_WORLD, acc_mates[me], me, &single_node_comm ));
	MPI_safecall(MPI_Comm_rank ( single_node_comm, &me_node));

	DEBUG_LOG(1,"Rank in single node comm: %d", me_node);

	free( mymate );
	free( acc_mates );
	free( all_hn_mem );
	free( all_hostnames );
}

int ghost_mpi_dataType(int datatype)
{
	if (datatype & GHOST_BINCRS_DT_FLOAT) {
		if (datatype & GHOST_BINCRS_DT_COMPLEX)
			return 0; // TODO
		else
			return MPI_FLOAT;
	} else {
		if (datatype & GHOST_BINCRS_DT_COMPLEX)
			return 0;
		else
			return MPI_DOUBLE;
	}
}

/*
void ghost_createDistributedContextSerial(ghost_context_t * context, CR_TYPE* cr, int options, ghost_mtraits_t *traits)
{

	UNUSED(context);
	UNUSED(cr);
	UNUSED(options);
	UNUSED(traits);
		ghost_midx_t i;

		int me; 

		ghost_comm_t *lcrp;

		unsigned int nprocs = ghost_getNumberOfProcesses();

		MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD, &me));

		DEBUG_LOG(1,"Entering context_communication");
		MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));

		lcrp = (ghost_comm_t*) allocateMemory( sizeof(ghost_comm_t), "lcrp");

		context->communicator = lcrp;
		context->fullMatrix = (ghost_mat_t *)allocateMemory(sizeof(ghost_mat_t),"fullMatrix");
		context->localMatrix = (ghost_mat_t *)allocateMemory(sizeof(ghost_mat_t),"fullMatrix");
		context->remoteMatrix = (ghost_mat_t *)allocateMemory(sizeof(ghost_mat_t),"fullMatrix");

	 *context->fullMatrix = (ghost_mat_t)MATRIX_INIT(.trait=traits[0]);
	 *context->localMatrix = (ghost_mat_t)MATRIX_INIT(.trait=traits[1]);
	 *context->remoteMatrix = (ghost_mat_t)MATRIX_INIT(.trait=traits[2]);

	 lcrp->wishes   = (ghost_mnnz_t*)       allocateMemory( nprocs*sizeof(ghost_mnnz_t), "lcrp->wishes" ); 
	 lcrp->dues     = (ghost_mnnz_t*)       allocateMemory( nprocs*sizeof(ghost_mnnz_t), "lcrp->dues" ); 


	 ghost_createDistribution(cr,options,lcrp);

	 CR_TYPE *fullCR;
	 fullCR = (CR_TYPE *) allocateMemory(sizeof(CR_TYPE),"fullCR");

	 fullCR->val = (ghost_mdat_t*) allocateMemory(lcrp->lnEnts[me]*sizeof( ghost_mdat_t ),"fullMatrix->val" );
	 fullCR->col = (ghost_midx_t*) allocateMemory(lcrp->lnEnts[me]*sizeof( ghost_midx_t ),"fullMatrix->col" ); 
	 fullCR->rpt = (ghost_midx_t*) allocateMemory((lcrp->lnrows[me]+1)*sizeof( ghost_midx_t ),"fullMatrix->rpt" ); 
	 fullCR->nrows = lcrp->lnrows[me];
	 fullCR->nEnts = lcrp->lnEnts[me];

	 context->fullMatrix->data = fullCR;
	 context->fullMatrix->nnz = lcrp->lnEnts[me];
	 context->fullMatrix->nrows = lcrp->lnrows[me];

#pragma omp parallel for schedule(runtime)
for (i=0; i<lcrp->lnEnts[me]; i++) fullCR->val[i] = 0.0;

#pragma omp parallel for schedule(runtime)
for (i=0; i<lcrp->lnEnts[me]; i++) fullCR->col[i] = 0.0;

#pragma omp parallel for schedule(runtime)
for (i=0; i<lcrp->lnrows[me]; i++) fullCR->rpt[i] = 0.0;


MPI_safecall(MPI_Scatterv ( cr->val, (int *)lcrp->lnEnts, (int *)lcrp->lfEnt, ghost_mpi_dt_mdat, 
fullCR->val, (int)lcrp->lnEnts[me],  ghost_mpi_dt_mdat, 0, MPI_COMM_WORLD));

MPI_safecall(MPI_Scatterv ( cr->col, (int *)lcrp->lnEnts, (int *)lcrp->lfEnt, MPI_INTEGER,
fullCR->col, (int)lcrp->lnEnts[me],  MPI_INTEGER, 0, MPI_COMM_WORLD));

MPI_safecall(MPI_Scatterv ( cr->rpt, (int *)lcrp->lnrows, (int *)lcrp->lfRow, MPI_INTEGER,
fullCR->rpt, (int)lcrp->lnrows[me],  MPI_INTEGER, 0, MPI_COMM_WORLD));


ghost_createCommunication(fullCR,options,context);
}*/


/*void ghost_createDistribution(CR_TYPE *cr, int options, ghost_comm_t *lcrp)
{
	
	int me = ghost_getRank(); 
	ghost_mnnz_t j;
	ghost_midx_t i;
	int hlpi;
	int target_rows;
	int nprocs = ghost_getNumberOfProcesses();

	lcrp->lnEnts   = (ghost_mnnz_t*)       allocateMemory( nprocs*sizeof(ghost_mnnz_t), "lcrp->lnEnts" ); 
	lcrp->lnrows   = (ghost_midx_t*)       allocateMemory( nprocs*sizeof(ghost_midx_t), "lcrp->lnrows" ); 
	lcrp->lfEnt    = (ghost_mnnz_t*)       allocateMemory( nprocs*sizeof(ghost_mnnz_t), "lcrp->lfEnt" ); 
	lcrp->lfRow    = (ghost_midx_t*)       allocateMemory( nprocs*sizeof(ghost_midx_t), "lcrp->lfRow" ); 

	if (me==0){

		if (options & GHOST_OPTION_WORKDIST_NZE){
			DEBUG_LOG(1,"Distribute Matrix with EQUAL_NZE on each PE");
			ghost_mnnz_t target_nnz;

			target_nnz = (cr->nEnts/nprocs)+1; 

			lcrp->lfRow[0]  = 0;
			lcrp->lfEnt[0] = 0;
			j = 1;

			for (i=0;i<cr->nrows;i++){
				if (cr->rpt[i] >= j*target_nnz){
					lcrp->lfRow[j] = i;
					lcrp->lfEnt[j] = cr->rpt[i];
					j = j+1;
				}
			}

		}
		else if (options & GHOST_OPTION_WORKDIST_LNZE){
			if (!(options & GHOST_OPTION_SERIAL_IO)) {
				DEBUG_LOG(0,"Warning! GHOST_OPTION_WORKDIST_LNZE has not (yet) been "
						"implemented for parallel IO! Switching to "
						"GHOST_OPTION_WORKDIST_NZE");
				options |= GHOST_OPTION_WORKDIST_NZE;
			} else {
				DEBUG_LOG(1,"Distribute Matrix with EQUAL_LNZE on each PE");
				ghost_mnnz_t *loc_count;
				int target_lnze;

				int trial_count, prev_count, trial_rows;
				int ideal, prev_rows;
				int outer_iter, outer_convergence;

				target_rows = (cr->nrows/nprocs);

				lcrp->lfRow[0] = 0;
				lcrp->lfEnt[0] = 0;

				for (i=1; i<nprocs; i++){
					lcrp->lfRow[i] = lcrp->lfRow[i-1]+target_rows;
					lcrp->lfEnt[i] = cr->rpt[lcrp->lfRow[i]];
				}

				for (i=0; i<nprocs-1; i++){
					lcrp->lnrows[i] = lcrp->lfRow[i+1] - lcrp->lfRow[i] ;
					lcrp->lnEnts[i] = lcrp->lfEnt[i+1] - lcrp->lfEnt[i] ;
				}
				lcrp->lnrows[nprocs-1] = cr->nrows - lcrp->lfRow[nprocs-1] ;
				lcrp->lnEnts[nprocs-1] = cr->nEnts - lcrp->lfEnt[nprocs-1];

				loc_count      = (ghost_mnnz_t*)       allocateMemory( nprocs*sizeof(ghost_mnnz_t), "loc_count" ); 
				for (i=0; i<nprocs; i++) loc_count[i] = 0;     

				for (i=0; i<nprocs; i++){
					for (j=lcrp->lfEnt[i]; j<lcrp->lfEnt[i]+lcrp->lnEnts[i]; j++){
						if (cr->col[j] >= lcrp->lfRow[i] && 
								cr->col[j]<lcrp->lfRow[i]+lcrp->lnrows[i])
							loc_count[i]++;
					}
				}
				DEBUG_LOG(2,"First run: local elements:");
				hlpi = 0;
				for (i=0; i<nprocs; i++){
					hlpi += loc_count[i];
					DEBUG_LOG(2,"Block %"PRmatIDX" %"PRmatNNZ" %"PRmatIDX, i, loc_count[i], lcrp->lnEnts[i]);
				}
				target_lnze = hlpi/nprocs;
				DEBUG_LOG(2,"total local elements: %d | per PE: %d", hlpi, target_lnze);

				outer_convergence = 0; 
				outer_iter = 0;

				while(outer_convergence==0){ 

					DEBUG_LOG(2,"Convergence Iteration %d", outer_iter);

					for (i=0; i<nprocs-1; i++){

						ideal = 0;
						prev_rows  = lcrp->lnrows[i];
						prev_count = loc_count[i];

						while (ideal==0){

							trial_rows = (int)( (double)(prev_rows) * sqrt((1.0*target_lnze)/(1.0*prev_count)) );

							if ( (trial_rows-prev_rows)*(trial_rows-prev_rows)<5.0 ) ideal=1;

							trial_count = 0;
							for (j=lcrp->lfEnt[i]; j<cr->rpt[lcrp->lfRow[i]+trial_rows]; j++){
								if (cr->col[j] >= lcrp->lfRow[i] && cr->col[j]<lcrp->lfRow[i]+trial_rows)
									trial_count++;
							}
							prev_rows  = trial_rows;
							prev_count = trial_count;
						}

						lcrp->lnrows[i]  = trial_rows;
						loc_count[i]     = trial_count;
						lcrp->lfRow[i+1] = lcrp->lfRow[i]+lcrp->lnrows[i];
						if (lcrp->lfRow[i+1]>cr->nrows) DEBUG_LOG(0,"Exceeded matrix dimension");
						lcrp->lfEnt[i+1] = cr->rpt[lcrp->lfRow[i+1]];
						lcrp->lnEnts[i] = lcrp->lfEnt[i+1] - lcrp->lfEnt[i] ;

					}

					lcrp->lnrows[nprocs-1] = cr->nrows - lcrp->lfRow[nprocs-1];
					lcrp->lnEnts[nprocs-1] = cr->nEnts - lcrp->lfEnt[nprocs-1];
					loc_count[nprocs-1] = 0;
					for (j=lcrp->lfEnt[nprocs-1]; j<cr->nEnts; j++)
						if (cr->col[j] >= lcrp->lfRow[nprocs-1]) loc_count[nprocs-1]++;

					DEBUG_LOG(2,"Next run: outer_iter=%d:", outer_iter);
					hlpi = 0;
					for (i=0; i<nprocs; i++){
						hlpi += loc_count[i];
						DEBUG_LOG(2,"Block %"PRmatIDX" %"PRmatNNZ" %"PRmatNNZ, i, loc_count[i], lcrp->lnEnts[i]);
					}
					target_lnze = hlpi/nprocs;
					DEBUG_LOG(2,"total local elements: %d | per PE: %d | total share: %6.3f%%",
							hlpi, target_lnze, 100.0*hlpi/(1.0*cr->nEnts));

					hlpi = 0;
					for (i=0; i<nprocs; i++) if ( (1.0*(loc_count[i]-target_lnze))/(1.0*target_lnze)>0.001) hlpi++;
					if (hlpi == 0) outer_convergence = 1;

					outer_iter++;

					if (outer_iter>20){
						DEBUG_LOG(0,"No convergence after 20 iterations, exiting iteration.");
						outer_convergence = 1;
					}

				}

				int p;
				for (p=0; i<nprocs; i++)  
					DEBUG_LOG(1,"PE%d lfRow=%"PRmatIDX" lfEnt=%"PRmatNNZ" lnrows=%"PRmatIDX" lnEnts=%"PRmatNNZ, p, lcrp->lfRow[i], lcrp->lfEnt[i], lcrp->lnrows[i], lcrp->lnEnts[i]);

				free(loc_count);
			}
		}
		else {

			DEBUG_LOG(1,"Distribute Matrix with EQUAL_ROWS on each PE");
			target_rows = (cr->nrows/nprocs);

			lcrp->lfRow[0] = 0;
			lcrp->lfEnt[0] = 0;

			for (i=1; i<nprocs; i++){
				lcrp->lfRow[i] = lcrp->lfRow[i-1]+target_rows;
				lcrp->lfEnt[i] = cr->rpt[lcrp->lfRow[i]];
			}
		}


		for (i=0; i<nprocs-1; i++){
			lcrp->lnrows[i] = lcrp->lfRow[i+1] - lcrp->lfRow[i] ;
			lcrp->lnEnts[i] = lcrp->lfEnt[i+1] - lcrp->lfEnt[i] ;
		}

		lcrp->lnrows[nprocs-1] = cr->nrows - lcrp->lfRow[nprocs-1] ;
		lcrp->lnEnts[nprocs-1] = cr->nEnts - lcrp->lfEnt[nprocs-1];
	}

	MPI_safecall(MPI_Bcast(lcrp->lfRow,  nprocs, MPI_INTEGER, 0, MPI_COMM_WORLD));
	MPI_safecall(MPI_Bcast(lcrp->lfEnt,  nprocs, MPI_INTEGER, 0, MPI_COMM_WORLD));
	MPI_safecall(MPI_Bcast(lcrp->lnrows, nprocs, MPI_INTEGER, 0, MPI_COMM_WORLD));
	MPI_safecall(MPI_Bcast(lcrp->lnEnts, nprocs, MPI_INTEGER, 0, MPI_COMM_WORLD));


}

void ghost_createCommunication(CR_TYPE *fullCR, CR_TYPE **localCR, CR_TYPE **remoteCR, int options, ghost_context_t *context)
{
	
	DEBUG_LOG(1,"Setting up communication");

	int hlpi;
	ghost_mnnz_t j;
	ghost_midx_t i;
	int me;
	ghost_mnnz_t max_loc_elements, thisentry;
	int *present_values;
	int acc_dues;
	int *tmp_transfers;
	int acc_wishes;
	int nEnts_glob;

	int *item_from;

	ghost_mnnz_t *wishlist_counts;

	int *wishlist_mem,  **wishlist;
	int *cwishlist_mem, **cwishlist;

	int this_pseudo_col;
	int *pseudocol;
	int *globcol;
	int *revcol;

	int *comm_remotePE, *comm_remoteEl;
	ghost_mnnz_t lnEnts_l, lnEnts_r;
	int current_l, current_r;
	ghost_midx_t pseudo_ldim;
	int acc_transfer_wishes, acc_transfer_dues;

	size_t size_nint, size_col;
	size_t size_revc, size_a2ai, size_nptr, size_pval;  
	size_t size_mem, size_wish, size_dues;

	int nprocs = ghost_getNumberOfProcesses();
	ghost_comm_t *lcrp = context->communicator;

	size_nint = (size_t)( (size_t)(nprocs)   * sizeof(int)  );
	size_nptr = (size_t)( nprocs             * sizeof(int*) );
	size_a2ai = (size_t)( nprocs*nprocs * sizeof(int)  );

	me = ghost_getRank();

	max_loc_elements = 0;
	for (i=0;i<nprocs;i++)
		if (max_loc_elements<lcrp->lnrows[i]) max_loc_elements = lcrp->lnrows[i];

	nEnts_glob = lcrp->lfEnt[nprocs-1]+lcrp->lnEnts[nprocs-1]; 

	size_pval = (size_t)( max_loc_elements * sizeof(int) );
	size_revc = (size_t)( nEnts_glob       * sizeof(int) );
	size_col  = (size_t)( (size_t)(lcrp->lnEnts[me])   * sizeof( int ) );


	item_from       = (int*) allocateMemory( size_nint, "item_from" ); 
	wishlist_counts = (ghost_mnnz_t *) allocateMemory( nprocs*sizeof(ghost_mnnz_t), "wishlist_counts" ); 
	comm_remotePE   = (int*) allocateMemory( size_col,  "comm_remotePE" );
	comm_remoteEl   = (int*) allocateMemory( size_col,  "comm_remoteEl" );
	present_values  = (int*) allocateMemory( size_pval, "present_values" ); 
	tmp_transfers   = (int*) allocateMemory( size_a2ai, "tmp_transfers" ); 
	pseudocol       = (int*) allocateMemory( size_col,  "pseudocol" );
	globcol         = (int*) allocateMemory( size_col,  "origincol" );
	revcol          = (int*) allocateMemory( size_revc, "revcol" );


	for (i=0; i<nprocs; i++) wishlist_counts[i] = 0;

	for (i=0;i<lcrp->lnEnts[me];i++){
		for (j=nprocs-1;j>=0; j--){
			if (lcrp->lfRow[j]<fullCR->col[i]+1) {
				comm_remotePE[i] = j;
				wishlist_counts[j]++;
				comm_remoteEl[i] = fullCR->col[i] -lcrp->lfRow[j];
				break;
			}
		}
	}

	acc_wishes = 0;
	for (i=0; i<nprocs; i++) acc_wishes += wishlist_counts[i];
	size_mem  = (size_t)( acc_wishes * sizeof(int) );

	wishlist        = (int**) allocateMemory( size_nptr, "wishlist" ); 
	cwishlist       = (int**) allocateMemory( size_nptr, "cwishlist" ); 
	wishlist_mem    = (int*)  allocateMemory( size_mem,  "wishlist_mem" ); 
	cwishlist_mem   = (int*)  allocateMemory( size_mem,  "cwishlist_mem" ); 

	hlpi = 0;
	for (i=0; i<nprocs; i++){
		wishlist[i]  = &wishlist_mem[hlpi];
		cwishlist[i] = &cwishlist_mem[hlpi];
		hlpi += wishlist_counts[i];
	}

	for (i=0;i<nprocs;i++) item_from[i] = 0;

	for (i=0;i<lcrp->lnEnts[me];i++){
		wishlist[comm_remotePE[i]][item_from[comm_remotePE[i]]] = comm_remoteEl[i];
		item_from[comm_remotePE[i]]++;
	}

	for (i=0; i<nprocs; i++){

		for (j=0; j<max_loc_elements; j++) present_values[j] = -1;

		if ( (i!=me) && (wishlist_counts[i]>0) ){
			thisentry = 0;
			for (j=0; j<wishlist_counts[i]; j++){
				if (present_values[wishlist[i][j]]<0){
					present_values[wishlist[i][j]] = thisentry;
					cwishlist[i][thisentry] = wishlist[i][j];
					thisentry = thisentry + 1;
				}
			}
			lcrp->wishes[i] = thisentry;
		}
		else lcrp->wishes[i] = 0; 

	}

	MPI_safecall(MPI_Allgather ( lcrp->wishes, nprocs, MPI_INTEGER, tmp_transfers, 
				nprocs, MPI_INTEGER, MPI_COMM_WORLD )) ;

	for (i=0; i<nprocs; i++) lcrp->dues[i] = tmp_transfers[i*nprocs+me];

	lcrp->dues[me] = 0; 

	acc_transfer_dues = 0;
	acc_transfer_wishes = 0;
	for (i=0; i<nprocs; i++){
		acc_transfer_wishes += lcrp->wishes[i];
		acc_transfer_dues   += lcrp->dues[i];
	}

	this_pseudo_col = lcrp->lnrows[me];
	lcrp->halo_elements = 0;
	for (i=0; i<nprocs; i++){
		if (i != me){ 
			for (j=0;j<lcrp->wishes[i];j++){
				pseudocol[lcrp->halo_elements] = this_pseudo_col;  
				globcol[lcrp->halo_elements]   = lcrp->lfRow[i]+cwishlist[i][j]; 
				revcol[globcol[lcrp->halo_elements]] = lcrp->halo_elements;
				lcrp->halo_elements++;
				this_pseudo_col++;
			}
		}
	}

	for (i=0;i<lcrp->lnEnts[me];i++)
	{
		if (comm_remotePE[i] == me) // local
			fullCR->col[i] =  comm_remoteEl[i];
		else // remote
			fullCR->col[i] = pseudocol[revcol[fullCR->col[i]]];
	}

	freeMemory ( size_col,  "comm_remoteEl",  comm_remoteEl);
	freeMemory ( size_col,  "comm_remotePE",  comm_remotePE);
	freeMemory ( size_col,  "pseudocol",      pseudocol);
	freeMemory ( size_col,  "globcol",        globcol);
	freeMemory ( size_revc, "revcol",         revcol);
	freeMemory ( size_pval, "present_values", present_values ); 

	size_wish = (size_t)( acc_transfer_wishes * sizeof(int) );
	size_dues = (size_t)( acc_transfer_dues   * sizeof(int) );

	MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));

	lcrp->wishlist      = (int**) allocateMemory( size_nptr, "lcrp->wishlist" ); 
	lcrp->duelist       = (int**) allocateMemory( size_nptr, "lcrp->duelist" ); 
	lcrp->wishlist_mem  = (int*)  allocateMemory( size_wish, "lcrp->wishlist_mem" ); 
	lcrp->duelist_mem   = (int*)  allocateMemory( size_dues, "lcrp->duelist_mem" ); 
	lcrp->wish_displ    = (int*)  allocateMemory( size_nint, "lcrp->wish_displ" ); 
	lcrp->due_displ     = (int*)  allocateMemory( size_nint, "lcrp->due_displ" ); 
	lcrp->hput_pos      = (int*)  allocateMemory( size_nptr, "lcrp->hput_pos" ); 

	acc_dues = 0;
	acc_wishes = 0;



	for (i=0; i<nprocs; i++){

		lcrp->due_displ[i]  = acc_dues;
		lcrp->wish_displ[i] = acc_wishes;
		lcrp->duelist[i]    = &(lcrp->duelist_mem[acc_dues]);
		lcrp->wishlist[i]   = &(lcrp->wishlist_mem[acc_wishes]);
		lcrp->hput_pos[i]   = lcrp->lnrows[me]+acc_wishes;

		if  ( (me != i) && !( (i == nprocs-2) && (me == nprocs-1) ) ){
			acc_dues   += lcrp->dues[i];
			acc_wishes += lcrp->wishes[i];
		}
	}

	for (i=0; i<nprocs; i++) for (j=0;j<lcrp->wishes[i];j++)
		lcrp->wishlist[i][j] = cwishlist[i][j]; 

	for(i=0; i<nprocs; i++) {
		MPI_safecall(MPI_Scatterv ( 
					lcrp->wishlist_mem, (int *)lcrp->wishes, lcrp->wish_displ, MPI_INTEGER, 
					lcrp->duelist[i], (int)lcrp->dues[i], MPI_INTEGER, i, MPI_COMM_WORLD ));
	}

	if (!(options & GHOST_OPTION_NO_SPLIT_SOLVERS)) { // split computation

		pseudo_ldim = lcrp->lnrows[me]+lcrp->halo_elements ;

		lnEnts_l=0;
		for (i=0; i<lcrp->lnEnts[me];i++) {
			if (fullCR->col[i]<lcrp->lnrows[me]) lnEnts_l++;
		}


		lnEnts_r = lcrp->lnEnts[me]-lnEnts_l;

		DEBUG_LOG(1,"PE%d: Rows=%"PRmatIDX"\t Ents=%"PRmatNNZ"(l),%"PRmatNNZ"(r),%"PRmatNNZ"(g)\t pdim=%"PRmatIDX, 
				me, lcrp->lnrows[me], lnEnts_l, lnEnts_r, lcrp->lnEnts[me], pseudo_ldim );

		//CR_TYPE *localCR;
		//CR_TYPE *remoteCR;


		(*localCR) = (CR_TYPE *) allocateMemory(sizeof(CR_TYPE),"fullCR");
		(*remoteCR) = (CR_TYPE *) allocateMemory(sizeof(CR_TYPE),"fullCR");

		(*localCR)->val = (ghost_mdat_t*) allocateMemory(lnEnts_l*sizeof( ghost_mdat_t ),"localMatrix->val" ); 
		(*localCR)->col = (ghost_midx_t*) allocateMemory(lnEnts_l*sizeof( ghost_midx_t ),"localMatrix->col" ); 
		(*localCR)->rpt = (ghost_midx_t*) allocateMemory((lcrp->lnrows[me]+1)*sizeof( ghost_midx_t ),"localMatrix->rpt" ); 

		(*remoteCR)->val = (ghost_mdat_t*) allocateMemory(lnEnts_r*sizeof( ghost_mdat_t ),"remoteMatrix->val" ); 
		(*remoteCR)->col = (ghost_midx_t*) allocateMemory(lnEnts_r*sizeof( ghost_midx_t ),"remoteMatrix->col" ); 
		(*remoteCR)->rpt = (ghost_midx_t*) allocateMemory((lcrp->lnrows[me]+1)*sizeof( ghost_midx_t ),"remoteMatrix->rpt" ); 

		//context->localMatrix->data = localCR;
		//context->remoteMatrix->data = remoteCR;

		(*localCR)->nrows = lcrp->lnrows[me];
		(*localCR)->nEnts = lnEnts_l;

		(*remoteCR)->nrows = lcrp->lnrows[me];
		(*remoteCR)->nEnts = lnEnts_r;

		//context->localMatrix->data = localCR;
		//context->localMatrix->nnz = lnEnts_l;
		//context->localMatrix->nrows = lcrp->lnrows[me];

		//context->remoteMatrix->data = remoteCR;
		//context->localMatrix->nnz = lnEnts_r;
		//context->localMatrix->nrows = lcrp->lnrows[me];

#pragma omp parallel for schedule(runtime)
		for (i=0; i<lnEnts_l; i++) (*localCR)->val[i] = 0.0;

#pragma omp parallel for schedule(runtime)
		for (i=0; i<lnEnts_l; i++) (*localCR)->col[i] = 0.0;

#pragma omp parallel for schedule(runtime)
		for (i=0; i<lnEnts_r; i++) (*remoteCR)->val[i] = 0.0;

#pragma omp parallel for schedule(runtime)
		for (i=0; i<lnEnts_r; i++) (*remoteCR)->col[i] = 0.0;


		(*localCR)->rpt[0] = 0;
		(*remoteCR)->rpt[0] = 0;

		MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));
		DEBUG_LOG(1,"PE%d: lnrows=%"PRmatIDX" row_ptr=%"PRmatIDX"..%"PRmatIDX,
				me, lcrp->lnrows[me], fullCR->rpt[0], fullCR->rpt[lcrp->lnrows[me]]);
		fflush(stdout);
		MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));

		for (i=0; i<lcrp->lnrows[me]; i++){

			current_l = 0;
			current_r = 0;

			for (j=fullCR->rpt[i]; j<fullCR->rpt[i+1]; j++){

				if (fullCR->col[j]<lcrp->lnrows[me]){
					(*localCR)->col[ (*localCR)->rpt[i]+current_l ] = fullCR->col[j]; 
					(*localCR)->val[ (*localCR)->rpt[i]+current_l ] = fullCR->val[j]; 
					current_l++;
				}
				else{
					(*remoteCR)->col[ (*remoteCR)->rpt[i]+current_r ] = fullCR->col[j];
					(*remoteCR)->val[ (*remoteCR)->rpt[i]+current_r ] = fullCR->val[j];
					current_r++;
				}

			}  

			(*localCR)->rpt[i+1] = (*localCR)->rpt[i] + current_l;
			(*remoteCR)->rpt[i+1] = (*remoteCR)->rpt[i] + current_r;
		}

		IF_DEBUG(3){
			for (i=0; i<lcrp->lnrows[me]+1; i++)
				DEBUG_LOG(3,"--Row_ptrs-- PE %d: i=%"PRmatIDX" local=%"PRmatIDX" remote=%"PRmatIDX, 
						me, i, (*localCR)->rpt[i], (*remoteCR)->rpt[i]);
			for (i=0; i<(*localCR)->rpt[lcrp->lnrows[me]]; i++)
				DEBUG_LOG(3,"-- local -- PE%d: localCR->col[%"PRmatIDX"]=%"PRmatIDX, me, i, (*localCR)->col[i]);
			for (i=0; i<(*remoteCR)->rpt[lcrp->lnrows[me]]; i++)
				DEBUG_LOG(3,"-- remote -- PE%d: remoteCR->col[%"PRmatIDX"]=%"PRmatIDX, me, i, (*remoteCR)->col[i]);
		}
		fflush(stdout);
		MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));

	}
	freeMemory ( size_mem,  "wishlist_mem",    wishlist_mem);
	freeMemory ( size_mem,  "cwishlist_mem",   cwishlist_mem);
	freeMemory ( size_nptr, "wishlist",        wishlist);
	freeMemory ( size_nptr, "cwishlist",       cwishlist);
	freeMemory ( size_a2ai, "tmp_transfers",   tmp_transfers);
	freeMemory ( size_nint, "wishlist_counts", wishlist_counts);
	freeMemory ( size_nint, "item_from",       item_from);

}*/
