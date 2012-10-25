#define _GNU_SOURCE
#include "mpihelper.h"
#include "matricks.h"
#include "spmvm.h"
#include "spmvm_util.h"

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

#define MAX_NUM_THREADS 128

static MPI_Comm single_node_comm;

static int getProcessorId() {

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

/******************************************************************************
 * Routine zum einmaligen Berechnen des Kommunikationsmusters fuer
 * den Fall einer seriell eingelesenen Matrix. Alle Matrixdaten liegen
 * auf PE 0 vor, dieser kannn alle Berechnungen durchfuehren und die
 * entsprechenden Daten dann an diejenigen PEs verteilen die es betrifft.
 *****************************************************************************/

LCRP_TYPE* setup_communication(CR_TYPE* cr, int options)
{

	/* Counting and auxilliary variables */
	int i, j, hlpi;

	/* Processor rank (MPI-process) */
	int me; 

	/* MPI-Errorcode */

	int max_loc_elements, thisentry;
	int *present_values;

	LCRP_TYPE *lcrp;

	int acc_dues;

	int *tmp_transfers;

	int target_nnz;
	int acc_wishes;

	int nEnts_glob;

	/* Counter how many entries are requested from each PE */
	int *item_from;

	int *wishlist_counts;

	int *wishlist_mem,  **wishlist;
	int *cwishlist_mem, **cwishlist;

	int this_pseudo_col;
	int *pseudocol;
	int *globcol;
	int *revcol;

	int *comm_remotePE, *comm_remoteEl;

	int target_rows;

	int lnEnts_l, lnEnts_r;
	int current_l, current_r;

	int target_lnze;

	int trial_count, prev_count, trial_rows;
	int *loc_count;
	int ideal, prev_rows;
	int outer_iter, outer_convergence;

	int pseudo_ldim;
	int acc_transfer_wishes, acc_transfer_dues;

	size_t size_nint, size_col, size_val, size_ptr;
	size_t size_lval, size_rval, size_lcol, size_rcol, size_lptr, size_rptr;
	size_t size_revc, size_a2ai, size_nptr, size_pval;  
	size_t size_mem, size_wish, size_dues;


	/****************************************************************************
	 *******            ........ Executable statements ........           *******
	 ***************************************************************************/

	MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD, &me));

	DEBUG_LOG(1,"Entering setup_communication");
	MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));

	lcrp = (LCRP_TYPE*) allocateMemory( sizeof(LCRP_TYPE), "lcrp");

	/*lcrp->fullRowPerm = NULL;
	lcrp->fullInvRowPerm = NULL;
	lcrp->splitRowPerm = NULL;
	lcrp->splitInvRowPerm = NULL;
	lcrp->fullMatrix = NULL;
	lcrp->localMatrix = NULL;
	lcrp->remoteMatrix = NULL;

	if (me==0) {
		lcrp->nEnts = cr->nEnts;
		lcrp->nRows = cr->nRows;
	}

	MPI_safecall(MPI_Bcast(&lcrp->nEnts,1,MPI_INT,0,MPI_COMM_WORLD));
	MPI_safecall(MPI_Bcast(&lcrp->nRows,1,MPI_INT,0,MPI_COMM_WORLD));*/

#pragma omp parallel
	lcrp->threads = omp_get_num_threads(); 

	MPI_safecall(MPI_Comm_size(MPI_COMM_WORLD, &(lcrp->nodes)));

	size_nint = (size_t)( (size_t)(lcrp->nodes)   * sizeof(int)  );
	size_nptr = (size_t)( lcrp->nodes             * sizeof(int*) );
	size_a2ai = (size_t)( lcrp->nodes*lcrp->nodes * sizeof(int)  );

	lcrp->lnEnts   = (int*)       allocateMemory( size_nint, "lcrp->lnEnts" ); 
	lcrp->lnRows   = (int*)       allocateMemory( size_nint, "lcrp->lnRows" ); 
	lcrp->lfEnt    = (int*)       allocateMemory( size_nint, "lcrp->lfEnt" ); 
	lcrp->lfRow    = (int*)       allocateMemory( size_nint, "lcrp->lfRow" ); 

	lcrp->wishes   = (int*)       allocateMemory( size_nint, "lcrp->wishes" ); 
	lcrp->dues     = (int*)       allocateMemory( size_nint, "lcrp->dues" ); 


	/****************************************************************************
	 *******  Calculate a fair partitioning of NZE and ROWS on master PE  *******
	 ***************************************************************************/
	if (me==0){

		if (options & SPMVM_OPTION_WORKDIST_NZE){
			DEBUG_LOG(1,"Distribute Matrix with EQUAL_NZE on each PE");
			target_nnz = (cr->nEnts/lcrp->nodes)+1; /* sonst bleiben welche uebrig! */

			lcrp->lfRow[0]  = 0;
			lcrp->lfEnt[0] = 0;
			j = 1;

			for (i=0;i<cr->nRows;i++){
				if (cr->rowOffset[i] >= j*target_nnz){
					lcrp->lfRow[j] = i;
					lcrp->lfEnt[j] = cr->rowOffset[i];
					j = j+1;
				}
			}

		}
		else if (options & SPMVM_OPTION_WORKDIST_LNZE){
			DEBUG_LOG(1,"Distribute Matrix with EQUAL_LNZE on each PE");

			/* A first attempt should be blocks of equal size */
			target_rows = (cr->nRows/lcrp->nodes);

			lcrp->lfRow[0] = 0;
			lcrp->lfEnt[0] = 0;

			for (i=1; i<lcrp->nodes; i++){
				lcrp->lfRow[i] = lcrp->lfRow[i-1]+target_rows;
				lcrp->lfEnt[i] = cr->rowOffset[lcrp->lfRow[i]];
			}

			for (i=0; i<lcrp->nodes-1; i++){
				lcrp->lnRows[i] = lcrp->lfRow[i+1] - lcrp->lfRow[i] ;
				lcrp->lnEnts[i] = lcrp->lfEnt[i+1] - lcrp->lfEnt[i] ;
			}
			lcrp->lnRows[lcrp->nodes-1] = cr->nRows - lcrp->lfRow[lcrp->nodes-1] ;
			lcrp->lnEnts[lcrp->nodes-1] = cr->nEnts - lcrp->lfEnt[lcrp->nodes-1];

			/* Count number of local elements in each block */
			loc_count      = (int*)       allocateMemory( size_nint, "loc_count" ); 
			for (i=0; i<lcrp->nodes; i++) loc_count[i] = 0;     

			for (i=0; i<lcrp->nodes; i++){
				for (j=lcrp->lfEnt[i]; j<lcrp->lfEnt[i]+lcrp->lnEnts[i]; j++){
					if (cr->col[j] >= lcrp->lfRow[i] && 
							cr->col[j]<lcrp->lfRow[i]+lcrp->lnRows[i])
						loc_count[i]++;
				}
			}
			DEBUG_LOG(2,"First run: local elements:");
			hlpi = 0;
			for (i=0; i<lcrp->nodes; i++){
				hlpi += loc_count[i];
				DEBUG_LOG(2,"Block %3d %8d %12d", i, loc_count[i], lcrp->lnEnts[i]);
			}
			target_lnze = hlpi/lcrp->nodes;
			DEBUG_LOG(2,"total local elements: %d | per PE: %d", hlpi, target_lnze);

			outer_convergence = 0; 
			outer_iter = 0;

			while(outer_convergence==0){ 

				DEBUG_LOG(2,"Convergence Iteration %d", outer_iter);

				for (i=0; i<lcrp->nodes-1; i++){

					ideal = 0;
					prev_rows  = lcrp->lnRows[i];
					prev_count = loc_count[i];

					while (ideal==0){

						trial_rows = (int)( (double)(prev_rows) * sqrt((1.0*target_lnze)/(1.0*prev_count)) );

						/* Check ob die Anzahl der Elemente schon das beste ist das ich erwarten kann */
						if ( (trial_rows-prev_rows)*(trial_rows-prev_rows)<5.0 ) ideal=1;

						trial_count = 0;
						for (j=lcrp->lfEnt[i]; j<cr->rowOffset[lcrp->lfRow[i]+trial_rows]; j++){
							if (cr->col[j] >= lcrp->lfRow[i] && cr->col[j]<lcrp->lfRow[i]+trial_rows)
								trial_count++;
						}
						prev_rows  = trial_rows;
						prev_count = trial_count;
					}

					lcrp->lnRows[i]  = trial_rows;
					loc_count[i]     = trial_count;
					lcrp->lfRow[i+1] = lcrp->lfRow[i]+lcrp->lnRows[i];
					if (lcrp->lfRow[i+1]>cr->nRows) DEBUG_LOG(0,"Exceeded matrix dimension");
					lcrp->lfEnt[i+1] = cr->rowOffset[lcrp->lfRow[i+1]];
					lcrp->lnEnts[i] = lcrp->lfEnt[i+1] - lcrp->lfEnt[i] ;

				}

				lcrp->lnRows[lcrp->nodes-1] = cr->nRows - lcrp->lfRow[lcrp->nodes-1];
				lcrp->lnEnts[lcrp->nodes-1] = cr->nEnts - lcrp->lfEnt[lcrp->nodes-1];
				loc_count[lcrp->nodes-1] = 0;
				for (j=lcrp->lfEnt[lcrp->nodes-1]; j<cr->nEnts; j++)
					if (cr->col[j] >= lcrp->lfRow[lcrp->nodes-1]) loc_count[lcrp->nodes-1]++;

				DEBUG_LOG(2,"Next run: outer_iter=%d:", outer_iter);
				hlpi = 0;
				for (i=0; i<lcrp->nodes; i++){
					hlpi += loc_count[i];
					DEBUG_LOG(2,"Block %3d %8d %12d", i, loc_count[i], lcrp->lnEnts[i]);
				}
				target_lnze = hlpi/lcrp->nodes;
				DEBUG_LOG(2,"total local elements: %d | per PE: %d | total share: %6.3f%%",
						hlpi, target_lnze, 100.0*hlpi/(1.0*cr->nEnts));

				hlpi = 0;
				for (i=0; i<lcrp->nodes; i++) if ( (1.0*(loc_count[i]-target_lnze))/(1.0*target_lnze)>0.001) hlpi++;
				if (hlpi == 0) outer_convergence = 1;

				outer_iter++;

				if (outer_iter>20){
					DEBUG_LOG(0,"No convergence after 20 iterations, exiting iteration.");
					outer_convergence = 1;
				}

			}

				for (i=0; i<lcrp->nodes; i++)  
					DEBUG_LOG(1,"PE%3d: lfRow=%8d lfEnt=%12d lnRows=%8d lnEnts=%12d", i, lcrp->lfRow[i], 
							lcrp->lfEnt[i], lcrp->lnRows[i], lcrp->lnEnts[i]);
			

			free(loc_count);
		}
		else {

			DEBUG_LOG(1,"Distribute Matrix with EQUAL_ROWS on each PE");
			target_rows = (cr->nRows/lcrp->nodes);

			lcrp->lfRow[0] = 0;
			lcrp->lfEnt[0] = 0;

			for (i=1; i<lcrp->nodes; i++){
				lcrp->lfRow[i] = lcrp->lfRow[i-1]+target_rows;
				lcrp->lfEnt[i] = cr->rowOffset[lcrp->lfRow[i]];
			}
		}


		for (i=0; i<lcrp->nodes-1; i++){
			lcrp->lnRows[i] = lcrp->lfRow[i+1] - lcrp->lfRow[i] ;
			lcrp->lnEnts[i] = lcrp->lfEnt[i+1] - lcrp->lfEnt[i] ;
		}

		lcrp->lnRows[lcrp->nodes-1] = cr->nRows - lcrp->lfRow[lcrp->nodes-1] ;
		lcrp->lnEnts[lcrp->nodes-1] = cr->nEnts - lcrp->lfEnt[lcrp->nodes-1];
	}

	/****************************************************************************
	 *******            Distribute correct share to all PEs               *******
	 ***************************************************************************/

	MPI_safecall(MPI_Bcast(lcrp->lfRow,  lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD));
	MPI_safecall(MPI_Bcast(lcrp->lfEnt,  lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD));
	MPI_safecall(MPI_Bcast(lcrp->lnRows, lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD));
	MPI_safecall(MPI_Bcast(lcrp->lnEnts, lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD));

	/****************************************************************************
	 *******   Allocate memory for matrix in distributed CRS storage      *******
	 ***************************************************************************/

	size_val  = (size_t)( (size_t)(lcrp->lnEnts[me])   * sizeof( mat_data_t ) );
	size_col  = (size_t)( (size_t)(lcrp->lnEnts[me])   * sizeof( int ) );
	size_ptr  = (size_t)( (size_t)(lcrp->lnRows[me]+1) * sizeof( int ) );

	lcrp->val      = (mat_data_t*)    allocateMemory( size_val,  "lcrp->val" ); 
	lcrp->col      = (int*)       allocateMemory( size_col,  "lcrp->col" ); 
	lcrp->lrow_ptr = (int*)       allocateMemory( size_ptr,  "lcrp->lrow_ptr" ); 

	/****************************************************************************
	 *******   Fill all fields with their corresponding values            *******
	 ***************************************************************************/


	#pragma omp parallel for schedule(runtime)
//#pragma omp parallel for schedule(static)
	for (i=0; i<lcrp->lnEnts[me]; i++) lcrp->val[i] = 0.0;

	#pragma omp parallel for schedule(runtime)
//#pragma omp parallel for schedule(static)
	for (i=0; i<lcrp->lnEnts[me]; i++) lcrp->col[i] = 0.0;

	#pragma omp parallel for schedule(runtime)
//#pragma omp parallel for schedule(static)
	for (i=0; i<lcrp->lnRows[me]; i++) lcrp->lrow_ptr[i] = 0.0;


	MPI_safecall(MPI_Scatterv ( cr->val, lcrp->lnEnts, lcrp->lfEnt, MPI_MYDATATYPE, 
				lcrp->val, lcrp->lnEnts[me],  MPI_MYDATATYPE, 0, MPI_COMM_WORLD));

	MPI_safecall(MPI_Scatterv ( cr->col, lcrp->lnEnts, lcrp->lfEnt, MPI_INTEGER,
				lcrp->col, lcrp->lnEnts[me],  MPI_INTEGER, 0, MPI_COMM_WORLD));

	MPI_safecall(MPI_Scatterv ( cr->rowOffset, lcrp->lnRows, lcrp->lfRow, MPI_INTEGER,
				lcrp->lrow_ptr, lcrp->lnRows[me],  MPI_INTEGER, 0, MPI_COMM_WORLD));

	/****************************************************************************
	 *******        Adapt row pointer to local numbering on this PE       *******         
	 ***************************************************************************/

	for (i=0;i<lcrp->lnRows[me]+1;i++)
		lcrp->lrow_ptr[i] =  lcrp->lrow_ptr[i] - lcrp->lfEnt[me]; 

	/* last entry of row_ptr holds the local number of entries */
	lcrp->lrow_ptr[lcrp->lnRows[me]] = lcrp->lnEnts[me]; 

	/****************************************************************************
	 *******         Extract maximum number of local elements             *******
	 ***************************************************************************/

	max_loc_elements = 0;
	for (i=0;i<lcrp->nodes;i++)
		if (max_loc_elements<lcrp->lnRows[i]) max_loc_elements = lcrp->lnRows[i];

	nEnts_glob = lcrp->lfEnt[lcrp->nodes-1]+lcrp->lnEnts[lcrp->nodes-1]; 


	/****************************************************************************
	 *******         Assemble wish- and duelists for communication        *******
	 ***************************************************************************/

	size_pval = (size_t)( max_loc_elements * sizeof(int) );
	size_revc = (size_t)( nEnts_glob       * sizeof(int) );

	item_from       = (int*) allocateMemory( size_nint, "item_from" ); 
	wishlist_counts = (int*) allocateMemory( size_nint, "wishlist_counts" ); 
	comm_remotePE   = (int*) allocateMemory( size_col,  "comm_remotePE" );
	comm_remoteEl   = (int*) allocateMemory( size_col,  "comm_remoteEl" );
	present_values  = (int*) allocateMemory( size_pval, "present_values" ); 
	tmp_transfers   = (int*) allocateMemory( size_a2ai, "tmp_transfers" ); 
	pseudocol       = (int*) allocateMemory( size_col,  "pseudocol" );
	globcol         = (int*) allocateMemory( size_col,  "origincol" );
	revcol          = (int*) allocateMemory( size_revc, "revcol" );


	for (i=0; i<lcrp->nodes; i++) wishlist_counts[i] = 0;

	/* Transform global column index into 2d-local/non-local index */
	for (i=0;i<lcrp->lnEnts[me];i++){
		for (j=lcrp->nodes-1;j>-1; j--){
			if (lcrp->lfRow[j]<lcrp->col[i]+1) {
				/* Entsprechendes Paarelement liegt auf PE j */
				comm_remotePE[i] = j;
				wishlist_counts[j]++;
				comm_remoteEl[i] = lcrp->col[i] -lcrp->lfRow[j];
				break;
			}
		}
	}

	acc_wishes = 0;
	for (i=0; i<lcrp->nodes; i++) acc_wishes += wishlist_counts[i];
	size_mem  = (size_t)( acc_wishes * sizeof(int) );

	wishlist        = (int**) allocateMemory( size_nptr, "wishlist" ); 
	cwishlist       = (int**) allocateMemory( size_nptr, "cwishlist" ); 
	wishlist_mem    = (int*)  allocateMemory( size_mem,  "wishlist_mem" ); 
	cwishlist_mem   = (int*)  allocateMemory( size_mem,  "cwishlist_mem" ); 

	hlpi = 0;
	for (i=0; i<lcrp->nodes; i++){
		wishlist[i]  = &wishlist_mem[hlpi];
		cwishlist[i] = &cwishlist_mem[hlpi];
		hlpi += wishlist_counts[i];
	}

	for (i=0;i<lcrp->nodes;i++) item_from[i] = 0;

	for (i=0;i<lcrp->lnEnts[me];i++){
		wishlist[comm_remotePE[i]][item_from[comm_remotePE[i]]] = comm_remoteEl[i];
		item_from[comm_remotePE[i]]++;
	}

	/****************************************************************************
	 *******                      Compress wishlist                       *******
	 ***************************************************************************/

	for (i=0; i<lcrp->nodes; i++){

		for (j=0; j<max_loc_elements; j++) present_values[j] = -1;

		if ( (i!=me) && (wishlist_counts[i]>0) ){
			thisentry = 0;
			for (j=0; j<wishlist_counts[i]; j++){
				if (present_values[wishlist[i][j]]<0){
					/* new entry which has not been found before */     
					present_values[wishlist[i][j]] = thisentry;
					cwishlist[i][thisentry] = wishlist[i][j];
					thisentry = thisentry + 1;
				}
			}
			lcrp->wishes[i] = thisentry;
		}
		else lcrp->wishes[i] = 0; /* no local wishes */

	}

	/****************************************************************************
	 *******       Allgather of wishes & transpose to get dues            *******
	 ***************************************************************************/

	MPI_safecall(MPI_Allgather ( lcrp->wishes, lcrp->nodes, MPI_INTEGER, tmp_transfers, 
				lcrp->nodes, MPI_INTEGER, MPI_COMM_WORLD )) ;

	for (i=0; i<lcrp->nodes; i++) lcrp->dues[i] = tmp_transfers[i*lcrp->nodes+me];

	lcrp->dues[me] = 0; /* keine lokalen Transfers */

	acc_transfer_dues = 0;
	acc_transfer_wishes = 0;
	for (i=0; i<lcrp->nodes; i++){
		acc_transfer_wishes += lcrp->wishes[i];
		acc_transfer_dues   += lcrp->dues[i];
	}

	/****************************************************************************
	 *******   Extract pseudo-indices for access to invec from cwishlist  ******* 
	 ***************************************************************************/

	this_pseudo_col = lcrp->lnRows[me];
	lcrp->halo_elements = 0;
	for (i=0; i<lcrp->nodes; i++){
		if (i != me){ /* natuerlich nur fuer remote-Elemente */
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
			lcrp->col[i] =  comm_remoteEl[i];
		else // remote
			lcrp->col[i] = pseudocol[revcol[lcrp->col[i]]];
	} /* !!!!!!!! Eintraege in wishlist gehen entsprechend Input-file von 1-9! */

	freeMemory ( size_col,  "comm_remoteEl",  comm_remoteEl);
	freeMemory ( size_col,  "comm_remotePE",  comm_remotePE);
	freeMemory ( size_col,  "pseudocol",      pseudocol);
	freeMemory ( size_col,  "globcol",        globcol);
	freeMemory ( size_revc, "revcol",         revcol);
	freeMemory ( size_pval, "present_values", present_values ); 

	/****************************************************************************
	 *******               Finally setup compressed wishlist              *******
	 ***************************************************************************/

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



	for (i=0; i<lcrp->nodes; i++){

		lcrp->due_displ[i]  = acc_dues;
		lcrp->wish_displ[i] = acc_wishes;
		lcrp->duelist[i]    = &(lcrp->duelist_mem[acc_dues]);
		lcrp->wishlist[i]   = &(lcrp->wishlist_mem[acc_wishes]);
		lcrp->hput_pos[i]   = lcrp->lnRows[me]+acc_wishes;

		if  ( (me != i) && !( (i == lcrp->nodes-2) && (me == lcrp->nodes-1) ) ){
			/* auf diese Weise zeigt der Anfang der wishlist fuer die lokalen
			 * Elemente auf die gleiche Position wie die wishlist fuer die
			 * naechste PE.  Sollte aber kein Problem sein, da ich fuer me eh nie
			 * drauf zugreifen sollte. Zweite Bedingung garantiert, dass der
			 * letzte pointer fuer die letzte PE nicht aus dem Feld heraus zeigt
			 * da in vorletzter Iteration bereits nochmal inkrementiert wurde */
			acc_dues   += lcrp->dues[i];
			acc_wishes += lcrp->wishes[i];
		}
	}

	for (i=0; i<lcrp->nodes; i++) for (j=0;j<lcrp->wishes[i];j++)
		lcrp->wishlist[i][j] = cwishlist[i][j]; 

	/* Alle Source-Variablen sind bei Scatterv nur auf root relevant; d.h. ich
	 * nehme automatisch _immer_ die richtige (lokale) wishlist zum Verteilen */

	for(i=0; i<lcrp->nodes; i++) {
		MPI_safecall(MPI_Scatterv ( 
					lcrp->wishlist_mem, lcrp->wishes, lcrp->wish_displ, MPI_INTEGER, 
					lcrp->duelist[i], lcrp->dues[i], MPI_INTEGER, i, MPI_COMM_WORLD ));
	}

	/****************************************************************************
	 *******        Setup the variant using local/non-local arrays        *******
	 ***************************************************************************/
	if (!(options & SPMVM_OPTION_NO_SPLIT_KERNELS)) { // split computation


		pseudo_ldim = lcrp->lnRows[me]+lcrp->halo_elements ;

		lnEnts_l=0;
		for (i=0; i<lcrp->lnEnts[me];i++)
			if (lcrp->col[i]<lcrp->lnRows[me]) lnEnts_l++;


		lnEnts_r = lcrp->lnEnts[me]-lnEnts_l;

		DEBUG_LOG(1,"PE%d: Rows=%6d\t Ents=%6d(l),%6d(r),%6d(g)\t pdim=%6d", 
				me, lcrp->lnRows[me], lnEnts_l, lnEnts_r, lcrp->lnEnts[me], pseudo_ldim );

		size_lval = (size_t)( lnEnts_l             * sizeof(mat_data_t) ); 
		size_rval = (size_t)( lnEnts_r             * sizeof(mat_data_t) ); 
		size_lcol = (size_t)( lnEnts_l             * sizeof(int) ); 
		size_rcol = (size_t)( lnEnts_r             * sizeof(int) ); 
		size_lptr = (size_t)( (lcrp->lnRows[me]+1) * sizeof(int) ); 
		size_rptr = (size_t)( (lcrp->lnRows[me]+1) * sizeof(int) ); 


		lcrp->lrow_ptr_l = (int*)    allocateMemory( size_lptr, "lcrp->lrow_ptr_l" ); 
		lcrp->lrow_ptr_r = (int*)    allocateMemory( size_rptr, "lcrp->lrow_ptr_r" ); 
		lcrp->lcol       = (int*)    allocateMemory( size_lcol, "lcrp->lcol" ); 
		lcrp->rcol       = (int*)    allocateMemory( size_rcol, "lcrp->rcol" ); 
		lcrp->lval       = (mat_data_t*) allocateMemory( size_lval, "lcrp->lval" ); 
		lcrp->rval       = (mat_data_t*) allocateMemory( size_rval, "lcrp->rval" ); 

#pragma omp parallel for schedule(runtime)
		for (i=0; i<lnEnts_l; i++) lcrp->lval[i] = 0.0;

#pragma omp parallel for schedule(runtime)
		for (i=0; i<lnEnts_l; i++) lcrp->lcol[i] = 0.0;

#pragma omp parallel for schedule(runtime)
		for (i=0; i<lnEnts_r; i++) lcrp->rval[i] = 0.0;

#pragma omp parallel for schedule(runtime)
		for (i=0; i<lnEnts_r; i++) lcrp->rcol[i] = 0.0;


		lcrp->lrow_ptr_l[0] = 0;
		lcrp->lrow_ptr_r[0] = 0;

		MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));
		DEBUG_LOG(1,"PE%d: lnRows=%d row_ptr=%d..%d", 
				me, lcrp->lnRows[me], lcrp->lrow_ptr[0], lcrp->lrow_ptr[lcrp->lnRows[me]]);
		fflush(stdout);
		MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));

		for (i=0; i<lcrp->lnRows[me]; i++){

			current_l = 0;
			current_r = 0;

			for (j=lcrp->lrow_ptr[i]; j<lcrp->lrow_ptr[i+1]; j++){

				if (lcrp->col[j]<lcrp->lnRows[me]){
					/* local element */
					lcrp->lcol[ lcrp->lrow_ptr_l[i]+current_l ] = lcrp->col[j]; 
					lcrp->lval[ lcrp->lrow_ptr_l[i]+current_l ] = lcrp->val[j]; 
					current_l++;
				}
				else{
					/* remote element */
					lcrp->rcol[ lcrp->lrow_ptr_r[i]+current_r ] = lcrp->col[j];
					lcrp->rval[ lcrp->lrow_ptr_r[i]+current_r ] = lcrp->val[j];
					current_r++;
				}

			}  

			lcrp->lrow_ptr_l[i+1] = lcrp->lrow_ptr_l[i] + current_l;
			lcrp->lrow_ptr_r[i+1] = lcrp->lrow_ptr_r[i] + current_r;
		}

		IF_DEBUG(2){
			for (i=0; i<lcrp->lnRows[me]+1; i++)
				DEBUG_LOG(2,"--Row_ptrs-- PE %d: i=%d local=%d remote=%d", 
						me, i, lcrp->lrow_ptr_l[i], lcrp->lrow_ptr_r[i]);
			for (i=0; i<lcrp->lrow_ptr_l[lcrp->lnRows[me]]; i++)
				DEBUG_LOG(2,"-- local -- PE%d: lcrp->lcol[%d]=%d", me, i, lcrp->lcol[i]);
			for (i=0; i<lcrp->lrow_ptr_r[lcrp->lnRows[me]]; i++)
				DEBUG_LOG(2,"-- remote -- PE%d: lcrp->rcol[%d]=%d", me, i, lcrp->rcol[i]);
		}
		fflush(stdout);
		MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));

	} else{
		lcrp->lrow_ptr_l = (int*)    allocateMemory( sizeof(int), "lcrp->lrow_ptr_l" ); 
		lcrp->lrow_ptr_r = (int*)    allocateMemory( sizeof(int), "lcrp->lrow_ptr_r" ); 
		lcrp->lcol       = (int*)    allocateMemory( sizeof(int), "lcrp->lcol" ); 
		lcrp->rcol       = (int*)    allocateMemory( sizeof(int), "lcrp->rcol" ); 
		lcrp->lval       = (mat_data_t*) allocateMemory( sizeof(mat_data_t), "lcrp->lval" ); 
		lcrp->rval       = (mat_data_t*) allocateMemory( sizeof(mat_data_t), "lcrp->rval" ); 
	}

	freeMemory ( size_mem,  "wishlist_mem",    wishlist_mem);
	freeMemory ( size_mem,  "cwishlist_mem",   cwishlist_mem);
	freeMemory ( size_nptr, "wishlist",        wishlist);
	freeMemory ( size_nptr, "cwishlist",       cwishlist);
	freeMemory ( size_a2ai, "tmp_transfers",   tmp_transfers);
	freeMemory ( size_nint, "wishlist_counts", wishlist_counts);
	freeMemory ( size_nint, "item_from",       item_from);


	/* Free memory for CR stored matrix and sweep memory */
	//freeCRMatrix( cr );

	return lcrp;
}

LCRP_TYPE* setup_communication_parallel(CR_TYPE* cr, char *matrixPath, int options)
{

	/* Counting and auxilliary variables */
	int i, j, hlpi;

	/* Processor rank (MPI-process) */
	int me; 

	/* MPI-Errorcode */
	int ierr;

	int max_loc_elements, thisentry;
	int *present_values;

	LCRP_TYPE *lcrp;

	int acc_dues;

	int *tmp_transfers;

	int target_nnz;
	int acc_wishes;

	int nEnts_glob;

	/* Counter how many entries are requested from each PE */
	int *item_from;

	int *wishlist_counts;

	int *wishlist_mem,  **wishlist;
	int *cwishlist_mem, **cwishlist;

	int this_pseudo_col;
	int *pseudocol;
	int *globcol;
	int *revcol;

	int *comm_remotePE, *comm_remoteEl;

	int target_rows;

	int lnEnts_l, lnEnts_r;
	int current_l, current_r;

	int pseudo_ldim;
	int acc_transfer_wishes, acc_transfer_dues;

	size_t size_nint, size_col, size_val, size_ptr;
	size_t size_lval, size_rval, size_lcol, size_rcol, size_lptr, size_rptr;
	size_t size_revc, size_a2ai, size_nptr, size_pval;  
	size_t size_mem, size_wish, size_dues;


	MPI_Info info = MPI_INFO_NULL;
	MPI_File file_handle;
	MPI_Offset offset_in_file;
	MPI_Status status;

	/****************************************************************************
	 *******            ........ Executable statements ........           *******
	 ***************************************************************************/

	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &me);

	DEBUG_LOG(1,"Entering setup_communication_parallel");

	lcrp = (LCRP_TYPE*) allocateMemory( sizeof(LCRP_TYPE), "lcrp");

	/*lcrp->fullRowPerm = NULL;
	lcrp->fullInvRowPerm = NULL;
	lcrp->splitRowPerm = NULL;
	lcrp->splitInvRowPerm = NULL;
	lcrp->fullMatrix = NULL;
	lcrp->localMatrix = NULL;
	lcrp->remoteMatrix = NULL;*/
	lcrp->lrow_ptr_l = NULL; 
	lcrp->lrow_ptr_r = NULL; 
	lcrp->lcol       = NULL;
	lcrp->rcol       = NULL;
	lcrp->lval       = NULL;
	lcrp->rval       = NULL;

/*	if (me==0) {
		lcrp->nEnts = cr->nEnts;
		lcrp->nRows = cr->nRows;
	}

	MPI_safecall(MPI_Bcast(&lcrp->nEnts,1,MPI_INT,0,MPI_COMM_WORLD));
	MPI_safecall(MPI_Bcast(&lcrp->nRows,1,MPI_INT,0,MPI_COMM_WORLD));
*/
#pragma omp parallel
	lcrp->threads = omp_get_num_threads(); 

	ierr = MPI_Comm_size(MPI_COMM_WORLD, &(lcrp->nodes));

	size_nint = (size_t)( (size_t)(lcrp->nodes)   * sizeof(int)  );
	size_nptr = (size_t)( lcrp->nodes             * sizeof(int*) );
	size_a2ai = (size_t)( lcrp->nodes*lcrp->nodes * sizeof(int)  );

	lcrp->lnEnts   = (int*)       allocateMemory( size_nint, "lcrp->lnEnts" ); 
	lcrp->lnRows   = (int*)       allocateMemory( size_nint, "lcrp->lnRows" ); 
	lcrp->lfEnt    = (int*)       allocateMemory( size_nint, "lcrp->lfEnt" ); 
	lcrp->lfRow    = (int*)       allocateMemory( size_nint, "lcrp->lfRow" ); 

	lcrp->wishes   = (int*)       allocateMemory( size_nint, "lcrp->wishes" ); 
	lcrp->dues     = (int*)       allocateMemory( size_nint, "lcrp->dues" ); 

	/****************************************************************************
	 *******  Calculate a fair partitioning of NZE and ROWS on master PE  *******
	 ***************************************************************************/
	if (me==0){
		if (options & SPMVM_OPTION_WORKDIST_LNZE){
			DEBUG_LOG(0,"Warning! SPMVM_OPTION_WORKDIST_LNZE has not (yet) been "
					"implemented for parallel IO! Switching to "
					"SPMVM_OPTION_WORKDIST_NZE");
			options |= SPMVM_OPTION_WORKDIST_NZE;
		}

		if (options & SPMVM_OPTION_WORKDIST_NZE){
			DEBUG_LOG(1, "Distribute Matrix with EQUAL_NZE on each PE");
			target_nnz = (cr->nEnts/lcrp->nodes)+1; /* sonst bleiben welche uebrig! */

			lcrp->lfRow[0]  = 0;
			lcrp->lfEnt[0] = 0;
			j = 1;

			for (i=0;i<cr->nRows;i++){
				if (cr->rowOffset[i] >= j*target_nnz){
					lcrp->lfRow[j] = i;
					lcrp->lfEnt[j] = cr->rowOffset[i];
					j = j+1;
				}
			}

		}
		else {

			DEBUG_LOG(1,"Distribute Matrix with EQUAL_ROWS on each PE");
			target_rows = (cr->nRows/lcrp->nodes);

			lcrp->lfRow[0] = 0;
			lcrp->lfEnt[0] = 0;

			for (i=1; i<lcrp->nodes; i++){
				lcrp->lfRow[i] = lcrp->lfRow[i-1]+target_rows;
				lcrp->lfEnt[i] = cr->rowOffset[lcrp->lfRow[i]];
			}
		}


		for (i=0; i<lcrp->nodes-1; i++){
			lcrp->lnRows[i] = lcrp->lfRow[i+1] - lcrp->lfRow[i] ;
			lcrp->lnEnts[i] = lcrp->lfEnt[i+1] - lcrp->lfEnt[i] ;
		}

		lcrp->lnRows[lcrp->nodes-1] = cr->nRows - lcrp->lfRow[lcrp->nodes-1] ;
		lcrp->lnEnts[lcrp->nodes-1] = cr->nEnts - lcrp->lfEnt[lcrp->nodes-1];
	}

	/****************************************************************************
	 *******            Distribute correct share to all PEs               *******
	 ***************************************************************************/

	//ierr = MPI_Bcast(&(lcrp->nEnts),  1, MPI_INTEGER, 0, MPI_COMM_WORLD);
	//ierr = MPI_Bcast(&(lcrp->nRows),  1, MPI_INTEGER, 0, MPI_COMM_WORLD);
	ierr = MPI_Bcast(lcrp->lfRow,  lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD);
	ierr = MPI_Bcast(lcrp->lfEnt,  lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD);
	ierr = MPI_Bcast(lcrp->lnRows, lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD);
	ierr = MPI_Bcast(lcrp->lnEnts, lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD);

	DEBUG_LOG(1,"local rows          = %i",lcrp->lnRows[me]);
	DEBUG_LOG(1,"local rows (offset) = %i",lcrp->lfRow[me]);
	DEBUG_LOG(1,"local entries          = %i",lcrp->lnEnts[me]);
	DEBUG_LOG(1,"local entires (offset) = %i",lcrp->lfEnt[me]);

	/****************************************************************************
	 *******   Allocate memory for matrix in distributed CRS storage      *******
	 ***************************************************************************/

	size_val  = (size_t)( (size_t)(lcrp->lnEnts[me])   * sizeof( mat_data_t ) );
	size_col  = (size_t)( (size_t)(lcrp->lnEnts[me])   * sizeof( int ) );
	size_ptr  = (size_t)( (size_t)(lcrp->lnRows[me]+1) * sizeof( int ) );

	lcrp->val      = (mat_data_t*)    allocateMemory( size_val,  "lcrp->val" ); 
	lcrp->col      = (int*)       allocateMemory( size_col,  "lcrp->col" ); 
	lcrp->lrow_ptr = (int*)       allocateMemory( size_ptr,  "lcrp->lrow_ptr" ); 

	/****************************************************************************
	 *******   Fill all fields with their corresponding values            *******
	 ***************************************************************************/


	#pragma omp parallel for schedule(runtime)
//#pragma omp parallel for schedule(static)
	for (i=0; i<lcrp->lnEnts[me]; i++) lcrp->val[i] = 0.0;

	#pragma omp parallel for schedule(runtime)
//#pragma omp parallel for schedule(static)
	for (i=0; i<lcrp->lnEnts[me]; i++) lcrp->col[i] = 0.0;

	#pragma omp parallel for schedule(runtime)
//#pragma omp parallel for schedule(static)
	for (i=0; i<lcrp->lnRows[me]; i++) lcrp->lrow_ptr[i] = 0.0;

	/* replace scattering with read-in */
	DEBUG_LOG(1,"Opening file %s for parallel read-in",matrixPath);
	ierr = MPI_File_open(MPI_COMM_WORLD, matrixPath, MPI_MODE_RDONLY, info, &file_handle);

	if( ierr ) {
		ABORT("Unable to open parallel file %s",matrixPath);
	}

	int datatype;
	MPI_safecall(MPI_File_seek(file_handle,0,MPI_SEEK_SET));
	MPI_safecall(MPI_File_read(file_handle,&datatype,1,MPI_INTEGER,&status));

	/* read col */
	offset_in_file = (4+cr->nRows+1)*sizeof(int) + (lcrp->lfEnt[me])*sizeof(int);
	DEBUG_LOG(1,"Read col -- offset=%lu | %d",(size_t)offset_in_file,lcrp->lfEnt[me]);
	MPI_safecall(MPI_File_seek(file_handle, offset_in_file, MPI_SEEK_SET));
	MPI_safecall(MPI_File_read(file_handle, lcrp->col, lcrp->lnEnts[me], MPI_INTEGER, &status));

	/* read val */
	if (datatype != DATATYPE_DESIRED) {
		if (me==0) {
			DEBUG_LOG(0,"Warning The library has been built for %s data but"
					" the file contains %s data. Casting...",
					DATATYPE_NAMES[DATATYPE_DESIRED],DATATYPE_NAMES[datatype]);
		}
		switch(datatype) {
			case DATATYPE_FLOAT:
				{
					float *tmp = (float *)allocateMemory(lcrp->lnEnts[me]*sizeof(float), "tmp");
					offset_in_file = (4+cr->nRows+1)*sizeof(int) + (cr->nEnts)*sizeof(int) + (lcrp->lfEnt[me])*sizeof(float);
					DEBUG_LOG(1,"Read val -- offset=%lu",(size_t)offset_in_file);
					MPI_safecall(MPI_File_seek(file_handle, offset_in_file, MPI_SEEK_SET));
					MPI_safecall(MPI_File_read(file_handle, tmp, lcrp->lnEnts[me], MPI_FLOAT, &status));
					for (i = 0; i<lcrp->lnEnts[me]; i++) lcrp->val[i] = (mat_data_t) tmp[i];
					free(tmp);
					break;
				}
			case DATATYPE_DOUBLE:
				{
					double *tmp = (double *)allocateMemory(lcrp->lnEnts[me]*sizeof(double), "tmp");
					offset_in_file = (4+cr->nRows+1)*sizeof(int) + (cr->nEnts)*sizeof(int) + (lcrp->lfEnt[me])*sizeof(double);
					DEBUG_LOG(1,"Read val -- offset=%lu",(size_t)offset_in_file);

					MPI_safecall(MPI_File_seek(file_handle, offset_in_file, MPI_SEEK_SET));
					MPI_safecall(MPI_File_read(file_handle, tmp, lcrp->lnEnts[me], MPI_DOUBLE, &status));
					for (i = 0; i<lcrp->lnEnts[me]; i++)
						lcrp->val[i] = (mat_data_t) tmp[i];
					free(tmp);
					break;
				}
			case DATATYPE_COMPLEX_DOUBLE:
				{
					_Complex double *tmp = (_Complex double *)allocateMemory(lcrp->lnEnts[me]*sizeof(_Complex double), "tmp");
					offset_in_file = (4+cr->nRows+1)*sizeof(int) + (cr->nEnts)*sizeof(int) + (lcrp->lfEnt[me])*sizeof(_Complex double);
					DEBUG_LOG(1,"Read val -- offset=%lu",(size_t)offset_in_file);

					MPI_Datatype tmpDT;
					MPI_safecall(MPI_Type_contiguous(2,MPI_DOUBLE,&tmpDT));
					MPI_safecall(MPI_Type_commit(&tmpDT));

					MPI_safecall(MPI_File_seek(file_handle, offset_in_file, MPI_SEEK_SET));
					MPI_safecall(MPI_File_read(file_handle, tmp, lcrp->lnEnts[me], tmpDT, &status));

					for (i = 0; i<lcrp->lnEnts[me]; i++) lcrp->val[i] = (mat_data_t) tmp[i];

					free(tmp);
					MPI_safecall(MPI_Type_free(&tmpDT));
					break;
				}
			case DATATYPE_COMPLEX_FLOAT:
				{
					_Complex float *tmp = (_Complex float *)allocateMemory(lcrp->lnEnts[me]*sizeof(_Complex float), "tmp");
					offset_in_file = (4+cr->nRows+1)*sizeof(int) + (cr->nEnts)*sizeof(int) + (lcrp->lfEnt[me])*sizeof(_Complex float);
					DEBUG_LOG(1,"Read val -- offset=%lu",(size_t)offset_in_file);

					MPI_Datatype tmpDT;
					MPI_safecall(MPI_Type_contiguous(2,MPI_FLOAT,&tmpDT));
					MPI_safecall(MPI_Type_commit(&tmpDT));

					MPI_safecall(MPI_File_seek(file_handle, offset_in_file, MPI_SEEK_SET));
					MPI_safecall(MPI_File_read(file_handle, tmp, lcrp->lnEnts[me], tmpDT, &status));

					for (i = 0; i<lcrp->lnEnts[me]; i++) lcrp->val[i] = (mat_data_t) tmp[i];

					free(tmp);
					MPI_safecall(MPI_Type_free(&tmpDT));
					break;
				}

		}
	} else {

		offset_in_file = (4+cr->nRows+1)*sizeof(int) + (cr->nEnts)*sizeof(int) + (lcrp->lfEnt[me])*sizeof(mat_data_t);
		DEBUG_LOG(1,"Read val -- offset=%lu",(size_t)offset_in_file);
		MPI_safecall(MPI_File_seek(file_handle, offset_in_file, MPI_SEEK_SET));
		MPI_safecall(MPI_File_read(file_handle, lcrp->val, lcrp->lnEnts[me], MPI_MYDATATYPE, &status));
	}

	ierr = MPI_File_close(&file_handle);

	/* Offsets are scattered */
	ierr = MPI_Scatterv ( cr->rowOffset, lcrp->lnRows, lcrp->lfRow, MPI_INTEGER,
			lcrp->lrow_ptr, lcrp->lnRows[me],  MPI_INTEGER, 0, MPI_COMM_WORLD);

	/****************************************************************************
	 *******        Adapt row pointer to local numbering on this PE       *******         
	 ***************************************************************************/

	for (i=0;i<lcrp->lnRows[me]+1;i++)
		lcrp->lrow_ptr[i] =  lcrp->lrow_ptr[i] - lcrp->lfEnt[me]; 

	/* last entry of row_ptr holds the local number of entries */
	lcrp->lrow_ptr[lcrp->lnRows[me]] = lcrp->lnEnts[me]; 

	/****************************************************************************
	 *******         Extract maximum number of local elements             *******
	 ***************************************************************************/

	max_loc_elements = 0;
	for (i=0;i<lcrp->nodes;i++)
		if (max_loc_elements<lcrp->lnRows[i]) max_loc_elements = lcrp->lnRows[i];

	nEnts_glob = lcrp->lfEnt[lcrp->nodes-1]+lcrp->lnEnts[lcrp->nodes-1]; 


	/****************************************************************************
	 *******         Assemble wish- and duelists for communication        *******
	 ***************************************************************************/

	size_pval = (size_t)( max_loc_elements * sizeof(int) );
	size_revc = (size_t)( nEnts_glob       * sizeof(int) );

	item_from       = (int*) allocateMemory( size_nint, "item_from" ); 
	wishlist_counts = (int*) allocateMemory( size_nint, "wishlist_counts" ); 
	comm_remotePE   = (int*) allocateMemory( size_col,  "comm_remotePE" );
	comm_remoteEl   = (int*) allocateMemory( size_col,  "comm_remoteEl" );
	present_values  = (int*) allocateMemory( size_pval, "present_values" ); 
	tmp_transfers   = (int*) allocateMemory( size_a2ai, "tmp_transfers" ); 
	pseudocol       = (int*) allocateMemory( size_col,  "pseudocol" );
	globcol         = (int*) allocateMemory( size_col,  "origincol" );
	revcol          = (int*) allocateMemory( size_revc, "revcol" );


	for (i=0; i<lcrp->nodes; i++) wishlist_counts[i] = 0;

	/* Transform global column index into 2d-local/non-local index */
	for (i=0;i<lcrp->lnEnts[me];i++){
		for (j=lcrp->nodes-1;j>-1; j--){
			if (lcrp->lfRow[j]<lcrp->col[i]+1) {
				/* Entsprechendes Paarelement liegt auf PE j */
				comm_remotePE[i] = j;
				wishlist_counts[j]++;
				comm_remoteEl[i] = lcrp->col[i] -lcrp->lfRow[j];
				break;
			}
		}
	}

	acc_wishes = 0;
	for (i=0; i<lcrp->nodes; i++) acc_wishes += wishlist_counts[i];
	size_mem  = (size_t)( acc_wishes * sizeof(int) );

	wishlist        = (int**) allocateMemory( size_nptr, "wishlist" ); 
	cwishlist       = (int**) allocateMemory( size_nptr, "cwishlist" ); 
	wishlist_mem    = (int*)  allocateMemory( size_mem,  "wishlist_mem" ); 
	cwishlist_mem   = (int*)  allocateMemory( size_mem,  "cwishlist_mem" ); 

	hlpi = 0;
	for (i=0; i<lcrp->nodes; i++){
		wishlist[i]  = &wishlist_mem[hlpi];
		cwishlist[i] = &cwishlist_mem[hlpi];
		hlpi += wishlist_counts[i];
	}

	for (i=0;i<lcrp->nodes;i++) item_from[i] = 0;

	for (i=0;i<lcrp->lnEnts[me];i++){
		wishlist[comm_remotePE[i]][item_from[comm_remotePE[i]]] = comm_remoteEl[i];
		item_from[comm_remotePE[i]]++;
	}

	/****************************************************************************
	 *******                      Compress wishlist                       *******
	 ***************************************************************************/

	for (i=0; i<lcrp->nodes; i++){

		for (j=0; j<max_loc_elements; j++) present_values[j] = -1;

		if ( (i!=me) && (wishlist_counts[i]>0) ){
			thisentry = 0;
			for (j=0; j<wishlist_counts[i]; j++){
				if (present_values[wishlist[i][j]]<0){
					/* new entry which has not been found before */     
					present_values[wishlist[i][j]] = thisentry;
					cwishlist[i][thisentry] = wishlist[i][j];
					thisentry = thisentry + 1;
				}
			}
			lcrp->wishes[i] = thisentry;
		}
		else lcrp->wishes[i] = 0; /* no local wishes */

	}

	/****************************************************************************
	 *******       Allgather of wishes & transpose to get dues            *******
	 ***************************************************************************/

	ierr = MPI_Allgather ( lcrp->wishes, lcrp->nodes, MPI_INTEGER, tmp_transfers, 
			lcrp->nodes, MPI_INTEGER, MPI_COMM_WORLD ) ;

	for (i=0; i<lcrp->nodes; i++) lcrp->dues[i] = tmp_transfers[i*lcrp->nodes+me];

	lcrp->dues[me] = 0; /* keine lokalen Transfers */

	acc_transfer_dues = 0;
	acc_transfer_wishes = 0;
	for (i=0; i<lcrp->nodes; i++){
		acc_transfer_wishes += lcrp->wishes[i];
		acc_transfer_dues   += lcrp->dues[i];
	}

	/****************************************************************************
	 *******   Extract pseudo-indices for access to invec from cwishlist  ******* 
	 ***************************************************************************/

	this_pseudo_col = lcrp->lnRows[me];
	lcrp->halo_elements = 0;
	for (i=0; i<lcrp->nodes; i++){
		if (i != me){ /* natuerlich nur fuer remote-Elemente */
			for (j=0;j<lcrp->wishes[i];j++){
				pseudocol[lcrp->halo_elements] = this_pseudo_col;  
				globcol[lcrp->halo_elements]   = lcrp->lfRow[i]+cwishlist[i][j]; 
				revcol[globcol[lcrp->halo_elements]] = lcrp->halo_elements;
				lcrp->halo_elements++;
				this_pseudo_col++;
			}
		}
	}

	for (i=0;i<lcrp->lnEnts[me];i++){
		if (comm_remotePE[i] == me) lcrp->col[i] = comm_remoteEl[i];
		else                        lcrp->col[i] = pseudocol[revcol[lcrp->col[i]]];
	} /* !!!!!!!! Eintraege in wishlist gehen entsprechend Input-file von 1-9! */

	freeMemory ( size_col,  "comm_remoteEl",  comm_remoteEl);
	freeMemory ( size_col,  "comm_remotePE",  comm_remotePE);
	freeMemory ( size_col,  "pseudocol",      pseudocol);
	freeMemory ( size_col,  "globcol",        globcol);
	freeMemory ( size_revc, "revcol",         revcol);
	freeMemory ( size_pval, "present_values", present_values ); 

	/****************************************************************************
	 *******               Finally setup compressed wishlist              *******
	 ***************************************************************************/

	size_wish = (size_t)( acc_transfer_wishes * sizeof(int) );
	size_dues = (size_t)( acc_transfer_dues   * sizeof(int) );

	lcrp->wishlist      = (int**) allocateMemory( size_nptr, "lcrp->wishlist" ); 
	lcrp->duelist       = (int**) allocateMemory( size_nptr, "lcrp->duelist" ); 
	lcrp->wishlist_mem  = (int*)  allocateMemory( size_wish, "lcrp->wishlist_mem" ); 
	lcrp->duelist_mem   = (int*)  allocateMemory( size_dues, "lcrp->duelist_mem" ); 
	lcrp->wish_displ    = (int*)  allocateMemory( size_nint, "lcrp->wish_displ" ); 
	lcrp->due_displ     = (int*)  allocateMemory( size_nint, "lcrp->due_displ" ); 
	lcrp->hput_pos      = (int*)  allocateMemory( size_nptr, "lcrp->hput_pos" ); 

	acc_dues = 0;
	acc_wishes = 0;

	for (i=0; i<lcrp->nodes; i++){

		lcrp->due_displ[i]  = acc_dues;
		lcrp->wish_displ[i] = acc_wishes;
		lcrp->duelist[i]    = &(lcrp->duelist_mem[acc_dues]);
		lcrp->wishlist[i]   = &(lcrp->wishlist_mem[acc_wishes]);
		lcrp->hput_pos[i]   = lcrp->lnRows[me]+acc_wishes;

		if  ( (me != i) && !( (i == lcrp->nodes-2) && (me == lcrp->nodes-1) ) ){
			/* auf diese Weise zeigt der Anfang der wishlist fuer die lokalen
			 * Elemente auf die gleiche Position wie die wishlist fuer die
			 * naechste PE.  Sollte aber kein Problem sein, da ich fuer me eh nie
			 * drauf zugreifen sollte. Zweite Bedingung garantiert, dass der
			 * letzte pointer fuer die letzte PE nicht aus dem Feld heraus zeigt
			 * da in vorletzter Iteration bereits nochmal inkrementiert wurde */
			acc_dues   += lcrp->dues[i];
			acc_wishes += lcrp->wishes[i];
		}
	}

	for (i=0; i<lcrp->nodes; i++) for (j=0;j<lcrp->wishes[i];j++)
		lcrp->wishlist[i][j] = cwishlist[i][j]; 

	/* Alle Source-Variablen sind bei Scatterv nur auf root relevant; d.h. ich
	 * nehme automatisch _immer_ die richtige (lokale) wishlist zum Verteilen */

	for(i=0; i<lcrp->nodes; i++) ierr = MPI_Scatterv ( 
			lcrp->wishlist_mem, lcrp->wishes, lcrp->wish_displ, MPI_INTEGER, 
			lcrp->duelist[i], lcrp->dues[i], MPI_INTEGER, i, MPI_COMM_WORLD );

	/****************************************************************************
	 *******        Setup the variant using local/non-local arrays        *******
	 ***************************************************************************/
	if (!(options & SPMVM_OPTION_NO_SPLIT_KERNELS)) { // split computation

		pseudo_ldim = lcrp->lnRows[me]+lcrp->halo_elements ;

		lnEnts_l=0;
		for (i=0; i<lcrp->lnEnts[me];i++)
			if (lcrp->col[i]<lcrp->lnRows[me]) lnEnts_l++;


		lnEnts_r = lcrp->lnEnts[me]-lnEnts_l;

		DEBUG_LOG(1,"Rows=%6d\t Ents=%6d(l),%6d(r),%6d(g)\t pdim=%6d", 
				 lcrp->lnRows[me], lnEnts_l, lnEnts_r, lcrp->lnEnts[me], pseudo_ldim );

		size_lval = (size_t)( lnEnts_l             * sizeof(mat_data_t) ); 
		size_rval = (size_t)( lnEnts_r             * sizeof(mat_data_t) ); 
		size_lcol = (size_t)( lnEnts_l             * sizeof(int) ); 
		size_rcol = (size_t)( lnEnts_r             * sizeof(int) ); 
		size_lptr = (size_t)( (lcrp->lnRows[me]+1) * sizeof(int) ); 
		size_rptr = (size_t)( (lcrp->lnRows[me]+1) * sizeof(int) ); 


		lcrp->lrow_ptr_l = (int*)    allocateMemory( size_lptr, "lcrp->lrow_ptr_l" ); 
		lcrp->lrow_ptr_r = (int*)    allocateMemory( size_rptr, "lcrp->lrow_ptr_r" ); 
		lcrp->lcol       = (int*)    allocateMemory( size_lcol, "lcrp->lcol" ); 
		lcrp->rcol       = (int*)    allocateMemory( size_rcol, "lcrp->rcol" ); 
		lcrp->lval       = (mat_data_t*) allocateMemory( size_lval, "lcrp->lval" ); 
		lcrp->rval       = (mat_data_t*) allocateMemory( size_rval, "lcrp->rval" ); 

#pragma omp parallel for schedule(runtime)
		for (i=0; i<lnEnts_l; i++) lcrp->lval[i] = 0.0;

#pragma omp parallel for schedule(runtime)
		for (i=0; i<lnEnts_l; i++) lcrp->lcol[i] = 0.0;

#pragma omp parallel for schedule(runtime)
		for (i=0; i<lnEnts_r; i++) lcrp->rval[i] = 0.0;

#pragma omp parallel for schedule(runtime)
		for (i=0; i<lnEnts_r; i++) lcrp->rcol[i] = 0.0;


		lcrp->lrow_ptr_l[0] = 0;
		lcrp->lrow_ptr_r[0] = 0;

		ierr = MPI_Barrier(MPI_COMM_WORLD);
		DEBUG_LOG(1,"lnRows=%d row_ptr=%d..%d", 
				 lcrp->lnRows[me], lcrp->lrow_ptr[0], lcrp->lrow_ptr[lcrp->lnRows[me]]);

		for (i=0; i<lcrp->lnRows[me]; i++){

			current_l = 0;
			current_r = 0;

			for (j=lcrp->lrow_ptr[i]; j<lcrp->lrow_ptr[i+1]; j++){

				if (lcrp->col[j]<lcrp->lnRows[me]){
					/* local element */
					lcrp->lcol[ lcrp->lrow_ptr_l[i]+current_l ] = lcrp->col[j]; 
					lcrp->lval[ lcrp->lrow_ptr_l[i]+current_l ] = lcrp->val[j]; 
					current_l++;
				}
				else{
					/* remote element */
					lcrp->rcol[ lcrp->lrow_ptr_r[i]+current_r ] = lcrp->col[j];
					lcrp->rval[ lcrp->lrow_ptr_r[i]+current_r ] = lcrp->val[j];
					current_r++;
				}

			}  

			lcrp->lrow_ptr_l[i+1] = lcrp->lrow_ptr_l[i] + current_l;
			lcrp->lrow_ptr_r[i+1] = lcrp->lrow_ptr_r[i] + current_r;
		}

		IF_DEBUG(2){
			for (i=0; i<lcrp->lnRows[me]+1; i++)
				DEBUG_LOG(2,"--Row_ptrs-- PE %d: i=%d local=%d remote=%d", 
						me, i, lcrp->lrow_ptr_l[i], lcrp->lrow_ptr_r[i]);
			for (i=0; i<lcrp->lrow_ptr_l[lcrp->lnRows[me]]; i++)
				DEBUG_LOG(2,"-- local -- PE%d: lcrp->lcol[%d]=%d", me, i, lcrp->lcol[i]);
			for (i=0; i<lcrp->lrow_ptr_r[lcrp->lnRows[me]]; i++)
				DEBUG_LOG(2,"-- remote -- PE%d: lcrp->rcol[%d]=%d", me, i, lcrp->rcol[i]);
		}

	}
	/*else{
	  lcrp->lrow_ptr_l = (int*)    allocateMemory( sizeof(int), "lcrp->lrow_ptr_l" ); 
	  lcrp->lrow_ptr_r = (int*)    allocateMemory( sizeof(int), "lcrp->lrow_ptr_r" ); 
	  lcrp->lcol       = (int*)    allocateMemory( sizeof(int), "lcrp->lcol" ); 
	  lcrp->rcol       = (int*)    allocateMemory( sizeof(int), "lcrp->rcol" ); 
	  lcrp->lval       = (mat_data_t*) allocateMemory( sizeof(mat_data_t), "lcrp->lval" ); 
	  lcrp->rval       = (mat_data_t*) allocateMemory( sizeof(mat_data_t), "lcrp->rval" ); 
	  }*/

	freeMemory ( size_mem,  "wishlist_mem",    wishlist_mem);
	freeMemory ( size_mem,  "cwishlist_mem",   cwishlist_mem);
	freeMemory ( size_nptr, "wishlist",        wishlist);
	freeMemory ( size_nptr, "cwishlist",       cwishlist);
	freeMemory ( size_a2ai, "tmp_transfers",   tmp_transfers);
	freeMemory ( size_nint, "wishlist_counts", wishlist_counts);
	freeMemory ( size_nint, "item_from",       item_from);


	return lcrp;
}




