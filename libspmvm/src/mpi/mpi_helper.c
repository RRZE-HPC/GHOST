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

void SpMVM_createDistributedSetupSerial(SETUP_TYPE * setup, CR_TYPE* cr, int options)
{

	/* Counting and auxilliary variables */
	mat_idx_t i, j;
	mat_nnz_t e;
	int hlpi;

	/* Processor rank (MPI-process) */
	int me; 

	/* MPI-Errorcode */

	mat_nnz_t max_loc_elements, thisentry;
	int *present_values;

	LCRP_TYPE *lcrp;

	int acc_dues;

	int *tmp_transfers;

	int target_nnz;
	int acc_wishes;

	int nEnts_glob;

	/* Counter how many entries are requested from each PE */
	int *item_from;

	unsigned int *wishlist_counts;

	int *wishlist_mem,  **wishlist;
	int *cwishlist_mem, **cwishlist;

	int this_pseudo_col;
	int *pseudocol;
	int *globcol;
	int *revcol;

	int *comm_remotePE, *comm_remoteEl;

	int target_rows;

	mat_nnz_t lnEnts_l, lnEnts_r;
	int current_l, current_r;

	int target_lnze;

	int trial_count, prev_count, trial_rows;
	int *loc_count;
	int ideal, prev_rows;
	int outer_iter, outer_convergence;

	int pseudo_ldim;
	int acc_transfer_wishes, acc_transfer_dues;

	size_t size_nint, size_col;
	size_t size_revc, size_a2ai, size_nptr, size_pval;  
	size_t size_mem, size_dues;


	/****************************************************************************
	 *******            ........ Executable statements ........           *******
	 ***************************************************************************/

	MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD, &me));

	DEBUG_LOG(1,"Entering setup_communication");
	MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));

	lcrp = (LCRP_TYPE*) allocateMemory( sizeof(LCRP_TYPE), "lcrp");

	setup->communicator = lcrp;

#pragma omp parallel
	lcrp->threads = omp_get_num_threads(); 

	lcrp->nodes = SpMVM_getNumberOfProcesses();

	size_nint = (size_t)( (size_t)(lcrp->nodes)   * sizeof(int)  );
	size_nptr = (size_t)( lcrp->nodes             * sizeof(int*) );
	size_a2ai = (size_t)( lcrp->nodes*lcrp->nodes * sizeof(int)  );

	lcrp->lnEnts   = (mat_nnz_t*)       allocateMemory( lcrp->nodes*sizeof(mat_nnz_t), "lcrp->lnEnts" ); 
	lcrp->lnrows   = (mat_idx_t*)       allocateMemory( lcrp->nodes*sizeof(mat_idx_t), "lcrp->lnrows" ); 
	lcrp->lfEnt    = (mat_nnz_t*)       allocateMemory( lcrp->nodes*sizeof(mat_nnz_t), "lcrp->lfEnt" ); 
	lcrp->lfRow    = (mat_idx_t*)       allocateMemory( lcrp->nodes*sizeof(mat_idx_t), "lcrp->lfRow" ); 

	lcrp->wishes   = (mat_idx_t*)       allocateMemory( lcrp->nodes*sizeof(mat_idx_t),  "lcrp->wishes" ); 
	lcrp->dues     = (mat_idx_t*)       allocateMemory( lcrp->nodes*sizeof(mat_idx_t), "lcrp->dues" ); 


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

			for (i=0;i<cr->nrows;i++){
				if (cr->rpt[i] >= j*target_nnz){
					lcrp->lfRow[j] = i;
					lcrp->lfEnt[j] = cr->rpt[i];
					j = j+1;
				}
			}

		}
		else if (options & SPMVM_OPTION_WORKDIST_LNZE){
			DEBUG_LOG(1,"Distribute Matrix with EQUAL_LNZE on each PE");

			/* A first attempt should be blocks of equal size */
			target_rows = (cr->nrows/lcrp->nodes);

			lcrp->lfRow[0] = 0;
			lcrp->lfEnt[0] = 0;

			for (i=1; i<lcrp->nodes; i++){
				lcrp->lfRow[i] = lcrp->lfRow[i-1]+target_rows;
				lcrp->lfEnt[i] = cr->rpt[lcrp->lfRow[i]];
			}

			for (i=0; i<lcrp->nodes-1; i++){
				lcrp->lnrows[i] = lcrp->lfRow[i+1] - lcrp->lfRow[i] ;
				lcrp->lnEnts[i] = lcrp->lfEnt[i+1] - lcrp->lfEnt[i] ;
			}
			lcrp->lnrows[lcrp->nodes-1] = cr->nrows - lcrp->lfRow[lcrp->nodes-1] ;
			lcrp->lnEnts[lcrp->nodes-1] = cr->nEnts - lcrp->lfEnt[lcrp->nodes-1];

			/* Count number of local elements in each block */
			loc_count      = (int*)       allocateMemory( size_nint, "loc_count" ); 
			for (i=0; i<lcrp->nodes; i++) loc_count[i] = 0;     

			for (i=0; i<lcrp->nodes; i++){
				for (j=lcrp->lfEnt[i]; j<lcrp->lfEnt[i]+lcrp->lnEnts[i]; j++){
					if (cr->col[j] >= lcrp->lfRow[i] && 
							cr->col[j]<lcrp->lfRow[i]+lcrp->lnrows[i])
						loc_count[i]++;
				}
			}
			DEBUG_LOG(2,"First run: local elements:");
			hlpi = 0;
			for (i=0; i<lcrp->nodes; i++){
				hlpi += loc_count[i];
				DEBUG_LOG(2,"Block %"PRmatNNZ" %8d %"PRmatNNZ, i, loc_count[i], lcrp->lnEnts[i]);
			}
			target_lnze = hlpi/lcrp->nodes;
			DEBUG_LOG(2,"total local elements: %d | per PE: %d", hlpi, target_lnze);

			outer_convergence = 0; 
			outer_iter = 0;

			while(outer_convergence==0){ 

				DEBUG_LOG(2,"Convergence Iteration %d", outer_iter);

				for (i=0; i<lcrp->nodes-1; i++){

					ideal = 0;
					prev_rows  = lcrp->lnrows[i];
					prev_count = loc_count[i];

					while (ideal==0){

						trial_rows = (int)( (double)(prev_rows) * sqrt((1.0*target_lnze)/(1.0*prev_count)) );

						/* Check ob die Anzahl der Elemente schon das beste ist das ich erwarten kann */
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

				lcrp->lnrows[lcrp->nodes-1] = cr->nrows - lcrp->lfRow[lcrp->nodes-1];
				lcrp->lnEnts[lcrp->nodes-1] = cr->nEnts - lcrp->lfEnt[lcrp->nodes-1];
				loc_count[lcrp->nodes-1] = 0;
				for (j=lcrp->lfEnt[lcrp->nodes-1]; j<cr->nEnts; j++)
					if (cr->col[j] >= lcrp->lfRow[lcrp->nodes-1]) loc_count[lcrp->nodes-1]++;

				DEBUG_LOG(2,"Next run: outer_iter=%d:", outer_iter);
				hlpi = 0;
				for (i=0; i<lcrp->nodes; i++){
					hlpi += loc_count[i];
					DEBUG_LOG(2,"Block %3"PRmatIDX" %8d %"PRmatIDX, i, loc_count[i], lcrp->lnEnts[i]);
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
					DEBUG_LOG(1,"PE%u: lfRow=%"PRmatIDX" lfEnt=%"PRmatNNZ" lnrows=%"PRmatIDX" lnEnts=%"PRmatNNZ, i, lcrp->lfRow[i], lcrp->lfEnt[i], lcrp->lnrows[i], lcrp->lnEnts[i]);
			

			free(loc_count);
		}
		else {

			DEBUG_LOG(1,"Distribute Matrix with EQUAL_ROWS on each PE");
			target_rows = (cr->nrows/lcrp->nodes);

			lcrp->lfRow[0] = 0;
			lcrp->lfEnt[0] = 0;

			for (i=1; i<lcrp->nodes; i++){
				lcrp->lfRow[i] = lcrp->lfRow[i-1]+target_rows;
				lcrp->lfEnt[i] = cr->rpt[lcrp->lfRow[i]];
			}
		}


		for (i=0; i<lcrp->nodes-1; i++){
			lcrp->lnrows[i] = lcrp->lfRow[i+1] - lcrp->lfRow[i] ;
			lcrp->lnEnts[i] = lcrp->lfEnt[i+1] - lcrp->lfEnt[i] ;
		}

		lcrp->lnrows[lcrp->nodes-1] = cr->nrows - lcrp->lfRow[lcrp->nodes-1] ;
		lcrp->lnEnts[lcrp->nodes-1] = cr->nEnts - lcrp->lfEnt[lcrp->nodes-1];
	}

	/****************************************************************************
	 *******            Distribute correct share to all PEs               *******
	 ***************************************************************************/

	MPI_safecall(MPI_Bcast(lcrp->lfRow,  lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD));
	MPI_safecall(MPI_Bcast(lcrp->lfEnt,  lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD));
	MPI_safecall(MPI_Bcast(lcrp->lnrows, lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD));
	MPI_safecall(MPI_Bcast(lcrp->lnEnts, lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD));

	/****************************************************************************
	 *******   Allocate memory for matrix in distributed CRS storage      *******
	 ***************************************************************************/

	CR_TYPE *fullCR;
	fullCR = (CR_TYPE *) allocateMemory(sizeof(CR_TYPE),"fullCR");

	fullCR->val = (mat_data_t*) allocateMemory(lcrp->lnEnts[me]*sizeof( mat_data_t ),"fullMatrix->val" );
	fullCR->col = (mat_idx_t*) allocateMemory(lcrp->lnEnts[me]*sizeof( mat_idx_t ),"fullMatrix->col" ); 
	fullCR->rpt = (mat_idx_t*) allocateMemory((lcrp->lnrows[me]+1)*sizeof( mat_idx_t ),"fullMatrix->rpt" ); 
	fullCR->nrows = lcrp->lnrows[me];
	fullCR->nEnts = lcrp->lnEnts[me];

	setup->fullMatrix->data = fullCR;
	
	/****************************************************************************
	 *******   Fill all fields with their corresponding values            *******
	 ***************************************************************************/


	#pragma omp parallel for schedule(runtime)
	for (i=0; i<lcrp->lnEnts[me]; i++) fullCR->val[i] = 0.0;

	#pragma omp parallel for schedule(runtime)
	for (i=0; i<lcrp->lnEnts[me]; i++) fullCR->col[i] = 0.0;

	#pragma omp parallel for schedule(runtime)
	for (i=0; i<lcrp->lnrows[me]; i++) fullCR->rpt[i] = 0.0;


	MPI_safecall(MPI_Scatterv ( cr->val, (int *)lcrp->lnEnts, (int *)lcrp->lfEnt, MPI_MYDATATYPE, 
				fullCR->val, (int)lcrp->lnEnts[me],  MPI_MYDATATYPE, 0, MPI_COMM_WORLD));

	MPI_safecall(MPI_Scatterv ( cr->col, (int *)lcrp->lnEnts, (int *)lcrp->lfEnt, MPI_INTEGER,
				fullCR->col, (int)lcrp->lnEnts[me],  MPI_INTEGER, 0, MPI_COMM_WORLD));

	MPI_safecall(MPI_Scatterv ( cr->rpt, (int *)lcrp->lnrows, (int *)lcrp->lfRow, MPI_INTEGER,
				fullCR->rpt, (int)lcrp->lnrows[me],  MPI_INTEGER, 0, MPI_COMM_WORLD));

	/****************************************************************************
	 *******        Adapt row pointer to local numbering on this PE       *******         
	 ***************************************************************************/

	for (i=0;i<lcrp->lnrows[me]+1;i++)
		fullCR->rpt[i] =  fullCR->rpt[i] - lcrp->lfEnt[me]; 

	/* last entry of row_ptr holds the local number of entries */
	fullCR->rpt[lcrp->lnrows[me]] = lcrp->lnEnts[me]; 

	/****************************************************************************
	 *******         Extract maximum number of local elements             *******
	 ***************************************************************************/

	max_loc_elements = 0;
	for (i=0;i<lcrp->nodes;i++)
		if (max_loc_elements<lcrp->lnrows[i]) max_loc_elements = lcrp->lnrows[i];

	nEnts_glob = lcrp->lfEnt[lcrp->nodes-1]+lcrp->lnEnts[lcrp->nodes-1]; 


	/****************************************************************************
	 *******         Assemble wish- and duelists for communication        *******
	 ***************************************************************************/

	size_pval = (size_t)( max_loc_elements * sizeof(int) );
	size_revc = (size_t)( nEnts_glob       * sizeof(int) );

	size_col  = (size_t)( (size_t)(lcrp->lnEnts[me])   * sizeof( int ) );
	item_from       = (int*) allocateMemory( size_nint, "item_from" ); 
	wishlist_counts = (unsigned int*) allocateMemory( lcrp->nodes*sizeof(unsigned int), "wishlist_counts" ); 
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
			if (lcrp->lfRow[j]<fullCR->col[i]+1) {
				/* Entsprechendes Paarelement liegt auf PE j */
				comm_remotePE[i] = j;
				wishlist_counts[j]++;
				comm_remoteEl[i] = fullCR->col[i] -lcrp->lfRow[j];
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

	this_pseudo_col = lcrp->lnrows[me];
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
			fullCR->col[i] =  comm_remoteEl[i];
		else // remote
			fullCR->col[i] = pseudocol[revcol[fullCR->col[i]]];
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

	size_dues = (size_t)( acc_transfer_dues   * sizeof(int) );

	MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));

	//int **wishlist      = (int**) allocateMemory( size_nptr, "wishlist" ); 
	lcrp->duelist       = (int**) allocateMemory( size_nptr, "lcrp->duelist" ); 
	//int *wishlist_mem  = (int*)  allocateMemory( size_wish, "wishlist_mem" ); 
	int *duelist_mem   = (int*)  allocateMemory( size_dues, "duelist_mem" ); 
	int *wish_displ    = (int*)  allocateMemory( size_nint, "wish_displ" ); 
	lcrp->due_displ     = (int*)  allocateMemory( size_nint, "lcrp->due_displ" ); 
	lcrp->hput_pos      = (int*)  allocateMemory( size_nptr, "lcrp->hput_pos" ); 

	acc_dues = 0;
	acc_wishes = 0;



	for (i=0; i<lcrp->nodes; i++){

		lcrp->due_displ[i]  = acc_dues;
		wish_displ[i] = acc_wishes;
		lcrp->duelist[i]    = &(duelist_mem[acc_dues]);
		wishlist[i]   = &(wishlist_mem[acc_wishes]);
		lcrp->hput_pos[i]   = lcrp->lnrows[me]+acc_wishes;

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
		wishlist[i][j] = cwishlist[i][j]; 

	/* Alle Source-Variablen sind bei Scatterv nur auf root relevant; d.h. ich
	 * nehme automatisch _immer_ die richtige (lokale) wishlist zum Verteilen */

	for(i=0; i<lcrp->nodes; i++) {
		MPI_safecall(MPI_Scatterv ( 
					wishlist_mem, (int *)lcrp->wishes, wish_displ, MPI_INTEGER, 
					lcrp->duelist[i], lcrp->dues[i], MPI_INTEGER, i, MPI_COMM_WORLD ));
	}

	/****************************************************************************
	 *******        Setup the variant using local/non-local arrays        *******
	 ***************************************************************************/
	if (!(options & SPMVM_OPTION_NO_SPLIT_KERNELS)) { // split computation


		pseudo_ldim = lcrp->lnrows[me]+lcrp->halo_elements ;

		lnEnts_l=0;
		for (i=0; i<lcrp->lnEnts[me];i++)
			if (fullCR->col[i]<lcrp->lnrows[me]) lnEnts_l++;


		lnEnts_r = lcrp->lnEnts[me]-lnEnts_l;

		DEBUG_LOG(1,"PE%d: Rows=%"PRmatIDX"\t Ents=%"PRmatNNZ"(l),%"PRmatNNZ"(r),%"PRmatNNZ"(g)\t pdim=%6d", 
				me, lcrp->lnrows[me], lnEnts_l, lnEnts_r, lcrp->lnEnts[me], pseudo_ldim );

		CR_TYPE *localCR;
		CR_TYPE *remoteCR;
	
		localCR = (CR_TYPE *) allocateMemory(sizeof(CR_TYPE),"fullCR");
		remoteCR = (CR_TYPE *) allocateMemory(sizeof(CR_TYPE),"fullCR");

		localCR->val = (mat_data_t*) allocateMemory(lnEnts_l*sizeof( mat_data_t ),"localMatrix->val" ); 
		localCR->col = (mat_idx_t*) allocateMemory(lnEnts_l*sizeof( mat_idx_t ),"localMatrix->col" ); 
		localCR->rpt = (mat_idx_t*) allocateMemory((lcrp->lnrows[me]+1)*sizeof( mat_idx_t ),"localMatrix->rpt" ); 

		remoteCR->val = (mat_data_t*) allocateMemory(lnEnts_r*sizeof( mat_data_t ),"remoteMatrix->val" ); 
		remoteCR->col = (mat_idx_t*) allocateMemory(lnEnts_r*sizeof( mat_idx_t ),"remoteMatrix->col" ); 
		remoteCR->rpt = (mat_idx_t*) allocateMemory((lcrp->lnrows[me]+1)*sizeof( mat_idx_t ),"remoteMatrix->rpt" ); 


		localCR->nrows = lcrp->lnrows[me];
		localCR->nEnts = lnEnts_l;
		
		remoteCR->nrows = lcrp->lnrows[me];
		remoteCR->nEnts = lnEnts_r;
		
		setup->localMatrix->data = localCR;
		setup->remoteMatrix->data = remoteCR;

#pragma omp parallel for schedule(runtime)
		for (i=0; i<lnEnts_l; i++) localCR->val[i] = 0.0;

#pragma omp parallel for schedule(runtime)
		for (i=0; i<lnEnts_l; i++) localCR->col[i] = 0.0;

#pragma omp parallel for schedule(runtime)
		for (i=0; i<lnEnts_r; i++) remoteCR->val[i] = 0.0;

#pragma omp parallel for schedule(runtime)
		for (i=0; i<lnEnts_r; i++) remoteCR->col[i] = 0.0;


		localCR->rpt[0] = 0;
		remoteCR->rpt[0] = 0;

		MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));
		DEBUG_LOG(1,"PE%d: lnrows=%"PRmatIDX" row_ptr=%"PRmatNNZ"..%"PRmatNNZ, 
				me, lcrp->lnrows[me], fullCR->rpt[0], fullCR->rpt[lcrp->lnrows[me]]);
		fflush(stdout);
		MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));

		for (i=0; i<lcrp->lnrows[me]; i++){

			current_l = 0;
			current_r = 0;

			for (j=fullCR->rpt[i]; j<fullCR->rpt[i+1]; j++){

				if (fullCR->col[j]<lcrp->lnrows[me]){
					/* local element */
					localCR->col[ localCR->rpt[i]+current_l ] = fullCR->col[j]; 
					localCR->val[ localCR->rpt[i]+current_l ] = fullCR->val[j]; 
					current_l++;
				}
				else{
					/* remote element */
					remoteCR->col[ remoteCR->rpt[i]+current_r ] = fullCR->col[j];
					remoteCR->val[ remoteCR->rpt[i]+current_r ] = fullCR->val[j];
					current_r++;
				}

			}  

			localCR->rpt[i+1] = localCR->rpt[i] + current_l;
			remoteCR->rpt[i+1] = remoteCR->rpt[i] + current_r;
		}

		IF_DEBUG(2){
			for (i=0; i<lcrp->lnrows[me]+1; i++)
				DEBUG_LOG(2,"--Row_ptrs-- PE%d: i=%"PRmatIDX" local=%"PRmatNNZ" remote=%"PRmatNNZ, 
						me, i, localCR->rpt[i], remoteCR->rpt[i]);
			for (e=0; e<localCR->rpt[lcrp->lnrows[me]]; e++)
				DEBUG_LOG(2,"-- local -- PE%d: localCR->col[%"PRmatIDX"]=%"PRmatIDX, me, e, localCR->col[e]);
			for (e=0; e<remoteCR->rpt[lcrp->lnrows[me]]; e++)
				DEBUG_LOG(2,"-- remote -- PE%d: remoteCR->col[%"PRmatIDX"]=%"PRmatIDX, me, e, remoteCR->col[e]);
		}
		fflush(stdout);
		MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));

	} /*else{
		localCR->rpt = (int*)    allocateMemory( sizeof(int), "localCR->rpt" ); 
		remoteCR->rpt = (int*)    allocateMemory( sizeof(int), "remoteCR->rpt" ); 
		localCR->col       = (int*)    allocateMemory( sizeof(int), "localCR->col" ); 
		remoteCR->col       = (int*)    allocateMemory( sizeof(int), "remoteCR->col" ); 
		localCR->val       = (mat_data_t*) allocateMemory( sizeof(mat_data_t), "localCR->val" ); 
		remoteCR->val       = (mat_data_t*) allocateMemory( sizeof(mat_data_t), "remoteCR->val" ); 
	}*/

	freeMemory ( size_mem,  "wishlist_mem",    wishlist_mem);
	freeMemory ( size_mem,  "cwishlist_mem",   cwishlist_mem);
	freeMemory ( size_nptr, "wishlist",        wishlist);
	freeMemory ( size_nptr, "cwishlist",       cwishlist);
	freeMemory ( size_a2ai, "tmp_transfers",   tmp_transfers);
	freeMemory ( size_nint, "wishlist_counts", wishlist_counts);
	freeMemory ( size_nint, "item_from",       item_from);


	/* Free memory for CR stored matrix and sweep memory */
	//freeCRMatrix( cr );

}

void SpMVM_createDistributedSetup(SETUP_TYPE * setup, CR_TYPE* cr, char * matrixPath, int options)
{

	/* Counting and auxilliary variables */
	mat_idx_t i, j;
	mat_nnz_t e;
	int hlpi;

	/* Processor rank (MPI-process) */
	int me; 

	/* MPI-Errorcode */

	mat_nnz_t max_loc_elements, thisentry;
	int *present_values;

	LCRP_TYPE *lcrp;

	int acc_dues;

	int *tmp_transfers;

	int target_nnz;
	int acc_wishes;

	int nEnts_glob;

	/* Counter how many entries are requested from each PE */
	int *item_from;

	unsigned int *wishlist_counts;

	int *wishlist_mem,  **wishlist;
	int *cwishlist_mem, **cwishlist;

	int this_pseudo_col;
	int *pseudocol;
	int *globcol;
	int *revcol;

	int *comm_remotePE, *comm_remoteEl;

	int target_rows;

	mat_nnz_t lnEnts_l, lnEnts_r;
	int current_l, current_r;

	int pseudo_ldim;
	int acc_transfer_wishes, acc_transfer_dues;

	size_t size_nint, size_col;
	size_t size_revc, size_a2ai, size_nptr, size_pval;  
	size_t size_mem, size_dues;

	MATRIX_TYPE *fullMatrix;
	MATRIX_TYPE *localMatrix;
	MATRIX_TYPE *remoteMatrix;
	
	MPI_Info info = MPI_INFO_NULL;
	MPI_File file_handle;
	MPI_Offset offset_in_file;
	MPI_Status status;

	/****************************************************************************
	 *******            ........ Executable statements ........           *******
	 ***************************************************************************/

	MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD, &me));

	DEBUG_LOG(1,"Entering setup_communication_parallel");

	lcrp = (LCRP_TYPE*) allocateMemory( sizeof(LCRP_TYPE), "lcrp");
	fullMatrix = (MATRIX_TYPE *)allocateMemory(sizeof(MATRIX_TYPE),"fullMatrix");
	localMatrix = (MATRIX_TYPE *)allocateMemory(sizeof(MATRIX_TYPE),"fullMatrix");
	remoteMatrix = (MATRIX_TYPE *)allocateMemory(sizeof(MATRIX_TYPE),"fullMatrix");
	
	setup->communicator = lcrp;
	setup->fullMatrix = fullMatrix;
	setup->localMatrix = localMatrix;
	setup->remoteMatrix = remoteMatrix;

	/*lcrp->fullRowPerm = NULL;
	lcrp->fullInvRowPerm = NULL;
	lcrp->splitRowPerm = NULL;
	lcrp->splitInvRowPerm = NULL;
	lcrp->fullMatrix = NULL;
	lcrp->localMatrix = NULL;
	lcrp->remoteMatrix = NULL;
	lcrp->lrow_ptr_l = NULL; 
	lcrp->lrow_ptr_r = NULL; 
	lcrp->lcol       = NULL;
	lcrp->rcol       = NULL;
	lcrp->lval       = NULL;
	lcrp->rval       = NULL;*/

/*	if (me==0) {
		lcrp->nEnts = cr->nEnts;
		lcrp->nrows = cr->nrows;
	}

	MPI_safecall(MPI_Bcast(&lcrp->nEnts,1,MPI_INT,0,MPI_COMM_WORLD));
	MPI_safecall(MPI_Bcast(&lcrp->nrows,1,MPI_INT,0,MPI_COMM_WORLD));
*/
#pragma omp parallel
	lcrp->threads = omp_get_num_threads(); 

	lcrp->nodes = SpMVM_getNumberOfProcesses();

	size_nint = (size_t)( (size_t)(lcrp->nodes)   * sizeof(int)  );
	size_nptr = (size_t)( lcrp->nodes             * sizeof(int*) );
	size_a2ai = (size_t)( lcrp->nodes*lcrp->nodes * sizeof(int)  );

	lcrp->lnEnts   = (mat_nnz_t*)       allocateMemory( lcrp->nodes*sizeof(mat_nnz_t), "lcrp->lnEnts" ); 
	lcrp->lnrows   = (mat_idx_t*)       allocateMemory( lcrp->nodes*sizeof(mat_idx_t), "lcrp->lnrows" ); 
	lcrp->lfEnt    = (mat_nnz_t*)       allocateMemory( lcrp->nodes*sizeof(mat_nnz_t), "lcrp->lfEnt" ); 
	lcrp->lfRow    = (mat_idx_t*)       allocateMemory( lcrp->nodes*sizeof(mat_idx_t), "lcrp->lfRow" ); 

	lcrp->wishes   = (mat_idx_t*)       allocateMemory( lcrp->nodes*sizeof(mat_idx_t),  "lcrp->wishes" ); 
	lcrp->dues     = (mat_idx_t*)       allocateMemory( lcrp->nodes*sizeof(mat_idx_t), "lcrp->dues" ); 

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

			for (i=0;i<cr->nrows;i++){
				if (cr->rpt[i] >= j*target_nnz){
					lcrp->lfRow[j] = i;
					lcrp->lfEnt[j] = cr->rpt[i];
					j = j+1;
				}
			}

		}
		else {

			DEBUG_LOG(1,"Distribute Matrix with EQUAL_ROWS on each PE");
			target_rows = (cr->nrows/lcrp->nodes);

			lcrp->lfRow[0] = 0;
			lcrp->lfEnt[0] = 0;

			for (i=1; i<lcrp->nodes; i++){
				lcrp->lfRow[i] = lcrp->lfRow[i-1]+target_rows;
				lcrp->lfEnt[i] = cr->rpt[lcrp->lfRow[i]];
			}
		}


		for (i=0; i<lcrp->nodes-1; i++){
			lcrp->lnrows[i] = lcrp->lfRow[i+1] - lcrp->lfRow[i] ;
			lcrp->lnEnts[i] = lcrp->lfEnt[i+1] - lcrp->lfEnt[i] ;
		}

		lcrp->lnrows[lcrp->nodes-1] = cr->nrows - lcrp->lfRow[lcrp->nodes-1] ;
		lcrp->lnEnts[lcrp->nodes-1] = cr->nEnts - lcrp->lfEnt[lcrp->nodes-1];
	}

	/****************************************************************************
	 *******            Distribute correct share to all PEs               *******
	 ***************************************************************************/

	MPI_safecall(MPI_Bcast(lcrp->lfRow,  lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD));
	MPI_safecall(MPI_Bcast(lcrp->lfEnt,  lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD));
	MPI_safecall(MPI_Bcast(lcrp->lnrows, lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD));
	MPI_safecall(MPI_Bcast(lcrp->lnEnts, lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD));

	DEBUG_LOG(1,"local rows          = %"PRmatIDX,lcrp->lnrows[me]);
	DEBUG_LOG(1,"local rows (offset) = %"PRmatIDX,lcrp->lfRow[me]);
	DEBUG_LOG(1,"local entries          = %"PRmatNNZ,lcrp->lnEnts[me]);
	DEBUG_LOG(1,"local entires (offset) = %"PRmatNNZ,lcrp->lfEnt[me]);

	/****************************************************************************
	 *******   Allocate memory for matrix in distributed CRS storage      *******
	 ***************************************************************************/

	CR_TYPE *fullCR;
	fullCR = (CR_TYPE *) allocateMemory(sizeof(CR_TYPE),"fullCR");

	fullCR->val = (mat_data_t*) allocateMemory(lcrp->lnEnts[me]*sizeof( mat_data_t ),"fullMatrix->val" );
	fullCR->col = (mat_idx_t*) allocateMemory(lcrp->lnEnts[me]*sizeof( mat_idx_t ),"fullMatrix->col" ); 
	fullCR->rpt = (mat_idx_t*) allocateMemory((lcrp->lnrows[me]+1)*sizeof( mat_idx_t ),"fullMatrix->rpt" ); 
	fullCR->nrows = lcrp->lnrows[me];
	fullCR->nEnts = lcrp->lnEnts[me];
	setup->fullMatrix->data = fullCR;

	/****************************************************************************
	 *******   Fill all fields with their corresponding values            *******
	 ***************************************************************************/


	#pragma omp parallel for schedule(runtime)
	for (i=0; i<lcrp->lnEnts[me]; i++) fullCR->val[i] = 0.0;

	#pragma omp parallel for schedule(runtime)
	for (i=0; i<lcrp->lnEnts[me]; i++) fullCR->col[i] = 0.0;

	#pragma omp parallel for schedule(runtime)
	for (i=0; i<lcrp->lnrows[me]; i++) fullCR->rpt[i] = 0.0;

	/* replace scattering with read-in */
	DEBUG_LOG(1,"Opening file %s for parallel read-in",matrixPath);
	MPI_safecall(MPI_File_open(MPI_COMM_WORLD, matrixPath, MPI_MODE_RDONLY, info, &file_handle));

	int datatype;
	MPI_safecall(MPI_File_seek(file_handle,0,MPI_SEEK_SET));
	MPI_safecall(MPI_File_read(file_handle,&datatype,1,MPI_INTEGER,&status));

	/* read col */
	offset_in_file = (4+cr->nrows+1)*sizeof(int) + (lcrp->lfEnt[me])*sizeof(int);
	DEBUG_LOG(1,"Read col -- offset=%lu | %"PRmatNNZ,(size_t)offset_in_file,lcrp->lfEnt[me]);
	MPI_safecall(MPI_File_seek(file_handle, offset_in_file, MPI_SEEK_SET));
	MPI_safecall(MPI_File_read(file_handle, fullCR->col, lcrp->lnEnts[me], MPI_INTEGER, &status));

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
					offset_in_file = (4+cr->nrows+1)*sizeof(int) + (cr->nEnts)*sizeof(int) + (lcrp->lfEnt[me])*sizeof(float);
					DEBUG_LOG(1,"Read val -- offset=%lu",(size_t)offset_in_file);
					MPI_safecall(MPI_File_seek(file_handle, offset_in_file, MPI_SEEK_SET));
					MPI_safecall(MPI_File_read(file_handle, tmp, lcrp->lnEnts[me], MPI_FLOAT, &status));
					for (i = 0; i<lcrp->lnEnts[me]; i++) fullCR->val[i] = (mat_data_t) tmp[i];
					free(tmp);
					break;
				}
			case DATATYPE_DOUBLE:
				{
					double *tmp = (double *)allocateMemory(lcrp->lnEnts[me]*sizeof(double), "tmp");
					offset_in_file = (4+cr->nrows+1)*sizeof(int) + (cr->nEnts)*sizeof(int) + (lcrp->lfEnt[me])*sizeof(double);
					DEBUG_LOG(1,"Read val -- offset=%lu",(size_t)offset_in_file);

					MPI_safecall(MPI_File_seek(file_handle, offset_in_file, MPI_SEEK_SET));
					MPI_safecall(MPI_File_read(file_handle, tmp, lcrp->lnEnts[me], MPI_DOUBLE, &status));
					for (i = 0; i<lcrp->lnEnts[me]; i++)
						fullCR->val[i] = (mat_data_t) tmp[i];
					free(tmp);
					break;
				}
			case DATATYPE_COMPLEX_DOUBLE:
				{
					_Complex double *tmp = (_Complex double *)allocateMemory(lcrp->lnEnts[me]*sizeof(_Complex double), "tmp");
					offset_in_file = (4+cr->nrows+1)*sizeof(int) + (cr->nEnts)*sizeof(int) + (lcrp->lfEnt[me])*sizeof(_Complex double);
					DEBUG_LOG(1,"Read val -- offset=%lu",(size_t)offset_in_file);

					MPI_Datatype tmpDT;
					MPI_safecall(MPI_Type_contiguous(2,MPI_DOUBLE,&tmpDT));
					MPI_safecall(MPI_Type_commit(&tmpDT));

					MPI_safecall(MPI_File_seek(file_handle, offset_in_file, MPI_SEEK_SET));
					MPI_safecall(MPI_File_read(file_handle, tmp, lcrp->lnEnts[me], tmpDT, &status));

					for (i = 0; i<lcrp->lnEnts[me]; i++) fullCR->val[i] = (mat_data_t) tmp[i];

					free(tmp);
					MPI_safecall(MPI_Type_free(&tmpDT));
					break;
				}
			case DATATYPE_COMPLEX_FLOAT:
				{
					_Complex float *tmp = (_Complex float *)allocateMemory(lcrp->lnEnts[me]*sizeof(_Complex float), "tmp");
					offset_in_file = (4+cr->nrows+1)*sizeof(int) + (cr->nEnts)*sizeof(int) + (lcrp->lfEnt[me])*sizeof(_Complex float);
					DEBUG_LOG(1,"Read val -- offset=%lu",(size_t)offset_in_file);

					MPI_Datatype tmpDT;
					MPI_safecall(MPI_Type_contiguous(2,MPI_FLOAT,&tmpDT));
					MPI_safecall(MPI_Type_commit(&tmpDT));

					MPI_safecall(MPI_File_seek(file_handle, offset_in_file, MPI_SEEK_SET));
					MPI_safecall(MPI_File_read(file_handle, tmp, lcrp->lnEnts[me], tmpDT, &status));

					for (i = 0; i<lcrp->lnEnts[me]; i++) fullCR->val[i] = (mat_data_t) tmp[i];

					free(tmp);
					MPI_safecall(MPI_Type_free(&tmpDT));
					break;
				}

		}
	} else {

		offset_in_file = (4+cr->nrows+1)*sizeof(int) + (cr->nEnts)*sizeof(int) + (lcrp->lfEnt[me])*sizeof(mat_data_t);
		DEBUG_LOG(1,"Read val -- offset=%lu",(size_t)offset_in_file);
		MPI_safecall(MPI_File_seek(file_handle, offset_in_file, MPI_SEEK_SET));
		MPI_safecall(MPI_File_read(file_handle, fullCR->val, lcrp->lnEnts[me], MPI_MYDATATYPE, &status));
	}

	MPI_safecall(MPI_File_close(&file_handle));

	/* Offsets are scattered */
	MPI_safecall(MPI_Scatterv ( cr->rpt, (int *)lcrp->lnrows, (int *)lcrp->lfRow, MPI_INTEGER,
			fullCR->rpt, (int)lcrp->lnrows[me],  MPI_INTEGER, 0, MPI_COMM_WORLD));

	/****************************************************************************
	 *******        Adapt row pointer to local numbering on this PE       *******         
	 ***************************************************************************/

	for (i=0;i<lcrp->lnrows[me]+1;i++)
		fullCR->rpt[i] =  fullCR->rpt[i] - lcrp->lfEnt[me]; 

	/* last entry of row_ptr holds the local number of entries */
	fullCR->rpt[lcrp->lnrows[me]] = lcrp->lnEnts[me]; 

	/****************************************************************************
	 *******         Extract maximum number of local elements             *******
	 ***************************************************************************/

	max_loc_elements = 0;
	for (i=0;i<lcrp->nodes;i++)
		if (max_loc_elements<lcrp->lnrows[i]) max_loc_elements = lcrp->lnrows[i];

	nEnts_glob = lcrp->lfEnt[lcrp->nodes-1]+lcrp->lnEnts[lcrp->nodes-1]; 


	/****************************************************************************
	 *******         Assemble wish- and duelists for communication        *******
	 ***************************************************************************/

	size_col  = (size_t)( (size_t)(lcrp->lnEnts[me])   * sizeof( int ) );
	size_pval = (size_t)( max_loc_elements * sizeof(int) );
	size_revc = (size_t)( nEnts_glob       * sizeof(int) );

	item_from       = (int*) allocateMemory( size_nint, "item_from" ); 
	wishlist_counts = (unsigned int*) allocateMemory( size_nint, "wishlist_counts" ); 
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
			if (lcrp->lfRow[j]<fullCR->col[i]+1) {
				/* Entsprechendes Paarelement liegt auf PE j */
				comm_remotePE[i] = j;
				wishlist_counts[j]++;
				comm_remoteEl[i] = fullCR->col[i] -lcrp->lfRow[j];
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

	this_pseudo_col = lcrp->lnrows[me];
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
		if (comm_remotePE[i] == me) fullCR->col[i] = comm_remoteEl[i];
		else                        fullCR->col[i] = pseudocol[revcol[fullCR->col[i]]];
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

	size_dues = (size_t)( acc_transfer_dues   * sizeof(int) );

	//wishlist      = (int**) allocateMemory( size_nptr, "wishlist" ); 
	lcrp->duelist       = (int**) allocateMemory( size_nptr, "lcrp->duelist" ); 
	//wishlist_mem  = (int*)  allocateMemory( size_wish, "wishlist_mem" ); 
	int *duelist_mem   = (int*)  allocateMemory( size_dues, "duelist_mem" ); 
	int *wish_displ    = (int*)  allocateMemory( size_nint, "wish_displ" ); 
	lcrp->due_displ     = (int*)  allocateMemory( size_nint, "lcrp->due_displ" ); 
	lcrp->hput_pos      = (int*)  allocateMemory( size_nptr, "lcrp->hput_pos" ); 

	acc_dues = 0;
	acc_wishes = 0;

	for (i=0; i<lcrp->nodes; i++){

		lcrp->due_displ[i]  = acc_dues;
		wish_displ[i] = acc_wishes;
		lcrp->duelist[i]    = &(duelist_mem[acc_dues]);
		wishlist[i]   = &(wishlist_mem[acc_wishes]);
		lcrp->hput_pos[i]   = lcrp->lnrows[me]+acc_wishes;

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
		wishlist[i][j] = cwishlist[i][j]; 

	/* Alle Source-Variablen sind bei Scatterv nur auf root relevant; d.h. ich
	 * nehme automatisch _immer_ die richtige (lokale) wishlist zum Verteilen */

	for(i=0; i<lcrp->nodes; i++) MPI_safecall(MPI_Scatterv ( 
			wishlist_mem, (int *)lcrp->wishes, (int *)wish_displ, MPI_INTEGER, 
			(int *)lcrp->duelist[i], (int)lcrp->dues[i], MPI_INTEGER, i, MPI_COMM_WORLD ));

	/****************************************************************************
	 *******        Setup the variant using local/non-local arrays        *******
	 ***************************************************************************/
	if (!(options & SPMVM_OPTION_NO_SPLIT_KERNELS)) { // split computation


		pseudo_ldim = lcrp->lnrows[me]+lcrp->halo_elements ;

		lnEnts_l=0;
		for (i=0; i<lcrp->lnEnts[me];i++)
			if (fullCR->col[i]<lcrp->lnrows[me]) lnEnts_l++;


		lnEnts_r = lcrp->lnEnts[me]-lnEnts_l;

		DEBUG_LOG(1,"PE%d: Rows=%"PRmatIDX"\t Ents=%"PRmatNNZ"(l),%"PRmatNNZ"(r),%"PRmatNNZ"(g)\t pdim=%6d", 
				me, lcrp->lnrows[me], lnEnts_l, lnEnts_r, lcrp->lnEnts[me], pseudo_ldim );

		CR_TYPE *localCR;
		CR_TYPE *remoteCR;
	
		localCR = (CR_TYPE *) allocateMemory(sizeof(CR_TYPE),"fullCR");
		remoteCR = (CR_TYPE *) allocateMemory(sizeof(CR_TYPE),"fullCR");

		localCR->val = (mat_data_t*) allocateMemory(lnEnts_l*sizeof( mat_data_t ),"localMatrix->val" ); 
		localCR->col = (mat_idx_t*) allocateMemory(lnEnts_l*sizeof( mat_idx_t ),"localMatrix->col" ); 
		localCR->rpt = (mat_idx_t*) allocateMemory((lcrp->lnrows[me]+1)*sizeof( mat_idx_t ),"localMatrix->rpt" ); 

		remoteCR->val = (mat_data_t*) allocateMemory(lnEnts_r*sizeof( mat_data_t ),"remoteMatrix->val" ); 
		remoteCR->col = (mat_idx_t*) allocateMemory(lnEnts_r*sizeof( mat_idx_t ),"remoteMatrix->col" ); 
		remoteCR->rpt = (mat_idx_t*) allocateMemory((lcrp->lnrows[me]+1)*sizeof( mat_idx_t ),"remoteMatrix->rpt" ); 
		localCR->nrows = lcrp->lnrows[me];
		localCR->nEnts = lnEnts_l;
		
		remoteCR->nrows = lcrp->lnrows[me];
		remoteCR->nEnts = lnEnts_r;

		setup->localMatrix->data = localCR;
		setup->remoteMatrix->data = remoteCR;

#pragma omp parallel for schedule(runtime)
		for (i=0; i<lnEnts_l; i++) localCR->val[i] = 0.0;

#pragma omp parallel for schedule(runtime)
		for (i=0; i<lnEnts_l; i++) localCR->col[i] = 0.0;

#pragma omp parallel for schedule(runtime)
		for (i=0; i<lnEnts_r; i++) remoteCR->val[i] = 0.0;

#pragma omp parallel for schedule(runtime)
		for (i=0; i<lnEnts_r; i++) remoteCR->col[i] = 0.0;


		localCR->rpt[0] = 0;
		remoteCR->rpt[0] = 0;

		MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));
		DEBUG_LOG(1,"PE%d: lnrows=%"PRmatIDX" row_ptr=%"PRmatNNZ"..%"PRmatNNZ, 
				me, lcrp->lnrows[me], fullCR->rpt[0], fullCR->rpt[lcrp->lnrows[me]]);
		fflush(stdout);
		MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));

		for (i=0; i<lcrp->lnrows[me]; i++){

			current_l = 0;
			current_r = 0;

			for (j=fullCR->rpt[i]; j<fullCR->rpt[i+1]; j++){

				if (fullCR->col[j]<lcrp->lnrows[me]){
					/* local element */
					localCR->col[ localCR->rpt[i]+current_l ] = fullCR->col[j]; 
					localCR->val[ localCR->rpt[i]+current_l ] = fullCR->val[j]; 
					current_l++;
				}
				else{
					/* remote element */
					remoteCR->col[ remoteCR->rpt[i]+current_r ] = fullCR->col[j];
					remoteCR->val[ remoteCR->rpt[i]+current_r ] = fullCR->val[j];
					current_r++;
				}

			}  

			localCR->rpt[i+1] = localCR->rpt[i] + current_l;
			remoteCR->rpt[i+1] = remoteCR->rpt[i] + current_r;
		}

		IF_DEBUG(2){
			for (i=0; i<lcrp->lnrows[me]+1; i++)
				DEBUG_LOG(2,"--Row_ptrs-- PE%d: i=%"PRmatIDX" local=%"PRmatNNZ" remote=%"PRmatNNZ, 
						me, i, localCR->rpt[i], remoteCR->rpt[i]);
			for (e=0; e<localCR->rpt[lcrp->lnrows[me]]; e++)
				DEBUG_LOG(2,"-- local -- PE%d: localCR->col[%"PRmatIDX"]=%"PRmatIDX, me, e, localCR->col[e]);
			for (e=0; e<remoteCR->rpt[lcrp->lnrows[me]]; e++)
				DEBUG_LOG(2,"-- remote -- PE%d: remoteCR->col[%"PRmatIDX"]=%"PRmatIDX, me, e, remoteCR->col[e]);
		}
		fflush(stdout);
		MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));

	} /*else{
		localCR->rpt = (int*)    allocateMemory( sizeof(int), "localCR->rpt" ); 
		remoteCR->rpt = (int*)    allocateMemory( sizeof(int), "remoteCR->rpt" ); 
		localCR->col       = (int*)    allocateMemory( sizeof(int), "localCR->col" ); 
		remoteCR->col       = (int*)    allocateMemory( sizeof(int), "remoteCR->col" ); 
		localCR->val       = (mat_data_t*) allocateMemory( sizeof(mat_data_t), "localCR->val" ); 
		remoteCR->val       = (mat_data_t*) allocateMemory( sizeof(mat_data_t), "remoteCR->val" ); 
	}*/
	
	freeMemory ( size_mem,  "wishlist_mem",    wishlist_mem);
	freeMemory ( size_mem,  "cwishlist_mem",   cwishlist_mem);
	freeMemory ( size_nptr, "wishlist",        wishlist);
	freeMemory ( size_nptr, "cwishlist",       cwishlist);
	freeMemory ( size_a2ai, "tmp_transfers",   tmp_transfers);
	freeMemory ( size_nint, "wishlist_counts", wishlist_counts);
	freeMemory ( size_nint, "item_from",       item_from);

}


// TODO
LCRP_TYPE *SpMVM_createCommunicator(CR_TYPE *cr, int options)
{
/*	int i, j, hlpi;
	int target_nnz;
	int target_rows;
	int *loc_count;
	int outer_iter, outer_convergence;
	int ideal, prev_rows;
	int target_lnze;

	int trial_count, prev_count, trial_rows;
	size_t size_nint;
	LCRP_TYPE *lcrp;
	int me = SpMVM_getRank();
	
	
	lcrp = (LCRP_TYPE*) allocateMemory( sizeof(LCRP_TYPE), "lcrp");

#pragma omp parallel
	lcrp->threads = omp_get_num_threads(); 

	MPI_safecall(MPI_Comm_size(MPI_COMM_WORLD, &(lcrp->nodes)));


	size_nint = (size_t)( (size_t)(lcrp->nodes)   * sizeof(int)  );

	lcrp->lnEnts   = (int*)       allocateMemory( size_nint, "lcrp->lnEnts" ); 
	lcrp->lnrows   = (int*)       allocateMemory( size_nint, "lcrp->lnrows" ); 
	lcrp->lfEnt    = (int*)       allocateMemory( size_nint, "lcrp->lfEnt" ); 
	lcrp->lfRow    = (int*)       allocateMemory( size_nint, "lcrp->lfRow" ); 

	lcrp->wishes   = (int*)       allocateMemory( size_nint, "lcrp->wishes" ); 
	lcrp->dues     = (int*)       allocateMemory( size_nint, "lcrp->dues" ); 
	
	if (me==0){

		if (options & SPMVM_OPTION_WORKDIST_NZE){
			DEBUG_LOG(1,"Distribute Matrix with EQUAL_NZE on each PE");
			target_nnz = (cr->nEnts/lcrp->nodes)+1; 

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
		else if (options & SPMVM_OPTION_WORKDIST_LNZE){
			DEBUG_LOG(1,"Distribute Matrix with EQUAL_LNZE on each PE");

			target_rows = (cr->nrows/lcrp->nodes);

			lcrp->lfRow[0] = 0;
			lcrp->lfEnt[0] = 0;

			for (i=1; i<lcrp->nodes; i++){
				lcrp->lfRow[i] = lcrp->lfRow[i-1]+target_rows;
				lcrp->lfEnt[i] = cr->rpt[lcrp->lfRow[i]];
			}

			for (i=0; i<lcrp->nodes-1; i++){
				lcrp->lnrows[i] = lcrp->lfRow[i+1] - lcrp->lfRow[i] ;
				lcrp->lnEnts[i] = lcrp->lfEnt[i+1] - lcrp->lfEnt[i] ;
			}
			lcrp->lnrows[lcrp->nodes-1] = cr->nrows - lcrp->lfRow[lcrp->nodes-1] ;
			lcrp->lnEnts[lcrp->nodes-1] = cr->nEnts - lcrp->lfEnt[lcrp->nodes-1];

			loc_count      = (int*)       allocateMemory( size_nint, "loc_count" ); 
			for (i=0; i<lcrp->nodes; i++) loc_count[i] = 0;     

			for (i=0; i<lcrp->nodes; i++){
				for (j=lcrp->lfEnt[i]; j<lcrp->lfEnt[i]+lcrp->lnEnts[i]; j++){
					if (cr->col[j] >= lcrp->lfRow[i] && 
							cr->col[j]<lcrp->lfRow[i]+lcrp->lnrows[i])
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

				lcrp->lnrows[lcrp->nodes-1] = cr->nrows - lcrp->lfRow[lcrp->nodes-1];
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
					DEBUG_LOG(1,"PE%3d: lfRow=%8d lfEnt=%12d lnrows=%8d lnEnts=%12d", i, lcrp->lfRow[i], 
							lcrp->lfEnt[i], lcrp->lnrows[i], lcrp->lnEnts[i]);
			

			free(loc_count);
		}
		else {

			DEBUG_LOG(1,"Distribute Matrix with EQUAL_ROWS on each PE");
			target_rows = (cr->nrows/lcrp->nodes);

			lcrp->lfRow[0] = 0;
			lcrp->lfEnt[0] = 0;

			for (i=1; i<lcrp->nodes; i++){
				lcrp->lfRow[i] = lcrp->lfRow[i-1]+target_rows;
				lcrp->lfEnt[i] = cr->rpt[lcrp->lfRow[i]];
			}
		}


		for (i=0; i<lcrp->nodes-1; i++){
			lcrp->lnrows[i] = lcrp->lfRow[i+1] - lcrp->lfRow[i] ;
			lcrp->lnEnts[i] = lcrp->lfEnt[i+1] - lcrp->lfEnt[i] ;
		}

		lcrp->lnrows[lcrp->nodes-1] = cr->nrows - lcrp->lfRow[lcrp->nodes-1] ;
		lcrp->lnEnts[lcrp->nodes-1] = cr->nEnts - lcrp->lfEnt[lcrp->nodes-1];
	}

	MPI_safecall(MPI_Bcast(lcrp->lfRow,  lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD));
	MPI_safecall(MPI_Bcast(lcrp->lfEnt,  lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD));
	MPI_safecall(MPI_Bcast(lcrp->lnrows, lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD));
	MPI_safecall(MPI_Bcast(lcrp->lnEnts, lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD));

	return lcrp;
*/
	UNUSED(cr);
	UNUSED(options);

	return NULL;
}

