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

#define MAX_NUM_THREADS 128
#define gettid() syscall(SYS_gettid)

static MPI_Comm single_node_comm;

static int getProcessorId() {

    cpu_set_t  cpu_set;
    int processorId;
    
	CPU_ZERO(&cpu_set);
    sched_getaffinity(gettid(),sizeof(cpu_set_t), &cpu_set);

    for (processorId=0;processorId<MAX_NUM_THREADS;processorId++){
        if (CPU_ISSET(processorId,&cpu_set))
        {  
            break;
        }
    }
    return processorId;

}

int getLocalRank() {
	int rank;
	MPI_safecall(MPI_Comm_rank ( single_node_comm, &rank));
	
	return rank;
}

int getNumberOfRanksOnNode()
{
	int size;
	MPI_safecall(MPI_Comm_size ( single_node_comm, &size));
	
	return size;

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

	IF_DEBUG(1) printf("PE%d hat in single_node_comm den rank %d\n", me, me_node);

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

LCRP_TYPE* setup_communication(CR_TYPE* cr, int work_dist, int options)
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

	IF_DEBUG(1){
		MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));
		if (me==0) printf("Entering setup_communication\n");
	}

	lcrp = (LCRP_TYPE*) allocateMemory( sizeof(LCRP_TYPE), "lcrp");

	lcrp->fullRowPerm = NULL;
	lcrp->fullInvRowPerm = NULL;
	lcrp->splitRowPerm = NULL;
	lcrp->splitInvRowPerm = NULL;
	lcrp->fullMatrix = NULL;
	lcrp->localMatrix = NULL;
	lcrp->remoteMatrix = NULL;

	lcrp->nEnts = cr->nEnts;
	lcrp->nRows = cr->nRows;
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

		if (work_dist == WORKDIST_EQUAL_NZE){
			IF_DEBUG(1) printf("Distribute Matrix with EQUAL_NZE on each PE\n");
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
		else if (work_dist == WORKDIST_EQUAL_LNZE){
			IF_DEBUG(1) printf("Distribute Matrix with EQUAL_LNZE on each PE\n");

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
			IF_DEBUG(2) printf("erster Durchgang: lokale Elemente:\n");
			hlpi = 0;
			for (i=0; i<lcrp->nodes; i++){
				hlpi += loc_count[i];
				IF_DEBUG(1) printf("Block %3d %8d %12d\n", i, loc_count[i], lcrp->lnEnts[i]);
			}
			target_lnze = hlpi/lcrp->nodes;
			IF_DEBUG(1) printf("insgesamt lokale Elemente: %d bzw pro PE: %d\n", hlpi, target_lnze);

			outer_convergence = 0; 
			outer_iter = 0;

			while(outer_convergence==0){ 

				IF_DEBUG(1) printf("Convergence Iteration %d\n", outer_iter);

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
					if (lcrp->lfRow[i+1]>cr->nRows) printf("Exceeded matrix dimension\n");
					lcrp->lfEnt[i+1] = cr->rowOffset[lcrp->lfRow[i+1]];
					lcrp->lnEnts[i] = lcrp->lfEnt[i+1] - lcrp->lfEnt[i] ;

				}

				lcrp->lnRows[lcrp->nodes-1] = cr->nRows - lcrp->lfRow[lcrp->nodes-1];
				lcrp->lnEnts[lcrp->nodes-1] = cr->nEnts - lcrp->lfEnt[lcrp->nodes-1];
				loc_count[lcrp->nodes-1] = 0;
				for (j=lcrp->lfEnt[lcrp->nodes-1]; j<cr->nEnts; j++)
					if (cr->col[j] >= lcrp->lfRow[lcrp->nodes-1]) loc_count[lcrp->nodes-1]++;

				IF_DEBUG(2) printf("naechster Durchgang: outer_iter=%d:\n", outer_iter);
				hlpi = 0;
				for (i=0; i<lcrp->nodes; i++){
					hlpi += loc_count[i];
					IF_DEBUG(2) printf("Block %3d %8d %12d\n", i, loc_count[i], lcrp->lnEnts[i]);
				}
				target_lnze = hlpi/lcrp->nodes;
				IF_DEBUG(2) printf("insgesamt lokale Elemente: %d bzw pro PE: %d Gesamtanteil: %6.3f%%\n",
						hlpi, target_lnze, 100.0*hlpi/(1.0*cr->nEnts));

				hlpi = 0;
				for (i=0; i<lcrp->nodes; i++) if ( (1.0*(loc_count[i]-target_lnze))/(1.0*target_lnze)>0.001) hlpi++;
				if (hlpi == 0) outer_convergence = 1;

				outer_iter++;

				if (outer_iter>20){
					printf("Zwar noch keine Konvergenz aber schon 20 Iterationen\n");
					printf("Verliere die Geduld und setze outer_convergence auf 1\n");
					outer_convergence = 1;
				}

			}

			IF_DEBUG(1) {
				for (i=0; i<lcrp->nodes; i++)  
				printf("PE%3d: lfRow=%8d lfEnt=%12d lnRows=%8d lnEnts=%12d\n", i, lcrp->lfRow[i], 
						lcrp->lfEnt[i], lcrp->lnRows[i], lcrp->lnEnts[i]);
			}


		}
		else {

			IF_DEBUG(1) printf("Distribute Matrix with EQUAL_ROWS on each PE\n");
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

	size_val  = (size_t)( (size_t)(lcrp->lnEnts[me])   * sizeof( data_t ) );
	size_col  = (size_t)( (size_t)(lcrp->lnEnts[me])   * sizeof( int ) );
	size_ptr  = (size_t)( (size_t)(lcrp->lnRows[me]+1) * sizeof( int ) );

	lcrp->val      = (data_t*)    allocateMemory( size_val,  "lcrp->val" ); 
	lcrp->col      = (int*)       allocateMemory( size_col,  "lcrp->col" ); 
	lcrp->lrow_ptr = (int*)       allocateMemory( size_ptr,  "lcrp->lrow_ptr" ); 

	/****************************************************************************
	 *******   Fill all fields with their corresponding values            *******
	 ***************************************************************************/


	//#pragma omp parallel for schedule(runtime)
#pragma omp parallel for schedule(static)
	for (i=0; i<lcrp->lnEnts[me]; i++) lcrp->val[i] = 0.0;

	//#pragma omp parallel for schedule(runtime)
#pragma omp parallel for schedule(static)
	for (i=0; i<lcrp->lnEnts[me]; i++) lcrp->col[i] = 0.0;

	//#pragma omp parallel for schedule(runtime)
#pragma omp parallel for schedule(static)
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
			if (lcrp->lfRow[j]<lcrp->col[i]) {
				/* Entsprechendes Paarelement liegt auf PE j */
				comm_remotePE[i] = j;
				wishlist_counts[j]++;
				/* Uebergang von FORTRAN-NUMERIERUNG AUF C-STYLE !!!!!!!!!!!!!!!! */
				comm_remoteEl[i] = lcrp->col[i]-1 -lcrp->lfRow[j];
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
			lcrp->col[i] = pseudocol[revcol[lcrp->col[i]-1]];
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
	if (!(options & SPMVM_OPTION_NO_TASKMODE_KERNEL)) { // split computation


		pseudo_ldim = lcrp->lnRows[me]+lcrp->halo_elements ;

		lnEnts_l=0;
		for (i=0; i<lcrp->lnEnts[me];i++)
			if (lcrp->col[i]<lcrp->lnRows[me]) lnEnts_l++;


		lnEnts_r = lcrp->lnEnts[me]-lnEnts_l;

		IF_DEBUG(1) printf("PE%d: Rows=%6d\t Ents=%6d(l),%6d(r),%6d(g)\t pdim=%6d\n", 
				me, lcrp->lnRows[me], lnEnts_l, lnEnts_r, lcrp->lnEnts[me], pseudo_ldim );

		size_lval = (size_t)( lnEnts_l             * sizeof(data_t) ); 
		size_rval = (size_t)( lnEnts_r             * sizeof(data_t) ); 
		size_lcol = (size_t)( lnEnts_l             * sizeof(int) ); 
		size_rcol = (size_t)( lnEnts_r             * sizeof(int) ); 
		size_lptr = (size_t)( (lcrp->lnRows[me]+1) * sizeof(int) ); 
		size_rptr = (size_t)( (lcrp->lnRows[me]+1) * sizeof(int) ); 


		lcrp->lrow_ptr_l = (int*)    allocateMemory( size_lptr, "lcrp->lrow_ptr_l" ); 
		lcrp->lrow_ptr_r = (int*)    allocateMemory( size_rptr, "lcrp->lrow_ptr_r" ); 
		lcrp->lcol       = (int*)    allocateMemory( size_lcol, "lcrp->lcol" ); 
		lcrp->rcol       = (int*)    allocateMemory( size_rcol, "lcrp->rcol" ); 
		lcrp->lval       = (data_t*) allocateMemory( size_lval, "lcrp->lval" ); 
		lcrp->rval       = (data_t*) allocateMemory( size_rval, "lcrp->rval" ); 

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
		IF_DEBUG(1) printf("PE%d: lnRows=%d row_ptr=%d..%d\n", 
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
				printf("--Row_ptrs-- PE %d: i=%d local=%d remote=%d\n", 
						me, i, lcrp->lrow_ptr_l[i], lcrp->lrow_ptr_r[i]);
			for (i=0; i<lcrp->lrow_ptr_l[lcrp->lnRows[me]]; i++)
				printf("-- local -- PE%d: lcrp->lcol[%d]=%d\n", me, i, lcrp->lcol[i]);
			for (i=0; i<lcrp->lrow_ptr_r[lcrp->lnRows[me]]; i++)
				printf("-- remote -- PE%d: lcrp->rcol[%d]=%d\n", me, i, lcrp->rcol[i]);
		}
		fflush(stdout);
		MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));

	} else{
		lcrp->lrow_ptr_l = (int*)    allocateMemory( sizeof(int), "lcrp->lrow_ptr_l" ); 
		lcrp->lrow_ptr_r = (int*)    allocateMemory( sizeof(int), "lcrp->lrow_ptr_r" ); 
		lcrp->lcol       = (int*)    allocateMemory( sizeof(int), "lcrp->lcol" ); 
		lcrp->rcol       = (int*)    allocateMemory( sizeof(int), "lcrp->rcol" ); 
		lcrp->lval       = (data_t*) allocateMemory( sizeof(data_t), "lcrp->lval" ); 
		lcrp->rval       = (data_t*) allocateMemory( sizeof(data_t), "lcrp->rval" ); 
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

LCRP_TYPE* setup_communication_parallel(CR_TYPE* cr, char *matrixPath, int work_dist, int options)
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
	MPI_Status *status=NULL;

	/****************************************************************************
	 *******            ........ Executable statements ........           *******
	 ***************************************************************************/

	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &me);

	IF_DEBUG(1){
		ierr = MPI_Barrier(MPI_COMM_WORLD);
		if (me==0) printf("Entering setup_communication_parallel\n");
	}

	lcrp = (LCRP_TYPE*) allocateMemory( sizeof(LCRP_TYPE), "lcrp");

	lcrp->nEnts = cr->nEnts;
	lcrp->nRows = cr->nRows;
#pragma omp parallel
	lcrp->threads = omp_get_num_threads(); 

	ierr = MPI_Comm_size(MPI_COMM_WORLD, &(lcrp->nodes));

	size_nint = (size_t)( (size_t)(lcrp->nodes)   * sizeof(int)  );
	size_nptr = (size_t)( lcrp->nodes             * sizeof(int*) );
	size_a2ai = (size_t)( lcrp->nodes*lcrp->nodes * sizeof(int)  );

	IF_DEBUG(1) {
		printf("PE%i: size_nint=%lu\n",me,size_nint);
		printf("PE%i: size_nptr=%lu\n",me,size_nptr);
		printf("PE%i: size_a2ai=%lu\n",me,size_a2ai);
	}

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

		if (work_dist == WORKDIST_EQUAL_NZE){
			IF_DEBUG(1) printf("Distribute Matrix with EQUAL_NZE on each PE\n");
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

			IF_DEBUG(1) printf("Distribute Matrix with EQUAL_ROWS on each PE\n");
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

	ierr = MPI_Bcast(&(lcrp->nEnts),  1, MPI_INTEGER, 0, MPI_COMM_WORLD);
	ierr = MPI_Bcast(&(lcrp->nRows),  1, MPI_INTEGER, 0, MPI_COMM_WORLD);
	ierr = MPI_Bcast(lcrp->lfRow,  lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD);
	ierr = MPI_Bcast(lcrp->lfEnt,  lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD);
	ierr = MPI_Bcast(lcrp->lnRows, lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD);
	ierr = MPI_Bcast(lcrp->lnEnts, lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD);

	IF_DEBUG(1) {
		printf("PE%i: local rows          = %i\n",me,lcrp->lnRows[me]);
		printf("PE%i: local rows (offset) = %i\n",me,lcrp->lfRow[me]);
		printf("PE%i: local entries          = %i\n",me,lcrp->lnEnts[me]);
		printf("PE%i: local entires (offset) = %i\n",me,lcrp->lfEnt[me]);
		fflush(stdout);
	}

	/****************************************************************************
	 *******   Allocate memory for matrix in distributed CRS storage      *******
	 ***************************************************************************/

	size_val  = (size_t)( (size_t)(lcrp->lnEnts[me])   * sizeof( data_t ) );
	size_col  = (size_t)( (size_t)(lcrp->lnEnts[me])   * sizeof( int ) );
	size_ptr  = (size_t)( (size_t)(lcrp->lnRows[me]+1) * sizeof( int ) );

	lcrp->val      = (data_t*)    allocateMemory( size_val,  "lcrp->val" ); 
	lcrp->col      = (int*)       allocateMemory( size_col,  "lcrp->col" ); 
	lcrp->lrow_ptr = (int*)       allocateMemory( size_ptr,  "lcrp->lrow_ptr" ); 

	/****************************************************************************
	 *******   Fill all fields with their corresponding values            *******
	 ***************************************************************************/


	//#pragma omp parallel for schedule(runtime)
#pragma omp parallel for schedule(static)
	for (i=0; i<lcrp->lnEnts[me]; i++) lcrp->val[i] = 0.0;

	//#pragma omp parallel for schedule(runtime)
#pragma omp parallel for schedule(static)
	for (i=0; i<lcrp->lnEnts[me]; i++) lcrp->col[i] = 0.0;

	//#pragma omp parallel for schedule(runtime)
#pragma omp parallel for schedule(static)
	for (i=0; i<lcrp->lnRows[me]; i++) lcrp->lrow_ptr[i] = 0.0;

	/* replace scattering with read-in */
	/*
	   ierr = MPI_Scatterv ( cr->val, lcrp->lnEnts, lcrp->lfEnt, MPI_DOUBLE, 
	   lcrp->val, lcrp->lnEnts[me],  MPI_DOUBLE, 0, MPI_COMM_WORLD);

	   ierr = MPI_Scatterv ( cr->col, lcrp->lnEnts, lcrp->lfEnt, MPI_INTEGER,
	   lcrp->col, lcrp->lnEnts[me],  MPI_INTEGER, 0, MPI_COMM_WORLD);
	 */

	IF_DEBUG(1) printf("PE%i: opening file %s for parallel read-in\n",me,matrixPath);
	ierr = MPI_File_open(MPI_COMM_WORLD, matrixPath, MPI_MODE_RDONLY, info, &file_handle);

	if( ierr ) {
		printf("PE%i: unable to open parallel file %s \n",me, matrixPath);
		exit(1);
	}

	/* read col */
	offset_in_file = (4+lcrp->nRows+1)*sizeof(int) + (lcrp->lfEnt[me])*sizeof(int);
	IF_DEBUG(1) printf("PE%i: read col -- offset=%i | %d\n",me,(int)offset_in_file,lcrp->lfEnt[me]);
	ierr = MPI_File_seek(file_handle, offset_in_file, MPI_SEEK_SET);
	ierr = MPI_File_read(file_handle, lcrp->col, lcrp->lnEnts[me], MPI_INTEGER, status);

	/* read val */
	offset_in_file = (4+lcrp->nRows+1)*sizeof(int) + (lcrp->nEnts)*sizeof(int) + (lcrp->lfEnt[me])*sizeof(data_t);
	IF_DEBUG(1) printf("PE%i: read val -- offset=%i\n",me,(int)offset_in_file);
	ierr = MPI_File_seek(file_handle, offset_in_file, MPI_SEEK_SET);
	ierr = MPI_File_read(file_handle, lcrp->val, lcrp->lnEnts[me], MPI_MYDATATYPE, status);

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


	for (i=0; i<lcrp->lnEnts[me]; i++) lcrp->col[i]++; // setup has to be done with fortran numbering
	for (i=0; i<lcrp->nodes; i++) wishlist_counts[i] = 0;

	/* Transform global column index into 2d-local/non-local index */
	for (i=0;i<lcrp->lnEnts[me];i++){
		for (j=lcrp->nodes-1;j>-1; j--){
			if (lcrp->lfRow[j]<lcrp->col[i]) {
				/* Entsprechendes Paarelement liegt auf PE j */
				comm_remotePE[i] = j;
				wishlist_counts[j]++;
				/* Uebergang von FORTRAN-NUMERIERUNG AUF C-STYLE !!!!!!!!!!!!!!!! */
				comm_remoteEl[i] = lcrp->col[i]-1 -lcrp->lfRow[j];
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
		else                        lcrp->col[i] = pseudocol[revcol[lcrp->col[i]-1]];
	} /* !!!!!!!! Eintraege in wishlist gehen entsprechend Input-file von 1-9! */

	freeMemory ( size_col,  "comm_remoteEl",  comm_remoteEl);
	freeMemory ( size_col,  "comm_remotePE",  comm_remotePE);
	freeMemory ( size_col,  "pseudocol",      pseudocol);
	freeMemory ( size_col,  "globcol",        globcol);
	freeMemory ( size_revc, "revcol",         revcol);
	freeMemory ( size_pval, "present_values", present_values ); 

	/* sollte hier auch nicht Not tun
#ifdef CMEM
sweepMemory(GLOBAL);
#endif
	 */

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
	if (!(options & SPMVM_OPTION_NO_TASKMODE_KERNEL)) { // split computation

		pseudo_ldim = lcrp->lnRows[me]+lcrp->halo_elements ;

		lnEnts_l=0;
		for (i=0; i<lcrp->lnEnts[me];i++)
			if (lcrp->col[i]<lcrp->lnRows[me]) lnEnts_l++;


		lnEnts_r = lcrp->lnEnts[me]-lnEnts_l;

		IF_DEBUG(1) printf("PE%d: Rows=%6d\t Ents=%6d(l),%6d(r),%6d(g)\t pdim=%6d\n", 
				me, lcrp->lnRows[me], lnEnts_l, lnEnts_r, lcrp->lnEnts[me], pseudo_ldim );

		size_lval = (size_t)( lnEnts_l             * sizeof(data_t) ); 
		size_rval = (size_t)( lnEnts_r             * sizeof(data_t) ); 
		size_lcol = (size_t)( lnEnts_l             * sizeof(int) ); 
		size_rcol = (size_t)( lnEnts_r             * sizeof(int) ); 
		size_lptr = (size_t)( (lcrp->lnRows[me]+1) * sizeof(int) ); 
		size_rptr = (size_t)( (lcrp->lnRows[me]+1) * sizeof(int) ); 


		lcrp->lrow_ptr_l = (int*)    allocateMemory( size_lptr, "lcrp->lrow_ptr_l" ); 
		lcrp->lrow_ptr_r = (int*)    allocateMemory( size_rptr, "lcrp->lrow_ptr_r" ); 
		lcrp->lcol       = (int*)    allocateMemory( size_lcol, "lcrp->lcol" ); 
		lcrp->rcol       = (int*)    allocateMemory( size_rcol, "lcrp->rcol" ); 
		lcrp->lval       = (data_t*) allocateMemory( size_lval, "lcrp->lval" ); 
		lcrp->rval       = (data_t*) allocateMemory( size_rval, "lcrp->rval" ); 

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
		IF_DEBUG(1) printf("PE%d: lnRows=%d row_ptr=%d..%d\n", 
				me, lcrp->lnRows[me], lcrp->lrow_ptr[0], lcrp->lrow_ptr[lcrp->lnRows[me]]);
		fflush(stdout);
		ierr = MPI_Barrier(MPI_COMM_WORLD);

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
				printf("--Row_ptrs-- PE %d: i=%d local=%d remote=%d\n", 
						me, i, lcrp->lrow_ptr_l[i], lcrp->lrow_ptr_r[i]);
			for (i=0; i<lcrp->lrow_ptr_l[lcrp->lnRows[me]]; i++)
				printf("-- local -- PE%d: lcrp->lcol[%d]=%d\n", me, i, lcrp->lcol[i]);
			for (i=0; i<lcrp->lrow_ptr_r[lcrp->lnRows[me]]; i++)
				printf("-- remote -- PE%d: lcrp->rcol[%d]=%d\n", me, i, lcrp->rcol[i]);
		}
		fflush(stdout);
		ierr = MPI_Barrier(MPI_COMM_WORLD);

	}
	else{
		lcrp->lrow_ptr_l = (int*)    allocateMemory( sizeof(int), "lcrp->lrow_ptr_l" ); 
		lcrp->lrow_ptr_r = (int*)    allocateMemory( sizeof(int), "lcrp->lrow_ptr_r" ); 
		lcrp->lcol       = (int*)    allocateMemory( sizeof(int), "lcrp->lcol" ); 
		lcrp->rcol       = (int*)    allocateMemory( sizeof(int), "lcrp->rcol" ); 
		lcrp->lval       = (data_t*) allocateMemory( sizeof(data_t), "lcrp->lval" ); 
		lcrp->rval       = (data_t*) allocateMemory( sizeof(data_t), "lcrp->rval" ); 
	}

	freeMemory ( size_mem,  "wishlist_mem",    wishlist_mem);
	freeMemory ( size_mem,  "cwishlist_mem",   cwishlist_mem);
	freeMemory ( size_nptr, "wishlist",        wishlist);
	freeMemory ( size_nptr, "cwishlist",       cwishlist);
	freeMemory ( size_a2ai, "tmp_transfers",   tmp_transfers);
	freeMemory ( size_nint, "wishlist_counts", wishlist_counts);
	freeMemory ( size_nint, "item_from",       item_from);


		return lcrp;
}

static int stringcmp(const void *x, const void *y)
{
	return (strcmp((char *)x, (char *)y));
}


int getNumberOfNodes() 
{
	int nameLen,me,size,i,distinctNames = 1;
	char name[MPI_MAX_PROCESSOR_NAME];
	char *names;

	MPI_safecall(MPI_Comm_rank(MPI_COMM_WORLD,&me));
	MPI_safecall(MPI_Comm_size(MPI_COMM_WORLD,&size));
	MPI_safecall(MPI_Get_processor_name(name,&nameLen));


	if (me==0) {
		names = (char *)allocateMemory(size*MPI_MAX_PROCESSOR_NAME*sizeof(char),
				"names");
	}

	
	MPI_safecall(MPI_Gather(name,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,names,
			MPI_MAX_PROCESSOR_NAME,MPI_CHAR,0,MPI_COMM_WORLD));

	if (me==0) {
		qsort(names,size,MPI_MAX_PROCESSOR_NAME*sizeof(char),stringcmp);
		for (i=1; i<size; i++) {
			if (strcmp(names+(i-1)*MPI_MAX_PROCESSOR_NAME,names+
						i*MPI_MAX_PROCESSOR_NAME)) {
					distinctNames++;
			}
		}
		free(names);
	}

	MPI_safecall(MPI_Bcast(&distinctNames,1,MPI_INT,0,MPI_COMM_WORLD));
	
	return distinctNames;
}

