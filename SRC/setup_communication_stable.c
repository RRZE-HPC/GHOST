#include <matricks.h>

#include <mpi.h>

/******************************************************************************
 * Routine zum einmaligen Berechnen des Kommunikationsmusters fuer
 * den Fall einer seriell eingelesenen Matrix. Alle Matrixdaten liegen
 * auf PE 0 vor, dieser kannn alle Berechnungen durchfuehren und die
 * entsprechenden Daten dann an diejenigen PEs verteilen die es betrifft.
 *****************************************************************************/

LCRP_TYPE* setup_communication(CR_TYPE* cr, int work_dist){

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

   int anzproc;
   int nEnts_glob;

   int *first_line, *nRows_loc;
   int *node_nnzptr, *nEnts_loc;
   int *wishlist_counts;
   int *wishlist_mem;
   int **wishlist;
   int *requested_from;
   int *different_wishes;
   int *cwishlist_mem, **cwishlist;

   int this_pseudo_col, num_pseudo_col; 
   int *pseudocol;
   int *globcol;
   int *reverse_col;

   int *comm_remotePE, *comm_remoteEl;

   int balancing; 
   int target_rows;

   int found_agreement;

   int lnEnts_l, lnEnts_r;
   int current_l, current_r;
   int loc_count;

   int pseudo_ldim;

   size_t size_nint, size_col, size_val, size_ptr, size_lcrp;
   size_t size_lval, size_rval, size_lcol, size_rcol, size_lptr, size_rptr;
   size_t size_revc, size_a2ai, size_cmem, size_np, size_pval;  
   size_t size_mem, size_remd, size_remw;

   int n_nodes;

   MPI_Status *status;

   /****************************************************************************
    *******            ........ Executable statements ........           *******
    ***************************************************************************/
   status  = (MPI_Status*)  allocateMemory( sizeof(MPI_Status) ,  "send_status" );

   ierr = MPI_Comm_rank(MPI_COMM_WORLD, &me);
   ierr = MPI_Comm_size(MPI_COMM_WORLD, &anzproc);

   n_nodes = anzproc;

   IF_DEBUG(1){
      ierr = MPI_Barrier(MPI_COMM_WORLD);
      if (me==0) printf("Entering setup_communication\n");
   }

   /* Temporaere Variablen die auf allen PEs vorgehalten werden */
   size_nint = (size_t)( (size_t)(n_nodes) * sizeof(int) );

   first_line     = (int*)       allocateMemory( size_nint, "first_line" );
   node_nnzptr    = (int*)       allocateMemory( size_nint, "node_nnzptr" );
   nRows_loc      = (int*)       allocateMemory( size_nint, "nRows_loc" );
   nEnts_loc      = (int*)       allocateMemory( size_nint, "nEnts_loc" );

   /* Berechne die jeder PE zustehenden nzes und rows auf der master-PE... */
   if (me==0){


      if (work_dist == EQUAL_NZE){
	 target_nnz = (cr->nEnts/n_nodes)+1; /* sonst bleiben welche uebrig! */

	 IF_DEBUG(1){
            printf("Distribute Matrix with EQUAL_NZE on each PE\n");
	    printf("cr->nCols %d\n", cr->nCols);
	    printf("nodes     %d\n", n_nodes);
	    printf("angestrebte Anzahl nnz pro node: %i\n", target_nnz);
	    fflush(stdout);
	 }

	 first_line[0] = 0;
	 node_nnzptr[0] = 0;

	 j = 1;
	 for (i=0;i<cr->nRows;i++){
	    IF_DEBUG(2) printf("SetupComm: PE==%d: Row_ptr[%d]=%d\n", me, i, cr->rowOffset[i]);
	    if (cr->rowOffset[i] >= j*target_nnz){
	       IF_DEBUG(1) printf("SetupComm: PE==%d: i=%d; Row_ptr[i]:%i target %i \n",
		     me, i, cr->rowOffset[i], j*target_nnz);
	       first_line[j] = i;
	       node_nnzptr[j] = cr->rowOffset[i];
	       j = j+1;
	    }
	 }
      }
      else if (work_dist == EQUAL_LNZE){
	 printf("Distribute Matrix with EQUAL_LNZE on each PE\n");
      }
      else {

	 IF_DEBUG(1) printf("Distribute Matrix with EQUAL_ROWS on each PE\n");
	 target_rows = (cr->nRows/n_nodes);

	 first_line[0] = 0;
	 node_nnzptr[0] = 0;

	 for (i=1; i<n_nodes; i++){
	    first_line[i] = first_line[i-1]+target_rows;
	    node_nnzptr[i] = cr->rowOffset[first_line[i]];
	 }
      }
      

      for (i=0; i<n_nodes-1; i++){
	 nRows_loc[i] = first_line[i+1] - first_line[i] ;
	 nEnts_loc[i] = node_nnzptr[i+1] - node_nnzptr[i] ;
      }

      nRows_loc[n_nodes-1] = cr->nRows - first_line[n_nodes-1] ;
      nEnts_loc[n_nodes-1] = cr->nEnts - node_nnzptr[n_nodes-1];
   }

   /****************************************************************************
    *******      Correct share for each PE has been calculated           *******
    ***************************************************************************/

   IF_DEBUG(1){
      ierr = MPI_Barrier(MPI_COMM_WORLD);
      if (me==0) printf("Vor Bcast\n");
   }

   /* ... und verteile diese Information an alle */
   ierr = MPI_Bcast(first_line,  n_nodes, MPI_INTEGER, 0, MPI_COMM_WORLD);
   ierr = MPI_Bcast(node_nnzptr, n_nodes, MPI_INTEGER, 0, MPI_COMM_WORLD);
   ierr = MPI_Bcast(nRows_loc,   n_nodes, MPI_INTEGER, 0, MPI_COMM_WORLD);
   ierr = MPI_Bcast(nEnts_loc,   n_nodes, MPI_INTEGER, 0, MPI_COMM_WORLD);


   IF_DEBUG(1){ 
      for (i=0; i<n_nodes; i++){
	 printf("SetupComm: PE %d: node %d : bekommt %d Zeilen (beginnend bei %d) und %d Elemente (beginnend bei %d)\n",
	       me, i, nRows_loc[i], first_line[i], nEnts_loc[i], node_nnzptr[i]);
      }
   }
   ierr = MPI_Barrier(MPI_COMM_WORLD);


   /* Entsprechnung lokaler <-> globaler Indizes im nze-array auf allen PEs*/
   IF_DEBUG(1){
      printf("PE %d: nz-elements (local): %7d -- %7d <-> (global): %7d -- %7d\n", 
	    me, 0, nEnts_loc[me]-1, node_nnzptr[me], node_nnzptr[me]+nEnts_loc[me]-1 );
      printf("PE %d: rows        (local): %7d -- %7d <-> (global): %7d -- %7d\n",
	    me, 0, nRows_loc[me]-1, first_line[me], first_line[me]+nRows_loc[me]-1 );
   }

   ierr = MPI_Barrier(MPI_COMM_WORLD); IF_DEBUG(1) if (me==0) printf("allocating lcrp\n");

   size_lcrp = sizeof(LCRP_TYPE);
   size_val  = (size_t)( (size_t)(nEnts_loc[me])   * sizeof( double ) );
   size_col  = (size_t)( (size_t)(nEnts_loc[me])   * sizeof( int ) );
   size_ptr  = (size_t)( (size_t)(nRows_loc[me]+1) * sizeof( int ) );

   IF_DEBUG(1) printf("PE%d: sizes: val=%lu, col=%lu, ptr=%lu\n", me, size_val, size_col, size_ptr);

   lcrp           = (LCRP_TYPE*) allocateMemory( size_lcrp, "lcrp");
   lcrp->lnEnts   = (int*)       allocateMemory( size_nint, "lcrp->lnEnts" ); 
   lcrp->lnRows   = (int*)       allocateMemory( size_nint, "lcrp->lnRows" ); 
   lcrp->val      = (double*)    allocateMemory( size_val,  "lcrp->val" ); 
   lcrp->col      = (int*)       allocateMemory( size_col,  "lcrp->col" ); 
   lcrp->lrow_ptr = (int*)       allocateMemory( size_ptr,  "lcrp->lrow_ptr" ); 
   lcrp->wishes   = (int*)       allocateMemory( size_nint, "lcrp->wishes" ); 
   lcrp->dues     = (int*)       allocateMemory( size_nint, "lcrp->dues" ); 

   lcrp->lfEnt    = (int*)       allocateMemory( size_nint, "lcrp->lfEnt" ); 
   lcrp->lfRow    = (int*)       allocateMemory( size_nint, "lcrp->lfRow" ); 
   //lcrp->rem_PE   = (int*)       allocateMemory( size_col,  "lcrp->rem_PE" ); 
   //lcrp->rem_El   = (int*)       allocateMemory( size_col,  "lcrp->rem_El" ); 

   ierr = MPI_Barrier(MPI_COMM_WORLD); IF_DEBUG(1) if (me==0) printf("lcrp allocated\n");

   lcrp->nodes = n_nodes;

   for (i=0;i<n_nodes;i++){
      lcrp->lnEnts[i] = nEnts_loc[i];
      lcrp->lnRows[i] = nRows_loc[i];
      lcrp->lfEnt[i]  = node_nnzptr[i];
      lcrp->lfRow[i]  = first_line[i];
   }

   requested_from  = (int*)       allocateMemory( size_nint, "requested_from" ); 

#ifdef PLACE
#pragma omp parallel for schedule(runtime)
#endif
   for (i=0; i<nEnts_loc[me]; i++) lcrp->val[i] = 0.0;

#ifdef PLACE
#pragma omp parallel for schedule(runtime)
#endif
   for (i=0; i<nEnts_loc[me]; i++) lcrp->col[i] = 0.0;


   ierr = MPI_Barrier(MPI_COMM_WORLD); IF_DEBUG(1) if (me==0) printf("Placement val & col done\n");


   /*   ierr = MPI_Barrier(MPI_COMM_WORLD); if (me==0) printf("vor haendischemScatterv\n");

	if (me==0){ 
	printf("%g\n", cr->val[0]);
	for(i=1; i<n_nodes; i++)
	ierr = MPI_Send( &(cr->val[node_nnzptr[i]]), nEnts_loc[i], MPI_DOUBLE, i, i, MPI_COMM_WORLD);
	printf("Send fertig %d\n", i); fflush(stdout);
	}
	else{
	ierr = MPI_Recv(lcrp->val, nEnts_loc[me], MPI_DOUBLE, 0, me, MPI_COMM_WORLD, status);
	printf("PE%d: Recv abgeschlossen\n", me);fflush(stdout);
	}
	*/

   ierr = MPI_Barrier(MPI_COMM_WORLD); IF_DEBUG(1)if (me==0) printf("vor Scatterv\n");

   /* Verteile die entsprechenden arrays fuer die Nichtnullelemente auf die PEs */
   ierr = MPI_Scatterv ( cr->val, nEnts_loc, node_nnzptr, MPI_DOUBLE, 
	 lcrp->val, nEnts_loc[me],  MPI_DOUBLE, 0, MPI_COMM_WORLD);

   ierr = MPI_Scatterv ( cr->col, nEnts_loc, node_nnzptr, MPI_INTEGER,
	 lcrp->col, nEnts_loc[me],  MPI_INTEGER, 0, MPI_COMM_WORLD);

   ierr = MPI_Scatterv ( cr->rowOffset, nRows_loc, first_line, MPI_INTEGER,
	 lcrp->lrow_ptr, nRows_loc[me],  MPI_INTEGER, 0, MPI_COMM_WORLD);

   IF_DEBUG(1) if (me==0) printf("main-arrays verteit\n");


   /* das bis hierher kann ich auch parallel haben */


   IF_DEBUG(2){
      for (i=0;i<nRows_loc[me];i++)
	 printf("PE %d: globaler Row_ptr(%d)=%d -> lokal: %d\n", 
	       me, i, lcrp->lrow_ptr[i], lcrp->lrow_ptr[i]-node_nnzptr[me]);
   }

   /* Relative offset with respect to row_ptr of first local row */
   for (i=0;i<nRows_loc[me]+1;i++)
      lcrp->lrow_ptr[i] =  lcrp->lrow_ptr[i] - node_nnzptr[me]; 

   /* last entry of row_ptr holds the local number of entries */
   lcrp->lrow_ptr[nRows_loc[me]] = lcrp->lnEnts[me]; 


   comm_remotePE      = (int*)          allocateMemory( size_col, "comm_remotePE" );
   comm_remoteEl      = (int*)          allocateMemory( size_col, "comm_remoteEl" );

   /* Berechne aus den globalen column-Indizes die (zweidimensionalen lokalen-/nichtlokalen Indizes) */
   for (i=0;i<nEnts_loc[me];i++){
      for (j=n_nodes-1;j>-1; j--){
	 if (first_line[j]<lcrp->col[i]) {
	    /* Entsprechendes Paarelement liegt auf PE j */
	    comm_remotePE[i] = j;
	    /* Uebergang von FORTRAN-NUMERIERUNG AUF C-STYLE !!!!!!!!!!!!!!!!!! */
	    comm_remoteEl[i] = lcrp->col[i]-1 -first_line[j];
	    break;
	 }
      }
   }

   IF_DEBUG(2){
      for (i=0;i<nEnts_loc[me];i++)
	 printf(" zzz PE %d: i=%d, lcrp->col[i]=%d, lcrp->val[i] = %f -- zugeh. Element auf PE %d an Pos. %d\n",
	       me, i, lcrp->col[i], lcrp->val[i], comm_remotePE[i], comm_remoteEl[i]);

      printf(" zzz PE %d: Lokale Zeilen/Spalten: %d -- %d\n", me, first_line[me], first_line[me]+nRows_loc[me]);
   }


   ierr = MPI_Barrier(MPI_COMM_WORLD); IF_DEBUG(1) if (me==0) printf("setting up wishlists\n");


   wishlist_counts  = (int*)       allocateMemory( size_nint,             "wishlist_counts" ); 

   for (i=0; i<n_nodes; i++) wishlist_counts[i] = 0;

   for (i=0;i<nEnts_loc[me];i++)
      wishlist_counts[comm_remotePE[i]]++;

   acc_wishes = 0;
   for (i=0; i<n_nodes; i++) acc_wishes += wishlist_counts[i];

   size_mem = (size_t)( acc_wishes * sizeof(int) );
   size_np  = (size_t)( n_nodes * sizeof(int*) );

   wishlist_mem  = (int*)       allocateMemory( size_mem, "wishlist_mem" ); 
   wishlist      = (int**)      allocateMemory( size_np,  "wishlist" ); 

   acc_wishes = 0;
   for (i=0; i<n_nodes; i++){
      wishlist[i] = &wishlist_mem[acc_wishes];
      IF_DEBUG(2) printf("zzzz PE %d: Eintraege in der wishlist an PE %d: %d\n", me, i, wishlist_counts[i]);
      acc_wishes += wishlist_counts[i];
   }


   //pseudoindex = lnRows[me];
   for (i=0;i<n_nodes;i++) requested_from[i] = 0;

   for (i=0;i<nEnts_loc[me];i++){
      IF_DEBUG(2) printf("zz PE %d: Teile %d: %d: %d %d\n", me, i, comm_remotePE[i] , 
	    requested_from[comm_remotePE[i]] , comm_remoteEl[i]);
      wishlist[comm_remotePE[i]][requested_from[comm_remotePE[i]]] = comm_remoteEl[i];
      requested_from[comm_remotePE[i]]++;
   }


   IF_DEBUG(2){
      for (i=0; i<n_nodes; i++){
	 for (j=0;j<wishlist_counts[i];j++){
	    printf("zz PE %d: wishlist von PE %d: %d -> %d\n", me, i, j, wishlist[i][j]);
	 }
      }
   }

   /* Komprimiere wishlist */
   different_wishes  = (int*)       allocateMemory( size_nint,             "different_wishes" ); 

   size_cmem = (size_t)( acc_wishes * sizeof(int) );

   cwishlist_mem  = (int*)       allocateMemory( size_cmem,             "cwishlist_mem" ); 
   cwishlist      = (int**)      allocateMemory( size_np,            "cwishlist" ); 

   ierr = MPI_Barrier(MPI_COMM_WORLD); IF_DEBUG(1) if (me==0) printf("next allocated\n");

   acc_wishes = 0;
   for (i=0; i<n_nodes; i++){
      cwishlist[i] = &cwishlist_mem[acc_wishes];
      acc_wishes += wishlist_counts[i];
   }


   ierr = MPI_Barrier(MPI_COMM_WORLD); IF_DEBUG(1) if (me==0) printf("wuehlen\n");


   /****************************************
    *
    * Zeitfresser
    * 
    *
    * *******************************************/

#ifdef TUNE_SETUP

   max_loc_elements = 0;
   for (i=0;i<n_nodes;i++)
      if (max_loc_elements<lcrp->lnRows[i]) max_loc_elements = lcrp->lnRows[i];

   IF_DEBUG(1)printf("Max_loc_elements: %d\n", max_loc_elements);

   size_pval = (size_t)( max_loc_elements * sizeof(int) );

   present_values  = (int*)       allocateMemory( size_pval,             "present_values" ); 

   for (i=0; i<n_nodes; i++){

      for (j=0; j<max_loc_elements; j++) present_values[j] = -1;

      IF_DEBUG(2) printf("SuC: PE %d: wishlist_counts[%d]=%d\n", me, i, wishlist_counts[i]);fflush(stdout);

      if ( (i!=me) && (wishlist_counts[i]>0) ){

	 thisentry = 0;
	 for (j=0; j<wishlist_counts[i]; j++){
	    //        printf("PE %d, i=%d, j=%d, wishlist-entry: %d\n", me, i, j, wishlist[i][j]);fflush(stdout);
	    if (present_values[wishlist[i][j]]<0){
	       /* new entry which has not been found before */     
	       present_values[wishlist[i][j]] = thisentry;
	       cwishlist[i][thisentry] = wishlist[i][j];
	       thisentry = thisentry + 1;
	    }
	    else{
	       /* this wishlist-value has already been found; only update of col required */ 
	       continue;
	    }              
	 }
	 different_wishes[i] = thisentry;
      }
      else{
	 different_wishes[i] = 0;
      }
      IF_DEBUG(2){printf("PE: %d: different_wishes von %d :%d\n", me, i, different_wishes[i]);fflush(stdout);}
   }
   /*
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Abort(MPI_COMM_WORLD,999);
      */

#else

#pragma omp parallel for schedule(runtime) private (j, found_agreement, k)
   for (i=0; i<n_nodes; i++){
      IF_DEBUG(2) printf("SuC: PE %d: wishlist_counts[%d]=%d\n", me, i, wishlist_counts[i]);fflush(stdout);
      if ( (i!=me) && (wishlist_counts[i]>0) ){
	 cwishlist[i][0] = wishlist[i][0];
	 different_wishes[i] = 1;
	 for (j=1;j<wishlist_counts[i];j++){
	    found_agreement = 0;
	    for (k=0; k<different_wishes[i]; k++){
	       if (wishlist[i][j] == cwishlist[i][k]){
		  found_agreement = 1;
		  break;
	       }
	    }
	    if (found_agreement == 0){ /* war wirklich noch nicht da */
	       cwishlist[i][different_wishes[i]] = wishlist[i][j];
	       different_wishes[i]++;
	    }
	 }
      }
      else{
	 different_wishes[i] = 0;
      }
   }
#endif

   /****************************************
    *
    * EndeEnde  Zeitfresser
    * 
    *
    * *******************************************/


   ierr = MPI_Barrier(MPI_COMM_WORLD); IF_DEBUG(1) if (me==0) printf("setting up cwishlists\n");

   for (i=0; i<n_nodes; i++){
      lcrp->wishes[i] = different_wishes[i];
      IF_DEBUG(2) printf("zzZ PE %d: Eintraege in der cwishlist an PE %d: %d\n", me, i, different_wishes[i]);
   }
   lcrp->wishes[me] = 0; /* kein lokaler Transfer */

   /* Verteile lcrp->wishes and die jeweiligen PEs um lcrp->dues zu erhalten */
   size_a2ai = (size_t)( n_nodes*n_nodes * sizeof(int) );

   tmp_transfers  = (int*)       allocateMemory( size_a2ai, "tmp_transfers" ); 
   //printf("PE%d: Anfangsadressen lcrp->wishes und tmp_transfers %p %p\n", 
   //me, &lcrp->wishes[0], &tmp_transfers[0]);
   /* Hier sagt valgrind etwas von source and destination overlap! liegt aber
    *  wohl an der Emulation eines distributed memories mit shared memory. vgl. 
    *  Anfangsadressen !*/
   ierr = MPI_Allgather ( lcrp->wishes, n_nodes, MPI_INTEGER, tmp_transfers, n_nodes, MPI_INTEGER, MPI_COMM_WORLD ) ;

   ierr = MPI_Barrier(MPI_COMM_WORLD); IF_DEBUG(1) if (me==0) printf("nach allgather\n");

   IF_DEBUG(2){
      if (me == 0) for(i=0;i<n_nodes*n_nodes;i++) printf("ME==0: tmp_tansfers[%d]=%d\n", i, tmp_transfers[i]);
   }

   for (i=0; i<n_nodes; i++){
      lcrp->dues[i] = tmp_transfers[i*n_nodes+me];
      IF_DEBUG(2) printf("ZzZ PE %d: lcrp->dues[%d]=%d\n", me, i, lcrp->dues[i]);
   }

   lcrp->dues[me] = 0; /* keine lokalen Transfers */

   IF_DEBUG(2){
      for (i=0; i<n_nodes; i++){
	 for (j=0;j<different_wishes[i];j++){
	    printf("zzZ PE %d: cwishlist von PE %d: %d -> %d\n", me, i, j, cwishlist[i][j]);
	 }
      }
   }

   nEnts_glob = node_nnzptr[anzproc-1]+lcrp->lnEnts[anzproc-1]; 

   size_revc = (size_t)( nEnts_glob * sizeof(int) );

   pseudocol      = (int*)          allocateMemory( size_col, "pseudocol" );
   globcol      = (int*)          allocateMemory( size_col, "origincol" );
   reverse_col      = (int*)          allocateMemory( size_revc, "reverse_col" );

   ierr = MPI_Barrier(MPI_COMM_WORLD); IF_DEBUG(1) if (me==0) printf("nach allokieren\n");

   /* Extrahiere aus cwishlist die komprimierten Pseudo-Indizes fuer den invec-Zugriff */
   this_pseudo_col = nRows_loc[me];
   num_pseudo_col = 0;
   for (i=0; i<n_nodes; i++){
      if (i != me){ /* natuerlich nur fuer remote-Elemente */
	 for (j=0;j<different_wishes[i];j++){
	    pseudocol[num_pseudo_col] = this_pseudo_col;  
	    globcol[num_pseudo_col]   = lcrp->lfRow[i]+cwishlist[i][j]; 
	    reverse_col[globcol[num_pseudo_col]] = num_pseudo_col;
	    IF_DEBUG(2) printf("zzZ PE %d: cwishlist von PE %d: Item %d -> loc_Element %d (= %d -- global)\n",
		  me, i, j, cwishlist[i][j], globcol[num_pseudo_col]);
	    num_pseudo_col++;
	    this_pseudo_col++;
	 }
      }
   }

   IF_DEBUG(2) printf("Anzahl Elemente im Halo fuer PE %d: %d\n", me, num_pseudo_col);

   lcrp->halo_elements = num_pseudo_col;

   IF_DEBUG(2){
      for (i=0;i<num_pseudo_col;i++) printf("zzZ PE %d: Pseudocols: %d %d %d \n", 
	    me, i, pseudocol[i], globcol[i]);
   }

   ierr = MPI_Barrier(MPI_COMM_WORLD);

   for (i=0;i<nEnts_loc[me];i++){

      if (comm_remotePE[i] == me){
	 IF_DEBUG(2) printf("ZZZZ PE %d: Belege_nun_if   %d orig: %d -> %d\n", 
	       me, i, lcrp->col[i]-1, comm_remoteEl[i]);
	 fflush(stdout);
	 lcrp->col[i] = comm_remoteEl[i];
      }
      else{
	 IF_DEBUG(2) printf("ZZZZ PE %d: Belege_nun_else %d orig: %d -> %d\n", 
	       me, i, lcrp->col[i]-1, pseudocol[reverse_col[lcrp->col[i]-1]]);
	 fflush(stdout);
	 lcrp->col[i] = pseudocol[reverse_col[lcrp->col[i]-1]];
      }
      /*
	 lcrp->rem_PE[i]  = comm_remotePE[i];
	 lcrp->rem_El[i]  = comm_remoteEl[i];
	 */
   }


   ierr = MPI_Barrier(MPI_COMM_WORLD); IF_DEBUG(1) if (me==0) printf("comm_remote\n");

   IF_DEBUG(2) for (i=0;i<nEnts_loc[me];i++) printf("zZZZ PE %d: i=%d -> col=%d\n", me, i, lcrp->col[i]);
   ierr = MPI_Barrier(MPI_COMM_WORLD);

   /* !!!!!!!!!!!!!! Eintraege in wishlist gehen entsprechend Input-file von 1-9! */



   hlpi = 0;
   for (i=0;i<n_nodes;i++)
      hlpi += lcrp->wishes[i];

   hlpi -= lcrp->wishes[me];

   size_remw = (size_t)( hlpi * sizeof(int) );

   lcrp->wish_displ    = (int*)       allocateMemory( size_nint, "lcrp->wish_displ" ); 
   lcrp->due_displ     = (int*)       allocateMemory( size_nint, "lcrp->due_displ" ); 
   lcrp->wishlist_mem  = (int*)       allocateMemory( size_remw, "lcrp->wishlist_mem" ); 
   lcrp->wishlist      = (int**)      allocateMemory( size_np,   "lcrp->wishlist" ); 
   lcrp->hput_pos      = (int*)       allocateMemory( size_np,   "lcrp->hput_pos" ); 

   acc_wishes = 0;
   for (i=0; i<n_nodes; i++){
      lcrp->wish_displ[i] = acc_wishes;
      lcrp->hput_pos[i] = lcrp->lnRows[me]+acc_wishes;
      lcrp->wishlist[i] = &(lcrp->wishlist_mem[acc_wishes]);
      IF_DEBUG(2) printf("XXXX PE %d: lcrp->wishes[%d] = %d lcrp->wish_displ[%d]=%d; lcrp->hput_pos[%d]=%d\n",
	    me, i, lcrp->wishes[i], i, lcrp->wish_displ[i], i, lcrp->hput_pos[i]);
      if  ( (me != i) && !( (i == n_nodes-2) && (me == n_nodes-1) ) ){
	 acc_wishes += lcrp->wishes[i];
      }
      /* auf diese Weise zeigt der Anfang der wishlist fuer die lokalen Elemente auf
       * die gleiche Position wie die wishlist fuer die naechste PE. Sollte aber kein
       * Problem sein, da ich fuer me eh nie drauf zugreifen sollte. Zweite Bedingung
       * garantiert, dass der letzte pointer fuer die letzte PE nicht aus dem Feld
       * heraus zeigt da in vorletzter Iteration bereits nochmal inkrementiert wurde */
   }

   for (i=0; i<n_nodes; i++){
      for (j=0;j<lcrp->wishes[i];j++){
	 lcrp->wishlist[i][j] = cwishlist[i][j]; 
      }
   }

   /* Selbiges jetzt nochmal fuer die duelist */

   hlpi = 0;
   for (i=0;i<n_nodes;i++) hlpi += lcrp->dues[i];
   hlpi -= lcrp->dues[me];

   size_remd = (size_t)( hlpi * sizeof(int) );
   lcrp->duelist_mem  = (int*)       allocateMemory( size_remd, "lcrp->duelist_mem" ); 
   lcrp->duelist      = (int**)      allocateMemory( size_np,   "lcrp->duelist" ); 

   acc_dues = 0;
   for (i=0; i<n_nodes; i++){
      lcrp->due_displ[i] = acc_dues;
      lcrp->duelist[i] = &(lcrp->duelist_mem[acc_dues]);
      if  ( (me != i) && !( (i == n_nodes-2) && (me == n_nodes-1) ) ){
	 acc_dues += lcrp->dues[i];
      }
      /* auf diese Weise zeigt der Anfang der wishlist fuer die lokalen Elemente auf
       * die gleiche Position wie die wishlist fuer die naechste PE. Sollte aber kein
       * Problem sein, da ich fuer me eh nie drauf zugreifen sollte. Zweite Bedingung
       * garantiert, dass der letzte pointer fuer die letzte PE nicht aus dem Feld
       * heraus zeigt da in vorletzter Iteration bereits nochmal inkrementiert wurde */
   }

   for (i=0;i<lcrp->dues[0]; i++) lcrp->duelist[0][i] = 77;

   IF_DEBUG(2){
      printf("ZZZZx PE %d: lcrp->dues[0] (vorher) =%d\n", me, lcrp->dues[0]);
      for (i=0;i<lcrp->dues[0]; i++)
	 printf("ZZZZx PE %d: lcrp->duelist[%d][%d] (vorher) =%d %p\n", 
	       me, 0, i, lcrp->duelist[0][i], &lcrp->duelist[0][i]);
   }

   /* Alle Source-Variablen sind bei Scatterv nur auf root relevant; d.h. ich nehme 
    * automatisch _immer_ die richtige (lokale) wishlist zum Verteilen */
   for(i=0; i<n_nodes; i++){
      // for(j=0; j<n_nodes; j++){
      ierr = MPI_Scatterv ( lcrp->wishlist_mem, lcrp->wishes, lcrp->wish_displ, MPI_INTEGER, 
	    lcrp->duelist[i], lcrp->dues[i], MPI_INTEGER, i, MPI_COMM_WORLD );
      //}
   }


   IF_DEBUG(2){
      for (i=0; i<lcrp->nodes; i++){
	 printf("XxX PE %d: lcrp->dues[%d]  = %d, lcrp->due_displ= %d\n", me, i, lcrp->dues[i], lcrp->due_displ[i]);
	 printf("XxX PE %d: lcrp->wishes[%d]= %d, wish_displ=%d\n", me, i, lcrp->wishes[i], lcrp->wish_displ[i]);
      }
   }

   pseudo_ldim = lcrp->lnRows[me]+lcrp->halo_elements ;

   IF_DEBUG(2) for (i=0; i<lcrp->lnEnts[me]; i++) 
      printf("PE:%d col[%d]=%d\n", me, i, lcrp->col[i] );


   loc_count=0;
   for (i=0; i<lcrp->lnEnts[me];i++)
      if (lcrp->col[i]<lcrp->lnRows[me]) loc_count++;

   //for (i=0;i<lcrp->lnRows[me];i++)
   //   printf("PE %d: lokaler row_ptr[%d]: %d )\n", me, i, lcrp->lrow_ptr[i]);


   lnEnts_l = loc_count;
   lnEnts_r = lcrp->lnEnts[me]-loc_count;

   IF_DEBUG(1)printf("PE%d: lnRows=%6d\t lnEnts=%6d\t pseudo_ldim=%6d\t lnEnts_l=%6d\t lnEnts_r=%6d\n", 
	 me, lcrp->lnRows[me], lcrp->lnEnts[me], pseudo_ldim, lnEnts_l, lnEnts_r);

   size_lval = (size_t)( lnEnts_l             * sizeof(double) ); 
   size_rval = (size_t)( lnEnts_r             * sizeof(double) ); 
   size_lcol = (size_t)( lnEnts_l             * sizeof(int) ); 
   size_rcol = (size_t)( lnEnts_r             * sizeof(int) ); 
   size_lptr = (size_t)( (lcrp->lnRows[me]+1) * sizeof(int) ); 
   size_rptr = (size_t)( (lcrp->lnRows[me]+1) * sizeof(int) ); 


   lcrp->lrow_ptr_l = (int*)    allocateMemory( size_lptr, "lcrp->lrow_ptr_l" ); 
   lcrp->lrow_ptr_r = (int*)    allocateMemory( size_rptr, "lcrp->lrow_ptr_r" ); 
   lcrp->lcol       = (int*)    allocateMemory( size_lcol, "lcrp->lcol" ); 
   lcrp->rcol       = (int*)    allocateMemory( size_rcol, "lcrp->rcol" ); 
   lcrp->lval       = (double*) allocateMemory( size_lval, "lcrp->lval" ); 
   lcrp->rval       = (double*) allocateMemory( size_rval, "lcrp->rval" ); 

   IF_DEBUG(1){
      MPI_Barrier(MPI_COMM_WORLD);
      printf("nach LCRP auf PE %d\n", me);
      MPI_Barrier(MPI_COMM_WORLD);
   }


#ifdef PLACE

#pragma omp parallel for schedule(runtime)
   for (i=0; i<lnEnts_l; i++) lcrp->lval[i] = 0.0;

#pragma omp parallel for schedule(runtime)
   for (i=0; i<lnEnts_l; i++) lcrp->lcol[i] = 0.0;

#pragma omp parallel for schedule(runtime)
   for (i=0; i<lnEnts_r; i++) lcrp->rval[i] = 0.0;

#pragma omp parallel for schedule(runtime)
   for (i=0; i<lnEnts_r; i++) lcrp->rcol[i] = 0.0;

#endif


   lcrp->lrow_ptr_l[0] = 0;
   lcrp->lrow_ptr_r[0] = 0;

   IF_DEBUG(1) printf("PE%d: lnRows=%d row_ptr=%d..%d\n", me, lcrp->lnRows[me], lcrp->lrow_ptr[0], lcrp->lrow_ptr[lcrp->lnRows[me]]);

   for (i=0; i<lcrp->lnRows[me]; i++){

      current_l = 0;
      current_r = 0;

      // printf("PE%d in loop: i=%d\n", me, i);


      for (j=lcrp->lrow_ptr[i]; j<lcrp->lrow_ptr[i+1]; j++){
	 //      printf("PE %d: j=%d : Vergleiche %d mit %d\n", me, j, lcrp->col[j], lcrp->lnRows[me]);
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
	 printf("--Row_ptrs-- PE %d: i=%d local=%d remote=%d\n", me, i, lcrp->lrow_ptr_l[i], lcrp->lrow_ptr_r[i]);
      for (i=0; i<lcrp->lrow_ptr_l[lcrp->lnRows[me]]; i++)
	 printf("-- local -- PE%d: lcrp->lcol[%d]=%d\n", me, i, lcrp->lcol[i]);
      for (i=0; i<lcrp->lrow_ptr_r[lcrp->lnRows[me]]; i++)
	 printf("-- remote -- PE%d: lcrp->rcol[%d]=%d\n", me, i, lcrp->rcol[i]);
   }


   //  MPI_Barrier(MPI_COMM_WORLD);
   // printf("Am Ende von LCRP auf PE %d\n", me); fflush(stdout);
   // MPI_Barrier(MPI_COMM_WORLD);


   return lcrp;
}
