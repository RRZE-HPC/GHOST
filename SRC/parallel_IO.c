#include <mpi.h>
#include <matricks.h>


void pio_read_cr_rownumbers(CR_TYPE* cr, const char* testcase) {

   char restartfilename[50];
   double startTime, stopTime, ct; 
   double mybytes;
   FILE* RESTFILE;
   int i, j;
   size_t size_offs, size_col, size_val;
   /* Number of successfully read data items */
   size_t sucr;

   size_t size_hlp;
   double* zusteller;

   timing( &startTime, &ct );

   sprintf(restartfilename, "./daten/%s_CRS_bin_rows.dat", testcase);
   IF_DEBUG(1) printf(" \n Lese %s \n", restartfilename);

   if ((RESTFILE = fopen(restartfilename, "rb"))==NULL){
      printf("Fehler beim Oeffnen von %s\n", restartfilename);
      exit(1);
   }

   sucr = fread(&cr->nRows,               sizeof(int),    1,           RESTFILE);
   sucr = fread(&cr->nCols,               sizeof(int),    1,           RESTFILE);
   sucr = fread(&cr->nEnts,               sizeof(int),    1,           RESTFILE);
   
   IF_DEBUG(1){ 
      printf("Number of rows in matrix       = %d\n", cr->nRows);
      printf("Number of columns in matrix    = %d\n", cr->nCols);
      printf("Number of non-zero elements    = %d\n", cr->nEnts);
   }
   IF_DEBUG(2) printf("Allocate memory for arrays\n");

   size_offs = (size_t)( (cr->nRows+1) * sizeof(int) );

   cr->rowOffset = (int*)    allocateMemory( size_offs, "rowOffset" );
   /* dummy allocation; real data is read directly into local structure */
   cr->col       = (int*)    allocateMemory( sizeof(int),  "col" );
   cr->val       = (double*) allocateMemory( sizeof(double),  "val" );

#ifndef NO_PLACEMENT
   IF_DEBUG(1) printf("NUMA-placement for cr->rowOffset (restart-version)\n");
#pragma omp parallel for schedule(runtime)
   for( i = 0; i < cr->nRows+1; i++ ) {
      cr->rowOffset[i] = 0;
   }
#endif

   sucr = fread(&cr->rowOffset[0],        sizeof(int),    cr->nRows+1, RESTFILE);

   timing( &stopTime, &ct );
   IF_DEBUG(2) printf("... done\n"); 
   IF_DEBUG(1){
      printf("Binary read of CRS-rows took %8.2f s \n", 
	    (double)(stopTime-startTime) );
   }

   return;
}

void pio_write_cr_rownumbers(const CR_TYPE* cr, const char* testcase) {
   
   char restartfilename[50];
   double startTime, stopTime, ct; 
   double mybytes;
   FILE* RESTFILE;

   timing( &startTime, &ct );

   sprintf(restartfilename, "./daten/%s_CRS_bin_rows.dat", testcase);

   mybytes = 3.0*sizeof(int) + (cr->nRows+1)*sizeof(int);

   printf(" \n Schreibe %s (%6.2f MB)\n", restartfilename,
	 mybytes/1048576.0) ;

   if ((RESTFILE = fopen(restartfilename, "wb"))==NULL){
      printf("Fehler beim Oeffnen von %s\n", restartfilename);
      exit(1);
   }
   printf("Schreibe: Anzahl Zeilen      = %d\n", cr->nRows);
   printf("Schreibe: Anzahl Spalten     = %d\n", cr->nCols);
   printf("Schreibe: Anzahl NZE         = %d\n", cr->nEnts);
   printf("Schreibe: Array mit row-Offsets (%d Eintraege)\n", cr->nRows);

   fwrite(&cr->nRows,               sizeof(int),    1,           RESTFILE);
   fwrite(&cr->nCols,               sizeof(int),    1,           RESTFILE);
   fwrite(&cr->nEnts,               sizeof(int),    1,           RESTFILE);
   fwrite(&cr->rowOffset[0],        sizeof(int),    cr->nRows+1, RESTFILE);

   fflush(RESTFILE);
   fclose(RESTFILE);

   timing( &stopTime, &ct );
   printf( "Schreiben der CRS-Zeilenlaengen (binaer): %8.2f s \n", 
	 (double)(stopTime-startTime) );
   printf( "Entspricht: %8.2f MB/s \n",  
	 (mybytes/1048576.0)/(double)(stopTime-startTime) );


   return;

}

LCRP_TYPE* parallel_MatRead(char* testcase, int work_dist){

   FILE* Infofile;
   FILE* RowP_file;
   char filename[50];
   int ierr, i, j; 

   int nEnts_glob;
   int one_item;
   int me, nnz, wdim, target_nnz;
   double ws;
   int *tmp_transfers;
   int acc_dues;

   int acc_wishes;
   int *wishlist_counts;
   int *wishlist_mem;
   int **wishlist;

   int fair_lines, lines_loc;
   int  *lines_on_PE, *displ_PE;
   int *nlrow_ptr;
   int *first_line, *nRows_loc;
   int *node_nnzptr, *nEnts_loc;

   int el_dim, ph_dim; 
   char dummychar[72];

   int *comm_remotePE, *comm_remoteEl;
   int *requested_from;
   int *different_wishes;
   int *cwishlist_mem, **cwishlist;
   int *getme_PE, *getme_El, *getme_item;

   int hlpi; 

   int this_pseudo_col, num_pseudo_col; 
   int *pseudocol;
   int *globcol;
   int *reverse_col;

   int max_loc_elements, thisentry;
   int *present_values;
   int *gl_row_ptr;
   int *row_ptr;
   int *col_idx;
   double *val;
   double *diag_el;

   MPI_Info info;
   MPI_File file_handle;
   MPI_Offset offset_in_file, filesize;
   MPI_Status *status = NULL;

   LCRP_TYPE *lcrp;

   int target_rows;

   size_t size_nint, size_a2ai, size_nptr, size_pval;  
   size_t size_glrp, size_lorp, size_val, size_ptr;
   size_t size_col;

   /****************************************************************************
    *******            ........ Executable statements ........           *******
    ***************************************************************************/

   ierr = MPI_Comm_rank(MPI_COMM_WORLD, &me);

   one_item = 1;

   IF_DEBUG(-1){
      ierr = MPI_Barrier(MPI_COMM_WORLD);
      if (me==0) printf("Entering setup_communication\n");
   }

   lcrp = (LCRP_TYPE*) allocateMemory( sizeof(LCRP_TYPE), "lcrp");

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


   lines_on_PE = (int*) allocateMemory( lcrp->nodes * sizeof(int), "lines_on_PE");
   displ_PE    = (int*) allocateMemory( lcrp->nodes * sizeof(int), "displ_PE"   );
   first_line  = (int*) allocateMemory( lcrp->nodes * sizeof(int), "first_line" );
   node_nnzptr = (int*) allocateMemory( lcrp->nodes * sizeof(int), "node_nnzptr");
   nRows_loc   = (int*) allocateMemory( lcrp->nodes * sizeof(int), "nRows_loc"  );
   nEnts_loc   = (int*) allocateMemory( lcrp->nodes * sizeof(int), "nEnts_loc"  );




   /****************************************************************************
    *******       Read global information on matrix on master PE        ********
    ***************************************************************************/
   if (me==0){

      sprintf(filename, "./daten/Info_%s_pio.txt", testcase);
      Infofile = fopen(filename, "r");
      if( Infofile == NULL ) mypabort("PE 0: failed to read Info file\n");

      while (fscanf(Infofile, "%d %d %d", &(lcrp->nRows), &el_dim, &ph_dim) <3) 
	 fscanf(Infofile, "%s", dummychar);
      while (fscanf(Infofile, "%d", &(lcrp->nEnts))<1)
	 fscanf(Infofile, "%s", dummychar);

      target_nnz = lcrp->nEnts / lcrp->nodes;
      ws = (lcrp->nRows*20.0 + lcrp->nEnts*12.0)/(1024*1024);

      printf("-----------------------------------------------------\n");
      printf("-------         Statistics about matrix       -------\n");
      printf("-----------------------------------------------------\n");
      printf("Investigated matrix         : %12s\n", testcase); 
      printf("Dimension of matrix         : %12.0f\n", (float)lcrp->nRows); 
      printf("Non-zero elements           : %12.0f\n", (float)lcrp->nEnts); 
      printf("Av. el. per row (ex. diag.) : %12.3f\n", (float)lcrp->nEnts/(float)lcrp->nRows); 
      printf("Working set [MB]            : %12.3f\n", ws); 
      printf("-----------------------------------------------------\n");
      printf("------   Setup matrices in different formats   ------\n");
      printf("-----------------------------------------------------\n");

      size_glrp = (size_t)( (size_t)(lcrp->nRows+1) * sizeof(int) );

      gl_row_ptr = (int*) allocateMemory( size_glrp, "gl_row_ptr");

   }

   /****************************************************************************
    *******     Distribute global information on matrix to all PEs      ********
    ***************************************************************************/
   ierr = MPI_Bcast(&lcrp->nRows, 1,  MPI_INT,  0, MPI_COMM_WORLD);
   ierr = MPI_Bcast(&lcrp->nEnts, 1,  MPI_INT,  0, MPI_COMM_WORLD);
   ierr = MPI_Bcast(testcase,     12, MPI_CHAR, 0, MPI_COMM_WORLD);



   /****************************************************************************
    *******     Trap uninteded use of equal local count partitioning     *******
    ***************************************************************************/

   if (work_dist == EQUAL_LNZE){
      printf("PE%d: falsches worksharing Konzept -- not yet implemented\n", me);
      MPI_Abort(MPI_COMM_WORLD, 999);
   }

   fair_lines = lcrp->nRows / lcrp->nodes;
   if (me==lcrp->nodes-1) 
      /* +1 wegen zusaetzlichem Eintrag (d.h. wdim) in row_ptr[n+1]*/ 
      lines_loc = lcrp->nRows - me*fair_lines + 1; 
   else	
      lines_loc = fair_lines;

   /****************************************************************************
    *******       Read an equal portion of the row_ptr on each PE        *******
    ***************************************************************************/

   sprintf(filename, "./daten/RowP_%s_pio.bin", testcase);
   ierr = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &file_handle);

// #define ROWPTR8
#ifdef ROWPTR8 

   printf("RowPtr angenommen als 8-byte integer\n");
   size_lorp = (size_t)( (size_t)(lines_loc) * sizeof(long long int) );
   nlrow_ptr = (int*) allocateMemory( size_lorp, "nlrow_ptr");
   offset_in_file = 8*me*fair_lines;
   ierr = MPI_File_seek(file_handle, offset_in_file, MPI_SEEK_SET);
   ierr = MPI_File_read(file_handle, nlrow_ptr, lines_loc, MPI_LONG_LONG_INT, status);

#else

   printf("RowPtr angenommen als 4-byte integer\n");
   size_lorp = (size_t)( (size_t)(lines_loc) * sizeof(int) );
   nlrow_ptr = (int*) allocateMemory( size_lorp, "nlrow_ptr");
   offset_in_file = 4*me*fair_lines;
   ierr = MPI_File_seek(file_handle, offset_in_file, MPI_SEEK_SET);
   ierr = MPI_File_read(file_handle, nlrow_ptr, lines_loc, MPI_INT, status);

#endif

   ierr = MPI_File_close(&file_handle);

   /****************************************************************************
    *******     Gather the number of lines on each PE at the master PE   *******
    ***************************************************************************/
   ierr = MPI_Gather(&lines_loc, one_item, MPI_INT, lines_on_PE, one_item, MPI_INT, 0, MPI_COMM_WORLD);

   if (me==0){
      displ_PE[0] = 0;
      for(i=1; i<lcrp->nodes;i++){
	 displ_PE[i] = displ_PE[i-1] + lines_on_PE[i-1];
      }
   }

#ifdef ROWPTR8
   ierr = MPI_Gatherv(nlrow_ptr, lines_loc, MPI_INT, gl_row_ptr, lines_on_PE, displ_PE, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
#else
   ierr = MPI_Gatherv(nlrow_ptr, lines_loc, MPI_INT, gl_row_ptr, lines_on_PE, displ_PE, MPI_INT, 0, MPI_COMM_WORLD);
#endif

   free(nlrow_ptr);

   /****************************************************************************
    *******  Calculate a fair partitioning of NZE and ROWS on master PE  *******
    ***************************************************************************/
   if (me==0){

      if (work_dist == EQUAL_NZE){
	 IF_DEBUG(-1) printf("Distribute Matrix with EQUAL_NZE on each PE\n");
	 target_nnz = (lcrp->nEnts/lcrp->nodes)+1; /* sonst bleiben welche uebrig! */

	 lcrp->lfRow[0]  = 0;
	 lcrp->lfEnt[0] = 0;
	 j = 1;

	 for (i=0; i<lcrp->nRows; i++){
	    if (gl_row_ptr[i] >= j*target_nnz){
	       lcrp->lfRow[j] = i;
	       lcrp->lfEnt[j] = gl_row_ptr[i];
	       j = j+1;
	    }
	 }

      }
      else if (work_dist == EQUAL_LNZE){
	 IF_DEBUG(-1) printf("Distribute Matrix with EQUAL_LNZE on each PE\n");
	 printf("Not yet implemented for parallel reading\n");
         MPI_Abort(MPI_COMM_WORLD, 999);
	 }
      else {

	 IF_DEBUG(-1) printf("Distribute Matrix with EQUAL_ROWS on each PE\n");
	 target_rows = (lcrp->nRows/lcrp->nodes);

	 lcrp->lfRow[0] = 0;
	 lcrp->lfEnt[0] = 0;

	 for (i=1; i<lcrp->nodes; i++){
	    lcrp->lfRow[i] = lcrp->lfRow[i-1]+target_rows;
	    lcrp->lfEnt[i] = nlrow_ptr[lcrp->lfRow[i]];
	 }
      }
      

      for (i=0; i<lcrp->nodes-1; i++){
	 lcrp->lnRows[i] = lcrp->lfRow[i+1] - lcrp->lfRow[i] ;
	 lcrp->lnEnts[i] = lcrp->lfEnt[i+1] - lcrp->lfEnt[i] ;
      }

      lcrp->lnRows[lcrp->nodes-1] = lcrp->nRows - lcrp->lfRow[lcrp->nodes-1] ;
      lcrp->lnEnts[lcrp->nodes-1] = lcrp->nEnts - lcrp->lfEnt[lcrp->nodes-1];
   }


   /****************************************************************************
    *******            Distribute correct share to all PEs               *******
    ***************************************************************************/

   ierr = MPI_Bcast(lcrp->lfRow,  lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD);
   ierr = MPI_Bcast(lcrp->lfEnt,  lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD);
   ierr = MPI_Bcast(lcrp->lnRows, lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD);
   ierr = MPI_Bcast(lcrp->lnEnts, lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD);

   /****************************************************************************
    *******   Allocate memory for matrix in distributed CRS storage      *******
    ***************************************************************************/

   size_val  = (size_t)( (size_t)(lcrp->lnEnts[me])   * sizeof( double ) );
   size_col  = (size_t)( (size_t)(lcrp->lnEnts[me])   * sizeof( int ) );
   size_ptr  = (size_t)( (size_t)(lcrp->lnRows[me]+1) * sizeof( int ) );

   lcrp->val      = (double*)    allocateMemory( size_val,  "lcrp->val" ); 
   lcrp->col      = (int*)       allocateMemory( size_col,  "lcrp->col" ); 
   lcrp->lrow_ptr = (int*)       allocateMemory( size_ptr,  "lcrp->lrow_ptr" ); 

   /****************************************************************************
    *******        Ensure correct NUMA-placement for local arrays        *******
    ***************************************************************************/

#ifdef PLACE

#pragma omp parallel for schedule(runtime)
   for (i=0; i<lcrp->lnEnts[me]; i++) lcrp->val[i] = 0.0;

#pragma omp parallel for schedule(runtime)
   for (i=0; i<lcrp->lnEnts[me]; i++) lcrp->col[i] = 0.0;

#endif

   /****************************************************************************
    *******   Parallel MPI-I/O of the corresponding part of each array   *******
    ***************************************************************************/

   col_idx   = (int*)    allocateMemory( lcrp->lnEnts[me] * sizeof(int),    "col_idx"  );
   val       = (double*) allocateMemory( lcrp->lnEnts[me] * sizeof(double), "val"  );
   diag_el   = (double*) allocateMemory( lcrp->lnRows[me] * sizeof(double), "diag_el"  );
   row_ptr   = (int*)    allocateMemory( lcrp->lnRows[me] * sizeof(double), "row_ptr"  );

   /* Fangen wir mit den Spaltenindizes an ... */ 
   sprintf(filename, "./daten/Cols_%s_pio.bin", testcase);
   offset_in_file = sizeof(int)*lcrp->lfEnt[me];

   ierr = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &file_handle);
   ierr = MPI_File_seek(file_handle, offset_in_file, MPI_SEEK_SET);
   ierr = MPI_File_read(file_handle, col_idx, lcrp->lnEnts[me], MPI_INT, status);
   ierr = MPI_File_close(&file_handle);

   /* ... weiter mit den Nichtnullelementen, diesmal double ... */ 
   sprintf(filename, "./daten/Ents_%s_pio.bin", testcase);
   offset_in_file = sizeof(double)*lcrp->lfEnt[me];

   ierr = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &file_handle);
   ierr = MPI_File_seek(file_handle, offset_in_file, MPI_SEEK_SET);
   ierr = MPI_File_read(file_handle, val, lcrp->lnEnts[me], MPI_DOUBLE, status);
   ierr = MPI_File_close(&file_handle);


#ifdef DIAG_ON_ITS_OWN
   /* ... jetzt noch die Diagonalelemente ...*/
   sprintf(filename, "./daten/Diag_%s_pio.bin", testcase);
   offset_in_file = sizeof(double)*lcrp->lfRow[me];

   ierr = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &file_handle);
   ierr = MPI_File_seek(file_handle, offset_in_file, MPI_SEEK_SET);
   ierr = MPI_File_read(file_handle, diag_el, lcrp->lnRows[me], MPI_DOUBLE, status);
   ierr = MPI_File_close(&file_handle);
#endif


   /* ... und schliesslich versende die (umverteilten) row_ptr ... */ 
   ierr = MPI_Scatterv (gl_row_ptr, lcrp->lnRows, lcrp->lfRow, MPI_INT,
	 lcrp->lrow_ptr, lcrp->lnRows[me], MPI_INT, 0, MPI_COMM_WORLD);


   printf("PE %d: Einlesen fertig\n", me);

   /*****************************************************************************
    ****** Einsortieren der Diagonalelemente in die normale CRS-Struktur   ******
    ****************************************************************************/
#ifdef INSERT_DIAG

   printf("PE %d: halte %d Zeilen, also muss ich meine %d nze entsprechend aufbohren\n",
	 me, nRows_loc[me], nEnts_loc[me]);

   nEnts_old = n_Ents_loc[me];
   nEnts_loc[me] = nEnts_loc[me] + nRpws_loc[me];

   /* Wenn ich mich nicht auf die Sortierung der Elemente verlassen will 
    * wird es das beste sein ich lege mir gleich komplett neue arrays an */

   printf("PE%d: Lokale row_ptr: \n", me, ); 




#endif

   //ierr = MPI_Abort(MPI_COMM_WORLD, 999);


   /* Entsprechnung lokaler <-> globaler Indizes im nze-array auf allen PEs*/
   IF_DEBUG(-1){
      printf("PE %d: nz-elements (local): %7d -- %7d <-> (global): %7d -- %7d\n", 
	    me, 0, nEnts_loc[me]-1, node_nnzptr[me], node_nnzptr[me]+nEnts_loc[me]-1 );
      printf("PE %d: rows        (local): %7d -- %7d <-> (global): %7d -- %7d\n",
	    me, 0, nRows_loc[me]-1, first_line[me], first_line[me]+nRows_loc[me]-1 );
   }

   comm_remotePE      = (int*)          allocateMemory( nEnts_loc[me] * sizeof( int ), "comm_remotePE" );
   comm_remoteEl      = (int*)          allocateMemory( nEnts_loc[me] * sizeof( int ), "comm_remoteEl" );

   /* Berechne aus den globalen column-Indizes die (zweidimensionalen lokalen-/nichtlokalen Indizes) */
   for (i=0;i<nEnts_loc[me];i++){
      for (j=lcrp->nodes-1;j>-1; j--){
	 if (first_line[j]<col_idx[i]) {
	    /* Entsprechendes Paarelement liegt auf PE j */
	    comm_remotePE[i] = j;
	    /* Uebergang von FORTRAN-NUMERIERUNG AUF C-STYLE !!!!!!!!!!!!!!!!!! */
	    comm_remoteEl[i] = col_idx[i]-1 -first_line[j];
	    break;
	 }
      }
   }

   ierr = MPI_Barrier(MPI_COMM_WORLD); IF_DEBUG(-1) if (me==0) printf("setting up wishlists\n");


   wishlist_counts  = (int*)       allocateMemory( lcrp->nodes * sizeof( int ),             "wishlist_counts" ); 

   for (i=0; i<lcrp->nodes; i++) wishlist_counts[i] = 0;

   for (i=0;i<nEnts_loc[me];i++)
      wishlist_counts[comm_remotePE[i]]++;

   ierr = MPI_Barrier(MPI_COMM_WORLD); IF_DEBUG(-1) if (me==0) printf("wishlist counts assembled\n");

   acc_wishes = 0;
   for (i=0; i<lcrp->nodes; i++) acc_wishes += wishlist_counts[i];

   wishlist_mem  = (int*)       allocateMemory( acc_wishes * sizeof( int ),             "wishlist_counts" ); 
   wishlist      = (int**)      allocateMemory( lcrp->nodes    * sizeof( int* ),            "wishlist" ); 

   ierr = MPI_Barrier(MPI_COMM_WORLD); IF_DEBUG(-1) if (me==0) printf("memory for wishlists allocated\n");

   acc_wishes = 0;
   for (i=0; i<lcrp->nodes; i++){
      wishlist[i] = &wishlist_mem[acc_wishes];
      IF_DEBUG(-2) printf("zzzz PE %d: Eintraege in der wishlist an PE %d: %d\n", me, i, wishlist_counts[i]);
      acc_wishes += wishlist_counts[i];
   }


   ierr = MPI_Barrier(MPI_COMM_WORLD); IF_DEBUG(-1) if (me==0) printf("pointers for wishlists set\n");

   IF_DEBUG(-2){
      for (i=0;i<nRows_loc[me];i++)
	 printf("PE %d: globaler Row_ptr(%d)=%d -> lokal: %d\n", 
	       me, i, row_ptr[i], row_ptr[i]-node_nnzptr[me]);
   }



   ierr = MPI_Barrier(MPI_COMM_WORLD); IF_DEBUG(-1) if (me==0) printf("allocating lcrp\n");

   lcrp->val     = (double*)    allocateMemory( nEnts_loc[me] * sizeof( double ),    "lcrp->val" ); 
   lcrp->col     = (int*)       allocateMemory( nEnts_loc[me] * sizeof( int ),       "lcrp->col" ); 
   lcrp->row_ptr = (int*)       allocateMemory( (nRows_loc[me]+1) * sizeof( int ),   "lcrp->row_ptr" ); 
   lcrp->lrow_ptr = (int*)      allocateMemory( (nRows_loc[me]+1) * sizeof( int ),   "lcrp->lrow_ptr" ); 

   ierr = MPI_Barrier(MPI_COMM_WORLD); IF_DEBUG(-1) if (me==0) printf("lcrp allocated\n");

   lcrp->nodes = lcrp->nodes;

   for (i=0;i<lcrp->nodes;i++){
      lcrp->lnEnts[i] = nEnts_loc[i];
      lcrp->lnRows[i] = nRows_loc[i];
      lcrp->lfEnt[i]  = node_nnzptr[i];
      lcrp->lfRow[i]  = first_line[i];
   }

   requested_from  = (int*)       allocateMemory( lcrp->nodes * sizeof( int ),             "requested_from" ); 


#ifdef PLACE
#pragma omp parallel for schedule(runtime)
#endif
   for (i=0; i<nEnts_loc[me]; i++) lcrp->val[i] = 0.0;

#ifdef PLACE
#pragma omp parallel for schedule(runtime)
#endif
   for (i=0; i<nEnts_loc[me]; i++) lcrp->col[i] = 0.0;


   //pseudoindex = lnRows[me];
   for (i=0;i<lcrp->nodes;i++) requested_from[i] = 0;



   for (i=0;i<nEnts_loc[me];i++){
      IF_DEBUG(-2) printf("zz PE %d: Teile %d: %d: %d %d\n", me, i, comm_remotePE[i] , 
	    requested_from[comm_remotePE[i]] , comm_remoteEl[i]);
      wishlist[comm_remotePE[i]][requested_from[comm_remotePE[i]]] = comm_remoteEl[i];
      requested_from[comm_remotePE[i]]++;
   }


   IF_DEBUG(-2){
      for (i=0; i<lcrp->nodes; i++){
	 for (j=0;j<wishlist_counts[i];j++){
	    printf("zz PE %d: wishlist von PE %d: %d -> %d\n", me, i, j, wishlist[i][j]);
	 }
      }
   }

   different_wishes  = (int*)       allocateMemory( lcrp->nodes * sizeof( int ),             "different_wishes" ); 

   cwishlist_mem  = (int*)       allocateMemory( acc_wishes * sizeof( int ),             "cwishlist_counts" ); 
   cwishlist      = (int**)      allocateMemory( lcrp->nodes    * sizeof( int* ),            "cwishlist" ); 

   ierr = MPI_Barrier(MPI_COMM_WORLD); IF_DEBUG(-1) if (me==0) printf("next allocated\n");

   acc_wishes = 0;
   for (i=0; i<lcrp->nodes; i++){
      cwishlist[i] = &cwishlist_mem[acc_wishes];
      acc_wishes += wishlist_counts[i];
   }


   ierr = MPI_Barrier(MPI_COMM_WORLD); IF_DEBUG(-1) if (me==0) printf("wuehlen\n");


   /****************************************
    *
    * Zeitfresser
    * 
    *
    * *******************************************/

#ifdef TUNE_SETUP

   max_loc_elements = 0;
   for (i=0;i<lcrp->nodes;i++)
      if (max_loc_elements<lcrp->lnRows[i]) max_loc_elements = lcrp->lnRows[i];

   //printf("Max_loc_elements: %d\n", max_loc_elements);

   present_values  = (int*)       allocateMemory( max_loc_elements * sizeof( int ),             "present_values" ); 

   for (i=0; i<lcrp->nodes; i++){

      for (j=0; j<max_loc_elements; j++) present_values[j] = -1;

      IF_DEBUG(-2) printf("SuC: PE %d: wishlist_counts[%d]=%d\n", me, i, wishlist_counts[i]);fflush(stdout);

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
      IF_DEBUG(-2){printf("PE: %d: different_wishes von %d :%d\n", me, i, different_wishes[i]);fflush(stdout);}
   }
   /*
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Abort(MPI_COMM_WORLD,999);
      */

#else

#pragma omp parallel for schedule(runtime) private (j, found_agreement, k)
   for (i=0; i<lcrp->nodes; i++){
      IF_DEBUG(-2) printf("SuC: PE %d: wishlist_counts[%d]=%d\n", me, i, wishlist_counts[i]);fflush(stdout);
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


   //printf("Komprimieren der Indizes PE %d: %f s\n", me, time_it_took);

   ierr = MPI_Barrier(MPI_COMM_WORLD); IF_DEBUG(-1) if (me==0) printf("setting up cwishlists\n");

   for (i=0; i<lcrp->nodes; i++){
      lcrp->wishes[i] = different_wishes[i];
      IF_DEBUG(-2) printf("zzZ PE %d: Eintraege in der cwishlist an PE %d: %d\n", me, i, different_wishes[i]);
   }
   lcrp->wishes[me] = 0; /* kein lokaler Transfer */

   /* Verteile lcrp->wishes and die jeweiligen PEs um lcrp->dues zu erhalten */
   tmp_transfers  = (int*)       allocateMemory( lcrp->nodes*lcrp->nodes * sizeof( int ),      "tmp_transfers" ); 
   ierr = MPI_Allgather ( lcrp->wishes, lcrp->nodes, MPI_INT, tmp_transfers, lcrp->nodes, MPI_INT, MPI_COMM_WORLD ) ;

   IF_DEBUG(-2){
      if (me == 0) for(i=0;i<lcrp->nodes*lcrp->nodes;i++) printf("ME==0: tmp_tansfers[%d]=%d\n", i, tmp_transfers[i]);
   }


   for (i=0; i<lcrp->nodes; i++){
      lcrp->dues[i] = tmp_transfers[i*lcrp->nodes+me];
      IF_DEBUG(-2) printf("ZzZ PE %d: lcrp->dues[%d]=%d\n", me, i, lcrp->dues[i]);
   }

   lcrp->dues[me] = 0; /* keine lokalen Transfers */

   IF_DEBUG(-2){
      for (i=0; i<lcrp->nodes; i++){
	 for (j=0;j<different_wishes[i];j++){
	    printf("zzZ PE %d: cwishlist von PE %d: %d -> %d\n", me, i, j, cwishlist[i][j]);
	 }
      }
   }

   nEnts_glob = node_nnzptr[lcrp->nodes-1]+lcrp->lnEnts[lcrp->nodes-1]; 



   pseudocol      = (int*)          allocateMemory( nEnts_loc[me] * sizeof( int ), "pseudocol" );
   globcol      = (int*)          allocateMemory( nEnts_loc[me] * sizeof( int ), "origincol" );
   getme_PE      = (int*)          allocateMemory( nEnts_loc[me] * sizeof( int ), "getme_PE" );
   getme_El      = (int*)          allocateMemory( nEnts_loc[me] * sizeof( int ), "getme_El" );
   getme_item      = (int*)          allocateMemory( nEnts_loc[me] * sizeof( int ), "getme_item" );
   reverse_col      = (int*)          allocateMemory( nEnts_glob * sizeof( int ), "reverse_col" );

   ierr = MPI_Barrier(MPI_COMM_WORLD);

   /* Extrahiere aus cwishlist die komprimierten Pseudo-Indizes fuer den invec-Zugriff */
   this_pseudo_col = nRows_loc[me];
   num_pseudo_col = 0;
   for (i=0; i<lcrp->nodes; i++){
      if (i != me){ /* natuerlich nur fuer remote-Elemente */
	 for (j=0;j<different_wishes[i];j++){
	    pseudocol[num_pseudo_col] = this_pseudo_col;  
	    globcol[num_pseudo_col]   = lcrp->lfRow[i]+cwishlist[i][j]; 
	    getme_PE[num_pseudo_col]  = i;
	    getme_item[num_pseudo_col]  = j;
	    getme_El[num_pseudo_col]  = cwishlist[i][j];
	    reverse_col[globcol[num_pseudo_col]] = num_pseudo_col;
	    IF_DEBUG(-2) printf("zzZ PE %d: cwishlist von PE %d: Item %d -> loc_Element %d (= %d -- global)\n",
		  me, i, j, cwishlist[i][j], globcol[num_pseudo_col]);
	    num_pseudo_col++;
	    this_pseudo_col++;
	 }
      }
   }

   IF_DEBUG(-2) printf("Anzahl Elemente im Halo fuer PE %d: %d\n", me, num_pseudo_col);

   lcrp->halo_elements = num_pseudo_col;

   IF_DEBUG(-2){
      for (i=0;i<num_pseudo_col;i++) printf("zzZ PE %d: Pseudocols: %d %d %d -> (%d,%d) %d \n", 
	    me, i, pseudocol[i], globcol[i], getme_PE[i], getme_El[i], getme_item[i]);
   }

   ierr = MPI_Barrier(MPI_COMM_WORLD);

   for (i=0;i<nEnts_loc[me];i++){

      lcrp->val[i]     = val[i];

      if (comm_remotePE[i] == me){
	 IF_DEBUG(-2) printf("ZZZZ PE %d: Belege_nun_if   %d orig: %d -> %d\n", 
	       me, i, col_idx[i]-1, comm_remoteEl[i]);
	 fflush(stdout);
	 lcrp->col[i] = comm_remoteEl[i];
      }
      else{
	 IF_DEBUG(-2) printf("ZZZZ PE %d: Belege_nun_else %d orig: %d -> %d\n", 
	       me, i, col_idx[i]-1, pseudocol[reverse_col[col_idx[i]-1]]);
	 fflush(stdout);
	 lcrp->col[i] = pseudocol[reverse_col[col_idx[i]-1]];
      }
      /*
	 lcrp->rem_PE[i]  = comm_remotePE[i];
	 lcrp->rem_El[i]  = comm_remoteEl[i];
	 */
   }


   ierr = MPI_Barrier(MPI_COMM_WORLD);
   IF_DEBUG(-2) for (i=0;i<nEnts_loc[me];i++) printf("zZZZ PE %d: i=%d -> col=%d\n", me, i, lcrp->col[i]);
   ierr = MPI_Barrier(MPI_COMM_WORLD);

   /* !!!!!!!!!!!!!! Eintraege in wishlist gehen entsprechend Input-file von 1-9! */



   for (i=0;i<nRows_loc[me];i++){
      IF_DEBUG(-2) printf("PE %d: Belege i=%d: %d\n", me, i, row_ptr[i]);
      lcrp->row_ptr[i] = row_ptr[i]; 
   }
   lcrp->row_ptr[nRows_loc[me]] = node_nnzptr[me]+lcrp->lnEnts[me]; 

   for (i=0;i<nRows_loc[me]+1;i++)
      lcrp->lrow_ptr[i] =  lcrp->row_ptr[i] - node_nnzptr[me]; 

   IF_DEBUG(-2) printf("PE %d: letzer row_ptr=: %d\n", me,  lcrp->row_ptr[nRows_loc[me]]);

   hlpi = 0;
   for (i=0;i<lcrp->nodes;i++)
      hlpi += lcrp->wishes[i];

   hlpi -= lcrp->wishes[me];

   lcrp->wish_displ    = (int*)       allocateMemory( lcrp->nodes * sizeof( int ),      "lcrp->wish_displ" ); 
   lcrp->due_displ     = (int*)       allocateMemory( lcrp->nodes * sizeof( int ),      "lcrp->due_displ" ); 
   lcrp->wishlist_mem  = (int*)       allocateMemory( hlpi * sizeof( int ),      "lcrp->wishlist_mem" ); 
   lcrp->wishlist      = (int**)      allocateMemory( lcrp->nodes * sizeof( int* ),    "lcrp->wishlist" ); 
   lcrp->hput_pos      = (int*)       allocateMemory( lcrp->nodes * sizeof( int* ),    "lcrp->hput_pos" ); 

   acc_wishes = 0;
   for (i=0; i<lcrp->nodes; i++){
      lcrp->wish_displ[i] = acc_wishes;
      lcrp->hput_pos[i] = lcrp->lnRows[me]+acc_wishes;
      lcrp->wishlist[i] = &(lcrp->wishlist_mem[acc_wishes]);
      IF_DEBUG(-2) printf("XXXX PE %d: lcrp->wishes[%d] = %d lcrp->wish_displ[%d]=%d; lcrp->hput_pos[%d]=%d\n",
	    me, i, lcrp->wishes[i], i, lcrp->wish_displ[i], i, lcrp->hput_pos[i]);
      if  ( (me != i) && !( (i == lcrp->nodes-2) && (me == lcrp->nodes-1) ) ){
	 acc_wishes += lcrp->wishes[i];
      }
      /* auf diese Weise zeigt der Anfang der wishlist fuer die lokalen Elemente auf
       * die gleiche Position wie die wishlist fuer die naechste PE. Sollte aber kein
       * Problem sein, da ich fuer me eh nie drauf zugreifen sollte. Zweite Bedingung
       * garantiert, dass der letzte pointer fuer die letzte PE nicht aus dem Feld
       * heraus zeigt da in vorletzter Iteration bereits nochmal inkrementiert wurde */
   }

   for (i=0; i<lcrp->nodes; i++){
      for (j=0;j<lcrp->wishes[i];j++){
	 lcrp->wishlist[i][j] = cwishlist[i][j]; 
      }
   }

   /* Selbiges jetzt nochmal fuer die duelist */

   hlpi = 0;
   for (i=0;i<lcrp->nodes;i++)
      hlpi += lcrp->dues[i];

   hlpi -= lcrp->dues[me];

   lcrp->duelist_mem  = (int*)       allocateMemory( hlpi * sizeof( int ),      "lcrp->duelist_mem" ); 
   lcrp->duelist      = (int**)      allocateMemory( lcrp->nodes * sizeof( int* ),  "lcrp->duelist" ); 

   acc_dues = 0;
   for (i=0; i<lcrp->nodes; i++){
      lcrp->due_displ[i] = acc_dues;
      lcrp->duelist[i] = &(lcrp->duelist_mem[acc_dues]);
      if  ( (me != i) && !( (i == lcrp->nodes-2) && (me == lcrp->nodes-1) ) ){
	 acc_dues += lcrp->dues[i];
      }
      /* auf diese Weise zeigt der Anfang der wishlist fuer die lokalen Elemente auf
       * die gleiche Position wie die wishlist fuer die naechste PE. Sollte aber kein
       * Problem sein, da ich fuer me eh nie drauf zugreifen sollte. Zweite Bedingung
       * garantiert, dass der letzte pointer fuer die letzte PE nicht aus dem Feld
       * heraus zeigt da in vorletzter Iteration bereits nochmal inkrementiert wurde */
   }

   for (i=0;i<lcrp->dues[0]; i++) lcrp->duelist[0][i] = 77;

   IF_DEBUG(-2){
      printf("ZZZZx PE %d: lcrp->dues[0] (vorher) =%d\n", me, lcrp->dues[0]);
      for (i=0;i<lcrp->dues[0]; i++)
	 printf("ZZZZx PE %d: lcrp->duelist[%d][%d] (vorher) =%d %p\n", 
	       me, 0, i, lcrp->duelist[0][i], &lcrp->duelist[0][i]);
   }

   /* Alle Source-Variablen sind bei Scatterv nur auf root relevant; d.h. ich nehme 
    * automatisch _immer_ die richtige (lokale) wishlist zum Verteilen */
   for(i=0; i<lcrp->nodes; i++){
      // for(j=0; j<lcrp->nodes; j++){
      ierr = MPI_Scatterv ( lcrp->wishlist_mem, lcrp->wishes, lcrp->wish_displ, MPI_INT, 
	    lcrp->duelist[i], lcrp->dues[i], MPI_INT, i, MPI_COMM_WORLD );
      //}
   }


   IF_DEBUG(-2){
      for (i=0; i<lcrp->nodes; i++){
	 printf("XxX PE %d: lcrp->dues[%d]  = %d, lcrp->due_displ= %d\n", me, i, lcrp->dues[i], lcrp->due_displ[i]);
	 printf("XxX PE %d: lcrp->wishes[%d]= %d, wish_displ=%d\n", me, i, lcrp->wishes[i], lcrp->wish_displ[i]);
      }
   }


   return lcrp;
}

void pio_write_cr(const CR_TYPE* cr, const char* testcase){

   int me, ierr;
   char filename[250];
   MPI_Info info;
   MPI_File file_handle;
   MPI_Offset offset_in_file, filesize;
   MPI_Status *status=NULL;

   ierr = MPI_Comm_rank(MPI_COMM_WORLD, &me);
 


   /* Fangen wir mit den Spaltenindizes an ... */ 
   sprintf(filename, "/home/vault/unrz/unrz265/myCols_%s_pio.bin", testcase);

   printf("%s\n", filename);

   offset_in_file = 0;

   ierr = MPI_File_open(MPI_COMM_SELF, filename, MPI_MODE_WRONLY + MPI_MODE_CREATE, 
	 MPI_INFO_NULL, &file_handle);
   printf("Rueckgabe:%d\n", ierr);
   ierr = MPI_File_seek(file_handle, offset_in_file, MPI_SEEK_SET);
   printf("Rueckgabe:%d\n", ierr);
   ierr = MPI_File_write(file_handle, cr->col, cr->nEnts, MPI_INT, status);
   printf("Rueckgabe:%d\n", ierr);
   ierr = MPI_File_close(&file_handle);

   /* nun die Werte ... */ 
   sprintf(filename, "/home/vault/unrz/unrz265/myVals_%s_pio.bin", testcase);
   offset_in_file = 0;

   if (me==0){
      ierr = MPI_File_open(MPI_COMM_SELF, filename, MPI_MODE_WRONLY + MPI_MODE_CREATE,
	    MPI_INFO_NULL, &file_handle);
      printf("Rueckgabe:%d\n", ierr);
      ierr = MPI_File_seek(file_handle, offset_in_file, MPI_SEEK_SET);
      printf("Rueckgabe:%d\n", ierr);
      ierr = MPI_File_write(file_handle, cr->val, cr->nEnts, MPI_DOUBLE, status);
      printf("Rueckgabe:%d\n", ierr);
      //ierr = MPI_File_write_at(file_handle, offset_in_file, cr->val, cr->nEnts, MPI_DOUBLE, status);
      ierr = MPI_File_close(&file_handle);
   }

   /* ... und jetzt noch die row_ptr ... */ 
   sprintf(filename, "/home/vault/unrz/unrz265/myRowptr_%s_pio.bin", testcase);
   offset_in_file = 0;

   if (me==0){
      ierr = MPI_File_open(MPI_COMM_SELF, filename, MPI_MODE_WRONLY + MPI_MODE_CREATE,
	    MPI_INFO_NULL, &file_handle);
      ierr = MPI_File_seek(file_handle, offset_in_file, MPI_SEEK_SET);
      printf("Rueckgabe:%d\n", ierr);
      ierr = MPI_File_write(file_handle, cr->rowOffset, (cr->nRows)+1, MPI_INT, status);
      //ierr = MPI_File_write_at(file_handle, offset_in_file, cr->rowOffset, (cr->nRows)+1, MPI_INT, status);
      ierr = MPI_File_close(&file_handle);
   }
}


