#include "matricks.h"
#include <stdio.h>
#include "timing.h"

char restartfilename[50];
double startTime, stopTime, ct; 
double mybytes;
FILE* RESTFILE;

/***************************************************************
 *          Ausgabe der Matrix in Binaerformat (CRS)           *
 **************************************************************/

void bin_write_cr(const CR_TYPE* cr, const char* testcase){

   timing( &startTime, &ct );

   sprintf(restartfilename, "./daten/%s_CRS_bin.dat", testcase);

   mybytes = 3.0*sizeof(int) + 1.0*(cr->nRows+cr->nEnts)*sizeof(int) +
      1.0*(cr->nEnts)*sizeof(double);

   IF_DEBUG(1) printf(" \n Schreibe %s (%6.2f MB)\n", restartfilename,
	 mybytes/1048576.0) ;

   if ((RESTFILE = fopen(restartfilename, "wb"))==NULL){
      printf("Fehler beim Oeffnen von %s\n", restartfilename);
      exit(1);
   }
   IF_DEBUG(1) printf("Schreibe: Anzahl Zeilen      = %d\n", cr->nRows);
   IF_DEBUG(1) printf("Schreibe: Anzahl Spalten     = %d\n", cr->nCols);
   IF_DEBUG(1) printf("Schreibe: Anzahl NZE         = %d\n", cr->nEnts);
   IF_DEBUG(1) printf("Schreibe: Array mit row-Offsets (%d Eintraege)\n", cr->nRows);
   IF_DEBUG(1) printf("Schreibe: Array mit Spalteneintraegen (%d Eintraege)\n", cr->nEnts);
   IF_DEBUG(1) printf("Schreibe: Array mit Matrixeintraegen (%d Eintraege)\n", cr->nEnts);

   fwrite(&cr->nRows,               sizeof(int),    1,           RESTFILE);
   fwrite(&cr->nCols,               sizeof(int),    1,           RESTFILE);
   fwrite(&cr->nEnts,               sizeof(int),    1,           RESTFILE);
   fwrite(&cr->rowOffset[0],        sizeof(int),    cr->nRows+1, RESTFILE);
   fwrite(&cr->col[0],              sizeof(int),    cr->nEnts,   RESTFILE);
   fwrite(&cr->val[0],              sizeof(double), cr->nEnts,   RESTFILE);

   fflush(RESTFILE);
   fclose(RESTFILE);

   timing( &stopTime, &ct );
   IF_DEBUG(1) printf( "Schreiben der Matrix im CRS-Format (binaer): %8.2f s \n", 
	 (double)(stopTime-startTime) );
   IF_DEBUG(1) printf( "Entspricht: %8.2f MB/s \n",  
	 (mybytes/1048576.0)/(double)(stopTime-startTime) );


   return;
}

/***************************************************************
 *          Ausgabe der Matrix in Binaerformat (JDS)           *
 **************************************************************/

void bin_write_jd(const JD_TYPE* jd, const char* testcase){

   timing( &startTime, &ct );

   sprintf(restartfilename, "./daten/%s_JDS_bin.dat", testcase);

   mybytes = 4.0*sizeof(int) 
      + 1.0*(jd->nRows + jd->nEnts + jd->nDiags+1)*sizeof(int) 
      + 1.0*(jd->nEnts)*sizeof(double);

   printf(" \n Schreibe %s (%6.2f MB)\n", restartfilename,
	 mybytes/1048576.0) ;

   if ((RESTFILE = fopen(restartfilename, "wb"))==NULL){
      printf("Fehler beim Oeffnen von %s\n", restartfilename);
      exit(1);
   }
   printf("Schreibe: Anzahl Zeilen      = %d\n", jd->nRows);
   printf("Schreibe: Anzahl Spalten     = %d\n", jd->nCols);
   printf("Schreibe: Anzahl NZE         = %d\n", jd->nEnts);
   printf("Schreibe: Anzahl Nebendiags  = %d\n", jd->nDiags);
   printf("Schreibe: Array mit Permutationen (%d Eintraege)\n", jd->nRows);
   printf("Schreibe: Array mit Offsets der NDs (%d Eintraege)\n", jd->nDiags+1);
   printf("Schreibe: Array mit Spaltenindizes (%d Eintraege)\n", jd->nEnts);
   printf("Schreibe: Array mit Matrixeintraegen (%d Eintraege)\n", jd->nEnts);

   fwrite(&jd->nRows,               sizeof(int),    1,            RESTFILE);
   fwrite(&jd->nCols,               sizeof(int),    1,            RESTFILE);
   fwrite(&jd->nEnts,               sizeof(int),    1,            RESTFILE);
   fwrite(&jd->nDiags,              sizeof(int),    1,            RESTFILE);
   fwrite(&jd->rowPerm[0],          sizeof(int),    jd->nRows,    RESTFILE);
   fwrite(&jd->diagOffset[0],       sizeof(int),    jd->nDiags+1, RESTFILE);
   fwrite(&jd->col[0],              sizeof(int),    jd->nEnts,    RESTFILE);
   fwrite(&jd->val[0],              sizeof(double), jd->nEnts,    RESTFILE);

   fflush(RESTFILE);
   fclose(RESTFILE);

   timing( &stopTime, &ct );
   printf( "Schreiben der Matrix im JDS-Format (binaer): %8.2f s \n", 
	 (double)(stopTime-startTime) );
   printf( "Entspricht: %8.2f MB/s \n",  
	 (mybytes/1048576.0)/(double)(stopTime-startTime) );

   return;
}

/*******************************************************************************
 *          Einlesen der Matrix in Binaerformat (CRS)                          *
 ******************************************************************************/

void bin_read_cr(CR_TYPE* cr, const char* testcase){

   int i, j;
   size_t size_offs, size_col, size_val;
   /* Number of successfully read data items */
   size_t sucr;

   size_t size_hlp;
   double* zusteller;

   timing( &startTime, &ct );

   sprintf(restartfilename, "/home/vault/unrz/unrza317/matrices/%s/%s_CRS_bin.dat", testcase,testcase);
   IF_DEBUG(1) printf(" \n Lese %s \n", restartfilename);

   if ((RESTFILE = fopen(restartfilename, "rb"))==NULL){
      printf("Fehler beim Oeffnen von %s\n", restartfilename);
      exit(1);
   }

   sucr = fread(&cr->nRows,               sizeof(int),    1,           RESTFILE);
   sucr = fread(&cr->nCols,               sizeof(int),    1,           RESTFILE);
   sucr = fread(&cr->nEnts,               sizeof(int),    1,           RESTFILE);

   mybytes = 3.0*sizeof(int) + 1.0*(cr->nRows+cr->nEnts)*sizeof(int) +
      1.0*(cr->nEnts)*sizeof(double);

   IF_DEBUG(1){ 
      printf("Number of rows in matrix       = %d\n", cr->nRows);
      printf("Number of columns in matrix    = %d\n", cr->nCols);
      printf("Number of non-zero elements    = %d\n", cr->nEnts);
      printf(" \n Entries to be read sum up to %6.2f MB\n", mybytes/1048576.0) ;
   }
   IF_DEBUG(2) printf("Allocate memory for arrays\n");

   size_offs = (size_t)( (cr->nRows+1) * sizeof(int) );
   size_col  = (size_t)( cr->nEnts * sizeof(int) );
   size_val  = (size_t)( cr->nEnts * sizeof(double) );

   cr->rowOffset = (int*)    allocateMemory( size_offs, "rowOffset" );
   cr->col       = (int*)    allocateMemory( size_col,  "col" );
   cr->val       = (double*) allocateMemory( size_val,  "val" );

   IF_DEBUG(2){
      printf("Reading array with row-offsets\n");
      printf("Reading array with column indices\n");
      printf("Reading array with values\n");
   }	

   NUMA_CHECK_SERIAL("before placement zusteller");

   IF_DEBUG(1) printf("gezieltes placement in die falschen LD\n");
   size_hlp = (size_t) ( 450000000*sizeof(double));
   zusteller = (double*) allocateMemory( size_hlp,  "zusteller" );

#pragma omp parallel for schedule(runtime)
   for( i = 0; i < 450000000; i++ )  zusteller[i] = 0;



   NUMA_CHECK_SERIAL("after placement zusteller");

#ifndef NO_PLACEMENT
   IF_DEBUG(1) printf("NUMA-placement for cr->rowOffset (restart-version)\n");
#pragma omp parallel for schedule(runtime)
   for( i = 0; i < cr->nRows+1; i++ ) {
      cr->rowOffset[i] = 0;
   }
#endif

   sucr = fread(&cr->rowOffset[0],        sizeof(int),    cr->nRows+1, RESTFILE);


#ifndef NO_PLACEMENT
   IF_DEBUG(1){
      printf("Doing NUMA-placement for cr->col (restart-version)\n");
      printf("Doing NUMA-placement for cr->val (restart-version)\n");
   }
#pragma omp parallel for schedule(runtime)
   for(i = 0 ; i < cr->nRows; ++i) {
      for(j = cr->rowOffset[i] ; j < cr->rowOffset[i+1] ; j++) {
	 cr->val[j] = 0.0;
	 cr->col[j] = 0;
      }
   }
#endif 

   sucr = fread(&cr->col[0],              sizeof(int),    cr->nEnts,   RESTFILE);
   sucr = fread(&cr->val[0],              sizeof(double), cr->nEnts,   RESTFILE);

   fclose(RESTFILE);

   NUMA_CHECK_SERIAL("after CR-binary read");

   freeMemory(size_hlp, "zusteller", zusteller);

   NUMA_CHECK_SERIAL("after freeing zusteller");

/* lieber ausserhalb der PE=0 Region um Probleme mit oversubscribing zu vermeiden
 * #ifdef CMEM
   if (allocatedMem > 0.02*total_mem){
      IF_DEBUG(1) printf("CR setup: Large matrix -- allocated mem=%8.3f MB\n",
	    (float)(allocatedMem)/(1024.0*1024.0));
      sweepMemory(SINGLE);
      IF_DEBUG(1) printf("Nach memsweep\n"); fflush(stdout);
   }
#endif
*/

   timing( &stopTime, &ct );
   IF_DEBUG(2) printf("... done\n"); 
   IF_DEBUG(1){
      printf("Binary read of matrix in CRS-format took %8.2f s \n", 
	    (double)(stopTime-startTime) );
      printf( "Data transfer rate : %8.2f MB/s \n",  
	    (mybytes/1048576.0)/(double)(stopTime-startTime) );
   }

   return;
}

/***************************************************************
 *          Einlesen der Matrix in Binaerformat (JDS)           *
 **************************************************************/


void bin_read_jd(JD_TYPE* jd, const int blocklen, const char* testcase){

   int i, ib, block_start, block_end, diag, diagLen, offset;
   size_t sucr;

   timing( &startTime, &ct );

   sprintf(restartfilename, "./daten/%s_JDS_bin.dat", testcase);
   IF_DEBUG(1) printf(" \n Lese %s \n", restartfilename);

   if ((RESTFILE = fopen(restartfilename, "rb"))==NULL){
      printf("Fehler beim Oeffnen von %s\n", restartfilename);
      exit(1);
   }

   sucr = fread(&jd->nRows,               sizeof(int),    1,            RESTFILE);
   sucr = fread(&jd->nCols,               sizeof(int),    1,            RESTFILE);
   sucr = fread(&jd->nEnts,               sizeof(int),    1,            RESTFILE);
   sucr = fread(&jd->nDiags,              sizeof(int),    1,            RESTFILE);

   mybytes = 4.0*sizeof(int) 
      + 1.0*(jd->nRows + jd->nEnts + jd->nDiags+1)*sizeof(int) 
      + 1.0*(jd->nEnts)*sizeof(double);

   IF_DEBUG(1) {
      printf("Number of rows in matrix       = %d\n", jd->nRows);
      printf("Number of columns in matrix    = %d\n", jd->nCols);
      printf("Number of non-zero elements    = %d\n", jd->nEnts);
      printf("Number of off-diagonals        = %d\n", jd->nDiags);
      printf(" \n Entries to be read sum up to %6.2f MB\n", mybytes/1048576.0) ;
   }

   IF_DEBUG(2) printf("Allocate memory for arrays\n");

   jd->rowPerm    = (int*)    allocateMemory( jd->nRows      * sizeof( int ),    "rowPerm" );
   jd->diagOffset = (int*)    allocateMemory( (jd->nDiags+1) * sizeof( int ),    "diagOffset" );
   jd->col        = (int*)    allocateMemory( jd->nEnts      * sizeof( int ),    "col" );
   jd->val        = (double*) allocateMemory( jd->nEnts      * sizeof( double ), "val" );

   IF_DEBUG(2) {
      printf("Reading array of permutations\n");
      printf("Reading array of offsets of off-diagonals\n");
      printf("Reading array with column indices\n");
      printf("Reading array with values\n");
   }	

   sucr = fread(&jd->rowPerm[0],          sizeof(int),    jd->nRows,    RESTFILE);
   sucr = fread(&jd->diagOffset[0],       sizeof(int),    jd->nDiags+1, RESTFILE);

#ifndef NO_PLACEMENT
   printf("NUMA-placement of jd->col[] and jd->val[]\n");
#pragma omp parallel for schedule(runtime) private (i, diag, diagLen, offset, block_start, block_end) 
   for(ib = 0 ; ib < jd->nRows ; ib += blocklen) {

      block_start = ib;
      block_end = MIN(ib+blocklen-2, jd->nRows-1);

      for(diag=0; diag < jd->nDiags ; diag++) {

	 diagLen = jd->diagOffset[diag+1]-jd->diagOffset[diag];
	 offset  = jd->diagOffset[diag];

	 if(diagLen >= block_start) {

	    for(i=block_start; i<= MIN(block_end,diagLen-1); ++i) {
	       jd->val[offset+i]=0.0;
	       jd->col[offset+i]=0.0;
	    }
	 }
      }
   } 
   /* GH: then fill matrix */
#endif   // placement of matrix in JDS format


   sucr = fread(&jd->col[0],              sizeof(int),    jd->nEnts,    RESTFILE);
   sucr = fread(&jd->val[0],              sizeof(double), jd->nEnts,    RESTFILE);

   fclose(RESTFILE);

   timing( &stopTime, &ct );
   IF_DEBUG(2) printf("... done\n"); 
   IF_DEBUG(1){
      printf("Binary read of matrix in JDS-format took %8.2f s \n", 
	    (double)(stopTime-startTime) );
      printf( "Data transfer rate : %8.2f MB/s \n\n",  
	    (mybytes/1048576.0)/(double)(stopTime-startTime) );
   }

   return;
}
