#include <matricks.h>
#include <mpi.h>

void check_lcrp(int me, LCRP_TYPE *lcrp){

   int i, j;
   size_t size;
   FILE* DATFILE;
   char datfilename[50];

   int* all_dues;

   IF_DEBUG(1) printf("============= PE %d : setup_communication completed ==============\n", me);
   IF_DEBUG(2){
      for(i=0;i<lcrp->lnEnts[me];i++) printf("CHCK_LCRP: PE %d: col[%d] = %d val=%f\n",
	    me, i, lcrp->col[i], lcrp->val[i]);
   }

   IF_DEBUG(1){
      printf("CHCK_LCRP: PE %d: lokale Anzahl Zeilen =%d, Elemente=%d, Halo=%d\n", 
	    me, lcrp->lnRows[me], lcrp->lnEnts[me], lcrp->halo_elements);

      if(lcrp->nodes>=4){

	 printf("CHCK_LCRP: PE %d: Number of wishes from PE 0-3: %d   %d   %d   %d\n", 
	       me, lcrp->wishes[0], lcrp->wishes[1], lcrp->wishes[2], lcrp->wishes[3]);

	 printf("CHCK_LCRP: PE %d: wishlist-pointer to   PE 0-3: %p   %p   %p   %p\n", 
	       me, lcrp->wishlist[0], lcrp->wishlist[1], lcrp->wishlist[2], lcrp->wishlist[3]);

	 printf("CHCK_LCRP: PE %d: Number of   dues  to  PE 0-3: %d   %d   %d   %d\n", 
	       me, lcrp->dues[0], lcrp->dues[1], lcrp->dues[2], lcrp->dues[3]);
      }
   }

   IF_DEBUG(2){

      for (i=0;i<lcrp->nodes;i++)
	 if (me != i) 
	    for(j=0;j<lcrp->wishes[i];j++)
	       printf("CHCK_LCRP: PE %d: wishlist from PE %d: Eintrag %d -> %d\n", 
		     me, i, j, lcrp->wishlist[i][j]);


      for (i=0;i<lcrp->nodes;i++)
	 if (me != i) 
	    for(j=0;j<lcrp->dues[i];j++)
	       printf("CHCK_LCRP: PE %d:  duelist  to  PE %d: Eintrag %d -> %d\n", 
		     me, i, j, lcrp->duelist[i][j]);
   }

   IF_DEBUG(1){
      size = (size_t)( lcrp->nodes*lcrp->nodes*sizeof(int) );
      all_dues = (int*) allocateMemory(size, "all_dues");

      MPI_Gather(&lcrp->dues[0], lcrp->nodes, MPI_INTEGER, all_dues, lcrp->nodes, MPI_INTEGER, 0, MPI_COMM_WORLD);

      if (me==0){ 
	 sprintf(datfilename, "Comm_scheme.dat");
	 if ((DATFILE = fopen(datfilename, "w"))==NULL) MPI_Abort(MPI_COMM_WORLD,999);
	 for (i=0; i<lcrp->nodes; i++) 
	    for (j=0; j<lcrp->nodes; j++)
	       fprintf(DATFILE, "%d %d %8.3f %d\n", i, j, 
		     all_dues[i*lcrp->nodes+j]*8/(1024.0*1024.0), all_dues[i*lcrp->nodes+j]);
	 fclose(DATFILE);
      }
   }
   return;

}
