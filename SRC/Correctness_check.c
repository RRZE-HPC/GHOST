#include <matricks.h>
#include <mpi.h>
#include <math.h>


void Correctness_check( VECTOR_TYPE* resCR, LCRP_TYPE* lcrp, double* hlpvec_out){

   /****************************************************************************
    *****         Perform correctness-check against serial result          *****
    ***************************************************************************/

   int i, ierr, error_count, acc_error_count;
   double mytol;
   static int me;
   static int init_check=1;
   static double* hlpres_serial;

  /*****************************************************************************
   *******            ........ Executable statements ........           ********
   ****************************************************************************/

   if (init_check==1){

      ierr = MPI_Comm_rank (MPI_COMM_WORLD, &me);

      hlpres_serial =  (double*) allocateMemory( 
	    lcrp->lnRows[me] * sizeof( double ), "hlpres_serial" );

      /* Scatter the serial result-vector */
      ierr = MPI_Scatterv ( resCR->val, lcrp->lnRows, lcrp->lfRow, MPI_DOUBLE, 
	    hlpres_serial, lcrp->lnRows[me], MPI_DOUBLE, 0, MPI_COMM_WORLD );

      init_check = 0;
      successful = 0;

   }

   error_count = 0;
   for (i=0; i<lcrp->lnRows[me]; i++){
      mytol = EPSILON * (1.0 + fabs(hlpres_serial[i]) ) ;
      if (fabs(hlpres_serial[i]-hlpvec_out[i]) > mytol){
	 printf( "Correctness-check Hybrid:  PE%d: error in row %i:", me, i);
	 printf(" Differences: %e   Value ser: %25.16e Value par: %25.16e\n",
	       hlpres_serial[i]-hlpvec_out[i], hlpres_serial[i], hlpvec_out[i]);
//	 MPI_ABORT(999, MPI_COMM_WORLD);
	 error_count++;
      }
   }

   ierr = MPI_Reduce ( &error_count, &acc_error_count, 1, MPI_INTEGER, 
	 MPI_SUM, 0, MPI_COMM_WORLD);

   if (me==0) if (acc_error_count == 0) successful++;

   IF_DEBUG(1) if (me==0){
      printf("-------------------------------------------------------\n");
      printf("----------------   Correctness-Check    ---------------\n");
      if (acc_error_count==0) 
	 printf("----------------    *** SUCCESS ***    ---------------\n");
      else printf("FAILED ---  %d errors\n", acc_error_count);
      printf("-------------------------------------------------------\n");
   }

   return;

}
