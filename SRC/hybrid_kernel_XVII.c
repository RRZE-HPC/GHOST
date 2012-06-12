#include <matricks.h>
#include <mpi.h>
#include <omp.h>
#include <sys/types.h>

void hybrid_kernel_XVII(int current_iteration, VECTOR_TYPE* res, LCRP_TYPE* lcrp, VECTOR_TYPE* invec){

   /*****************************************************************************
    ********              Kernel ir -- cs -- lc -- wa -- nl              ********
    ********   'Good faith' hybrid -- mit parallelem Umkopieren          ********
    ********     - das was andere als 'hybrid' bezeichnen                ********
    ********     - ob es klappt oder nicht haengt vom MPI ab...          ********
    ********     - Zusammenfassung der send und receive requests         ********
    ****************************************************************************/

   static int init_kernel=1; 
   static int max_dues;
   static double *work_mem, **work;
   static double hlp_sent;
   static double hlp_recv;

   int me; 
   int i, j, ierr;
   int from_PE, to_PE;
   int send_messages, recv_messages;

   uint64 asm_cycles, asm_cyclecounter, asm_acccyclecounter;
   double time_it_took;
   uint64 glob_cycles, glob_cyclecounter;

   /* Required cycles for the individual contributions */
   uint64 ir_cycles, cs_cycles, wa_cycles, lc_cycles, nl_cycles;
   uint64 cp_lin_cycles, cp_nlin_cycles, cp_res_cycles;

   double hlp1;
   static MPI_Request *all_requests;
   static MPI_Status  *all_status;

   size_t size_request, size_status, size_work, size_mem;

   /*****************************************************************************
    *******            ........ Executable statements ........           ********
    ****************************************************************************/
   IF_DEBUG(1) for_timing_start_asm_( &glob_cyclecounter);

   ierr = MPI_Comm_rank(MPI_COMM_WORLD, &me);

   if (init_kernel==1){

      max_dues = 0;
      for (i=0;i<lcrp->nodes;i++)
	 if (lcrp->dues[i]>max_dues) 
	    max_dues = lcrp->dues[i];

      hlp_sent = 0.0;
      hlp_recv = 0.0;
      for (i=0;i<lcrp->nodes; i++){
	 hlp_sent += lcrp->dues[i];
	 hlp_recv += lcrp->wishes[i];
      }

      IF_DEBUG(2) printf("Hybrid_kernel: PE %d: max_dues= %d\n", me, max_dues);

      size_mem     = (size_t)( max_dues*lcrp->nodes * sizeof( double  ) );
      size_work    = (size_t)( lcrp->nodes          * sizeof( double* ) );
      size_request = (size_t)( 2*lcrp->nodes        * sizeof( MPI_Request ) );
      size_status  = (size_t)( 2*lcrp->nodes        * sizeof( MPI_Status ) );

      work_mem = (double*)  allocateMemory( size_mem,  "work_mem" );
      work     = (double**) allocateMemory( size_work, "work" );

      for (i=0; i<lcrp->nodes; i++) work[i] = &work_mem[lcrp->due_displ[i]];

      all_requests = (MPI_Request*) allocateMemory( size_request, "send_request" );
      all_status   = (MPI_Status*)  allocateMemory( size_status,  "send_status" );

      init_kernel = 0;
   }


   send_messages=0;
   recv_messages = 0;
   for (i=0;i<lcrp->nodes;i++) all_requests[2*i] = MPI_REQUEST_NULL;

   /*****************************************************************************
    *******        Post of Irecv to ensure that we are prepared...       ********
    ****************************************************************************/
   IF_DEBUG(1) for_timing_start_asm_( &asm_acccyclecounter);

   for (from_PE=0; from_PE<lcrp->nodes; from_PE++){
      if (lcrp->wishes[from_PE]>0){
	 ierr = MPI_Irecv( &invec->val[lcrp->hput_pos[from_PE]], lcrp->wishes[from_PE], 
	       MPI_DOUBLE, from_PE, from_PE, MPI_COMM_WORLD, 
	       &all_requests[2*recv_messages+1] );
	 recv_messages++;
      }
   }

   IF_DEBUG(1){
      for_timing_stop_asm_( &asm_acccyclecounter, &asm_cycles);
      ir_cycles = asm_cycles - cycles4measurement; 
   } 
   /*****************************************************************************
    *******       Local assembly of halo-elements  & Communication       ********
    ****************************************************************************/
   IF_DEBUG(1) for_timing_start_asm_( &asm_cyclecounter);

   for (to_PE=0 ; to_PE<lcrp->nodes ; to_PE++){

#pragma omp parallel if (lcrp->dues[to_PE]>1000)                               \
      default (none)                                                           \
      private (j)                                                              \
      shared (work, invec, lcrp, to_PE) 
      {
#pragma omp for 
	 for (j=0; j<lcrp->dues[to_PE]; j++){
	    work[to_PE][j] = invec->val[lcrp->duelist[to_PE][j]];
	 }
      }

      if (lcrp->dues[to_PE]>0){
	 ierr = MPI_Isend( &work[to_PE][0], lcrp->dues[to_PE], 
	       MPI_DOUBLE, to_PE, me, MPI_COMM_WORLD, 
	       &all_requests[2*send_messages] );
	 send_messages++;
      }
   }

   IF_DEBUG(1){
      for_timing_stop_asm_( &asm_cyclecounter, &asm_cycles);
      cs_cycles = asm_cycles - cycles4measurement; 
   }
   /****************************************************************************
    *******       Calculation of SpMVM for local entries of invec->val        *******
    ***************************************************************************/
    
   spmvmKernLocal( lcrp, invec, res, &asm_cyclecounter, &asm_cycles, &cycles4measurement,
                &lc_cycles, &cp_lin_cycles, &me );
                
   /****************************************************************************
    *******       Finishing communication: MPI_Waitall                   *******
    ***************************************************************************/
   IF_DEBUG(1) for_timing_start_asm_( &asm_cyclecounter);

   ierr = MPI_Waitall(2*send_messages, all_requests, all_status);

   IF_DEBUG(1){
      for_timing_stop_asm_( &asm_cyclecounter, &asm_cycles);
      wa_cycles = asm_cycles - cycles4measurement; 
   }
   /****************************************************************************
    *******     Calculation of SpMVM for non-local entries of invec->val      *******
    ***************************************************************************/
    
   spmvmKernRemote( lcrp, invec, res, &asm_cyclecounter, &asm_cycles, &cycles4measurement,
                  &nl_cycles, &cp_nlin_cycles, &cp_res_cycles, &me );
                  
   /****************************************************************************
    *******    Writeout of timing res->valults for individual contributions   *******
    ***************************************************************************/
   IF_DEBUG(1){

      for_timing_stop_asm_( &glob_cyclecounter, &glob_cycles);
      glob_cycles = glob_cycles - cycles4measurement; 

      time_it_took = (1.0*ir_cycles)/clockfreq;
      printf("HyK_XVII: PE %d: It %d: Absetzen des Irecv [ms]           : %8.3f",
	    me, current_iteration, 1000*time_it_took);
      printf(" (in cycles %llu) Nachrichten: %d\n", ir_cycles, recv_messages );

      time_it_took = (1.0*cs_cycles)/clockfreq;
      printf("HyK_XVII: PE %d: It %d: Umkopieren & Absetzen ISend [ms]  : %8.3f",
	    me, current_iteration, 1000*time_it_took);
      printf(" Datenvolumen: %6.3f MB (@%6.3f GB/s) Nachrichten: %d\n", 
	    8e-6*hlp_sent, 8e-9*hlp_sent/time_it_took, send_messages );

      time_it_took = (1.0*wa_cycles)/clockfreq;
      printf("HyK_XVII: PE %d: It %d: MPI_Waitall                 [ms]  : %8.3f",
	    me, current_iteration, 1000*time_it_took);
      printf(" Datenvolumen: %6.3f MB (@%6.3f GB/s) Nachrichten: %d\n", 
	    8e-6*hlp_sent, 8e-9*hlp_sent/time_it_took, send_messages );

      time_it_took = (1.0*lc_cycles)/clockfreq;
      printf("HyK_XVII: PE %d: It %d: SpMVM (lokale Elemente) [ms]      : %8.3f",
	    me, current_iteration, 1000*time_it_took);
      printf(" nnz_l = %d (@%7.3f GFlop/s)\n", lcrp->lrow_ptr_l[lcrp->lnRows[me]], 
	    2e-9*lcrp->lrow_ptr_l[lcrp->lnRows[me]]/time_it_took);

      time_it_took = (1.0*nl_cycles)/clockfreq;
      printf("HyK_XVII: PE %d: It %d: SpMVM (nichtlokale Elemente) [ms] : %8.3f",
	    me, current_iteration, 1000*time_it_took);
      printf(" nnz_nl= %d (@%7.3f GFlop/s)\n", lcrp->lrow_ptr_r[lcrp->lnRows[me]], 
	    2e-9*lcrp->lrow_ptr_r[lcrp->lnRows[me]]/time_it_took);

    #ifdef OCLKERNEL
      time_it_took = (1.0*cp_lin_cycles)/clockfreq;
      printf("HyK_XVII: PE %d: It %d: Rhs (lokal) nach Device [ms]      : %8.3f",
	    me, current_iteration, 1000*time_it_took);
      printf(" Datenvolumen: %6.3f MB (@%6.3f GB/s)\n", 8e-6*lcrp->lnRows[me], 
	     8e-9*lcrp->lnRows[me]/time_it_took);

      time_it_took = (1.0*cp_nlin_cycles)/clockfreq;
      printf("HyK_XVII: PE %d: It %d: Rhs (nichtlokal) nach Device [ms] : %8.3f",
	    me, current_iteration, 1000*time_it_took);
      printf(" Datenvolumen: %6.3f MB (@%6.3f GB/s)\n", 8e-6*lcrp->halo_elements, 
	     8e-9*lcrp->halo_elements/time_it_took);

      time_it_took = (1.0*cp_res_cycles)/clockfreq;
      printf("HyK_XVII: PE %d: It %d: Res von Device [ms]               : %8.3f",
	    me, current_iteration, 1000*time_it_took);
      printf(" Datenvolumen: %6.3f MB (@%6.3f GB/s)\n", 8e-6*res->nRows, 
	     8e-9*res->nRows/time_it_took);
    #endif
    
      time_it_took = (1.0*glob_cycles)/clockfreq;
      printf("HyK_XVII: PE %d: It %d: Kompletter Hybrid-kernel [ms]     : %8.3f\n", 
	    me, current_iteration, 1000*time_it_took); fflush(stdout); 

   }

}
