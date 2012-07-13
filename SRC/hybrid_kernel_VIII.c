#include <matricks.h>
#include "kernel_helper.h"
#include <mpi.h>

void hybrid_kernel_VIII(int current_iteration, VECTOR_TYPE* res, LCRP_TYPE* lcrp, VECTOR_TYPE* invec){

   /*****************************************************************************
    ********                  Kernel ir -- cs -- wa -- ca                ********   
    ********          Kommunikation mittels MPI_ISend, MPI_IRecv         ********
    ********    Umkopieren und Senden in omp parallel for ueber to_PE    ********
    ********                erfordert MPI_THREAD_MULTIPLE                ********
    ****************************************************************************/

   static int init_kernel=1; 
   static int max_dues;
   static double *work_mem, **work;
   static double hlp_sent;
   static double hlp_recv;

   int me; 
   int i, j, ierr, from_PE, to_PE;
   int send_messages, recv_messages;

   uint64 asm_cycles, asm_cyclecounter;
   uint64 ir_cycles, cs_cycles, wa_cycles, ca_cycles, glob_cycles, glob_cyclecounter;
   uint64 cp_in_cycles, cp_res_cycles;

   double time_it_took;

   static MPI_Request *send_request, *recv_request;
   static MPI_Status  *send_status,  *recv_status;

   double hlp1;
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
      size_request = (size_t)( lcrp->nodes          * sizeof( MPI_Request ) );
      size_status  = (size_t)( lcrp->nodes          * sizeof( MPI_Status ) );

      work_mem = (double*)  allocateMemory( size_mem,  "work_mem" );
      work     = (double**) allocateMemory( size_work, "work" );

      for (i=0; i<lcrp->nodes; i++) work[i] = &work_mem[lcrp->due_displ[i]];

      send_request = (MPI_Request*) allocateMemory( size_request, "send_request" );
      recv_request = (MPI_Request*) allocateMemory( size_request, "recv_request" );
      send_status  = (MPI_Status*)  allocateMemory( size_status,  "send_status" );
      recv_status  = (MPI_Status*)  allocateMemory( size_status,  "recv_status" );

      init_kernel = 0;
   }

   send_messages = 0;
   recv_messages = 0;
   for (i=0;i<lcrp->nodes;i++) send_request[i] = MPI_REQUEST_NULL;

   /*****************************************************************************
    *******                Initiate MPI_IRecv calls                       *******
    ****************************************************************************/
   IF_DEBUG(1) for_timing_start_asm_( &asm_cyclecounter);

   for (from_PE=0; from_PE<lcrp->nodes; from_PE++){
      if (lcrp->wishes[from_PE]>0){
	 ierr = MPI_Irecv(&invec->val[lcrp->hput_pos[from_PE]], lcrp->wishes[from_PE], 
	       MPI_DOUBLE, from_PE, from_PE, MPI_COMM_WORLD, 
	       &recv_request[recv_messages] );
	 recv_messages++;
      }
   }

   IF_DEBUG(1){
      for_timing_stop_asm_( &asm_cyclecounter, &asm_cycles);
      ir_cycles = asm_cycles - cycles4measurement; 
   }
   /*****************************************************************************
    *******       Local assembly of halo-elements  & Communication       ********
    ****************************************************************************/
   IF_DEBUG(1) for_timing_start_asm_( &asm_cyclecounter);

#pragma omp parallel for                                                       \
   private(j, ierr)                                                             \
   reduction(+:send_messages)
   for (to_PE=0 ; to_PE<lcrp->nodes ; to_PE++){

      for (j=0; j<lcrp->dues[to_PE]; j++){
	 work[to_PE][j] = invec->val[lcrp->duelist[to_PE][j]];
      }

      /* hierfuer ist MPI_THREAD_MULTIPLE noetig */
      if (lcrp->dues[to_PE]>0){
	 ierr = MPI_Isend( &work[to_PE][0], lcrp->dues[to_PE], MPI_DOUBLE, 
	       to_PE, me, MPI_COMM_WORLD, &send_request[to_PE] );
	 send_messages++;
      }
   }

   IF_DEBUG(1){
      for_timing_stop_asm_( &asm_cyclecounter, &asm_cycles);
      cs_cycles = asm_cycles - cycles4measurement; 
   }
   /*****************************************************************************
    *******       Finishing communication MPI_Waitall                     *******
    ****************************************************************************/
   IF_DEBUG(1) for_timing_start_asm_( &asm_cyclecounter);

   ierr = MPI_Waitall(send_messages, send_request, send_status);
   ierr = MPI_Waitall(recv_messages, recv_request, recv_status);

   IF_DEBUG(1){
      for_timing_stop_asm_( &asm_cyclecounter, &asm_cycles);
      wa_cycles = asm_cycles - cycles4measurement; 
   }
   /*****************************************************************************
    *******         Calculation of SpMVM for all entries of invec->val         *******
    ****************************************************************************/
   
  spmvmKernAll( lcrp, invec, res, &asm_cyclecounter, &asm_cycles, &cycles4measurement,
              &ca_cycles, &cp_in_cycles, &cp_res_cycles, &me);

   /*****************************************************************************
    *******    Writeout of timing res->valults for individual contributions    *******
    ****************************************************************************/

   IF_DEBUG(1){

      for_timing_stop_asm_( &glob_cyclecounter, &glob_cycles);
      glob_cycles = glob_cycles - cycles4measurement; 

      time_it_took = (1.0*ir_cycles)/clockfreq;
      printf("HyK_VIII: PE %d: It %d: Absetzen IRecv [ms]               : %8.3f\n",
	    me, current_iteration, 1000*time_it_took);

      time_it_took = (1.0*cs_cycles)/clockfreq;
      printf("HyK_VIII: PE %d: It %d: Umkopieren & Kommunikation [ms]   : %8.3f",
	    me, current_iteration, 1000*time_it_took);
      printf(" Datenvolumen: %6.3f MB (@%6.3f GB/s) Nachrichten: %d\n", 
	    8e-6*hlp_sent, 8e-9*hlp_sent/time_it_took, send_messages );

      time_it_took = (1.0*wa_cycles)/clockfreq;
      printf("HyK_VIII: PE %d: It %d: MPI_Waitall [ms]                  : %8.3f\n",
	    me, current_iteration, 1000*time_it_took);

      time_it_took = (1.0*ca_cycles)/clockfreq;
      printf("HyK_VIII: PE %d: It %d: SpMVM (alle Elemente) [ms]        : %8.3f",
	    me, current_iteration, 1000*time_it_took);
      printf(" nnz = %d (@%7.3f GFlop/s)\n", lcrp->lrow_ptr[lcrp->lnRows[me]], 
	    2e-9*lcrp->lrow_ptr[lcrp->lnRows[me]]/time_it_took);

      #ifdef OPENCL
      time_it_took = (1.0*cp_in_cycles)/clockfreq;
      printf("HyK_VIII: PE %d: It %d: Rhs nach Device [ms]              : %8.3f",
	    me, current_iteration, 1000*time_it_took);
      printf(" Datenvolumen: %6.3f MB (@%6.3f GB/s)\n", 8e-6*invec->nRows, 
	    8e-9*invec->nRows/time_it_took);

      time_it_took = (1.0*cp_res_cycles)/clockfreq;
      printf("HyK_VIII: PE %d: It %d: Res von Device [ms]               : %8.3f",
	    me, current_iteration, 1000*time_it_took);
      printf(" Datenvolumen: %6.3f MB (@%6.3f GB/s)\n", 8e-6*res->nRows, 
	     8e-9*res->nRows/time_it_took);
      #endif

      time_it_took = (1.0*glob_cycles)/clockfreq;
      printf("HyK_VIII: PE %d: It %d: Kompletter Hybrid-kernel [ms]     : %8.3f\n", 
	    me, current_iteration, 1000*time_it_took); fflush(stdout); 

   }

}

