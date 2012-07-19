#include <matricks.h>
#include <mpi.h>

void hybrid_kernel_V(int current_iteration, VECTOR_TYPE* res, LCRP_TYPE* lcrp, VECTOR_TYPE* invec){

   /*****************************************************************************
    ********                  Kernel ir -- cs -- wa -- ca                ********   
    ********          Kommunikation mittels MPI_ISend, MPI_IRecv         ********
    ********                serielles Umkopieren und Senden              ********
    ********          !!! Unterdrueckung aller Kommunikation !!!         ********
    ****************************************************************************/

   static int init_kernel=1; 
   static int max_dues;
   static real *work_mem, **work;
   static real hlp_sent;
   static real hlp_recv;

   int me; 
   int i, j, ierr, from_PE, to_PE;
   int send_messages, recv_messages;

   uint64 asm_cycles, asm_cyclecounter;
   uint64 ir_cycles, cs_cycles, wa_cycles, ca_cycles, glob_cycles, glob_cyclecounter;
   real time_it_took;

   static MPI_Request *send_request, *recv_request;
   static MPI_Status  *send_status,  *recv_status;


   real hlp1;

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


      size_mem     = (size_t)( max_dues*lcrp->nodes * sizeof( real  ) );
      size_work    = (size_t)( lcrp->nodes          * sizeof( real* ) );
      size_request = (size_t)( lcrp->nodes          * sizeof( MPI_Request ) );
      size_status  = (size_t)( lcrp->nodes          * sizeof( MPI_Status ) );

      work_mem = (real*)  allocateMemory( size_mem,  "work_mem" );
      work     = (real**) allocateMemory( size_work, "work" );

      for (i=0; i<lcrp->nodes; i++) work[i] = &work_mem[lcrp->due_displ[i]];

      send_request = (MPI_Request*) allocateMemory( size_request, "send_request" );
      recv_request = (MPI_Request*) allocateMemory( size_request, "recv_request" );
      send_status  = (MPI_Status*)  allocateMemory( size_status,  "send_status" );
      recv_status  = (MPI_Status*)  allocateMemory( size_status,  "recv_status" );



      init_kernel = 0;
   }

   send_messages = 0;
   recv_messages = 0;

   /*****************************************************************************
    *******                Initiate MPI_IRecv calls                       *******
    ****************************************************************************/
   IF_DEBUG(1) for_timing_start_asm_( &asm_cyclecounter);

   /*
      for (from_PE=0; from_PE<lcrp->nodes; from_PE++){
      if (lcrp->wishes[from_PE]>0){
      ierr = MPI_Irecv(&invec->val[lcrp->hput_pos[from_PE]], lcrp->wishes[from_PE], 
      MPI_MYDATATYPE, from_PE, from_PE, MPI_COMM_WORLD, 
      &recv_request[recv_messages] );
      recv_messages++;
      }
      }
      */

   IF_DEBUG(1){
      for_timing_stop_asm_( &asm_cyclecounter, &asm_cycles);
      ir_cycles = asm_cycles - cycles4measurement; 
   }
   /*****************************************************************************
    *******       Local assembly of halo-elements  & Communication       ********
    ****************************************************************************/
   IF_DEBUG(1) for_timing_start_asm_( &asm_cyclecounter);

   for (to_PE=0 ; to_PE<lcrp->nodes ; to_PE++){
      for (j=0; j<lcrp->dues[to_PE]; j++){
	 work[to_PE][j] = invec->val[lcrp->duelist[to_PE][j]];
      }
      if (lcrp->dues[to_PE]>0){
	 /*
	    ierr = MPI_Isend( &work[to_PE][0], lcrp->dues[to_PE], 
	    MPI_MYDATATYPE, to_PE, me, MPI_COMM_WORLD, 
	    &send_request[send_messages] );
	    send_messages++;
	    */  
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

   /*
      ierr = MPI_Waitall(send_messages, send_request, send_status);
      ierr = MPI_Waitall(recv_messages, recv_request, recv_status);
      */

   IF_DEBUG(1){
      for_timing_stop_asm_( &asm_cyclecounter, &asm_cycles);
      wa_cycles = asm_cycles - cycles4measurement; 
   }
   /*****************************************************************************
    *******         Calculation of SpMVM for all entries of invec->val         *******
    ****************************************************************************/
   IF_DEBUG(1) for_timing_start_asm_( &asm_cyclecounter);

#pragma omp parallel for schedule(runtime) private (hlp1, j)
   for (i=0; i<lcrp->lnRows[me]; i++){
      hlp1 = 0.0;
      for (j=lcrp->lrow_ptr[i]; j<lcrp->lrow_ptr[i+1]; j++){
	 hlp1 = hlp1 + lcrp->val[j] * invec->val[lcrp->col[j]]; 
      }
      res->val[i] = hlp1;
   }

   IF_DEBUG(1){
      for_timing_stop_asm_( &asm_cyclecounter, &asm_cycles);
      ca_cycles = asm_cycles - cycles4measurement; 
   }
   /*****************************************************************************
    *******    Writeout of timing res->valults for individual contributions    *******
    ****************************************************************************/

   IF_DEBUG(1){

      for_timing_stop_asm_( &glob_cyclecounter, &glob_cycles);
      glob_cycles = glob_cycles - cycles4measurement; 

      time_it_took = (1.0*ir_cycles)/clockfreq;
      printf("HyK_V: PE %d: It %d: Absetzen IRecv [ms]               : %8.3f\n",
	    me, current_iteration, 1000*time_it_took);

      time_it_took = (1.0*cs_cycles)/clockfreq;
      printf("HyK_V: PE %d: It %d: Umkopieren & Kommunikation [ms]   : %8.3f",
	    me, current_iteration, 1000*time_it_took);
      printf(" Datenvolumen: %6.3f MB (@%6.3f GB/s) Nachrichten: %d\n", 
	    8e-6*hlp_sent, 8e-9*hlp_sent/time_it_took, send_messages );

      time_it_took = (1.0*wa_cycles)/clockfreq;
      printf("HyK_V: PE %d: It %d: MPI_Waitall [ms]                  : %8.3f\n",
	    me, current_iteration, 1000*time_it_took);

      time_it_took = (1.0*ca_cycles)/clockfreq;
      printf("HyK_V: PE %d: It %d: SpMVM (alle Elemente) [ms]        : %8.3f",
	    me, current_iteration, 1000*time_it_took);
      printf(" nnz = %d (@%7.3f GFlop/s)\n", lcrp->lrow_ptr[lcrp->lnRows[me]], 
	    2e-9*lcrp->lrow_ptr[lcrp->lnRows[me]]/time_it_took);

      time_it_took = (1.0*glob_cycles)/clockfreq;
      printf("HyK_V: PE %d: It %d: Kompletter Hybrid-kernel [ms]     : %8.3f\n", 
	    me, current_iteration, 1000*time_it_took); fflush(stdout); 

   }

}



/*  IF_DEBUG(2){
    for (from_PE=0; from_PE<lcrp->nodes; from_PE++){
    if (lcrp->wishes[from_PE]>0) printf
    ("HyK_V: PE%d: erwarte %d Elemente von PE%d (tag=%d) und schreibe diese an Stelle %d in invec->val\n", 
    me,  lcrp->wishes[from_PE], from_PE, from_PE, lcrp->hput_pos[from_PE]);
    }
    }
    */ 
