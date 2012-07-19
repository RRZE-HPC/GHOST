#include <matricks.h>
#include "kernel_helper.h"
#include <mpi.h>

void hybrid_kernel_IV(int current_iteration, VECTOR_TYPE* res, LCRP_TYPE* lcrp, VECTOR_TYPE* invec){


   /****************************************************************************
    ********                 Kernel cp -- co -- ca                      ********   
    ********          Kommunikation mittels MPI_Allgatherv              ********
    ********                 paralleles Umkopieren                      ********
    ***************************************************************************/

   static int init_kernel=1; 
   static int max_dues;
   static real *work_mem, **work;
   static real hlp_sent;
   static real hlp_recv;
   static int *offs_vec;

   int me; 
   int i, j, ierr;
   int send_messages;

   uint64 asm_cycles, asm_cyclecounter;
   uint64 cp_cycles, co_cycles, ca_cycles, glob_cycles, glob_cyclecounter;
   uint64 cp_in_cycles, cp_res_cycles;

   real time_it_took;

   real hlp1;

   size_t size_work, size_mem, size_offs;

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
      size_offs    = (size_t)( lcrp->nodes          * sizeof( int ) );

      work_mem = (real*)  allocateMemory( size_mem,  "work_mem" );
      work     = (real**) allocateMemory( size_work, "work" );
      offs_vec = (int*)     allocateMemory( size_offs, "work" );

      for (i=0; i<lcrp->nodes; i++){
	 work[i] = &work_mem[i*max_dues];
	 offs_vec[i] = i*max_dues;
      }

      init_kernel = 0;
   }

   send_messages = 0;

   /*****************************************************************************
    *******       Local assembly of halo-elements to be transfered       ********
    ****************************************************************************/
   IF_DEBUG(1) for_timing_start_asm_( &asm_cyclecounter);

//#pragma omp parallel for schedule(runtime) private(j) collapse(2)
#pragma omp parallel for schedule(runtime) private(j) 
   for (i=0 ; i<lcrp->nodes ; i++){
      for (j=0; j<lcrp->dues[i]; j++){
	 work[i][j] = invec->val[lcrp->duelist[i][j]];
      }
   }

   IF_DEBUG(1){
      for_timing_stop_asm_( &asm_cyclecounter, &asm_cycles);
      cp_cycles = asm_cycles - cycles4measurement; 
   }
   /*****************************************************************************
    *******                        Communication                         ********
    ****************************************************************************/
   IF_DEBUG(1) for_timing_start_asm_( &asm_cyclecounter);

   ierr = MPI_Alltoallv(work_mem, lcrp->dues, offs_vec, MPI_MYDATATYPE, 
	 &invec->val[lcrp->lnRows[me]], lcrp->wishes, lcrp->wish_displ, MPI_MYDATATYPE, 
	 MPI_COMM_WORLD);

   send_messages = lcrp->nodes-1;

   IF_DEBUG(1){
      for_timing_stop_asm_( &asm_cyclecounter, &asm_cycles);
      co_cycles = asm_cycles - cycles4measurement; 
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

      time_it_took = (1.0*cp_cycles)/clockfreq;
      printf("HyK_IV: PE %d: It %d: Umkopieren [ms]                   : %8.3f",
	    me, current_iteration, 1000*time_it_took);
      printf(" Datenvolumen: %6.3f MB (@%6.3f GB/s)\n", 8e-6*hlp_sent, 
	    8e-9*hlp_sent/time_it_took);

      time_it_took = (1.0*co_cycles)/clockfreq;
      printf("HyK_IV: PE %d: It %d: Kommunikation [ms]                : %8.3f",
	    me, current_iteration, 1000*time_it_took);
      printf(" Datenvolumen: %6.3f MB (@%6.3f GB/s) Nachrichten: %d\n", 
	    8e-6*hlp_sent, 8e-9*hlp_sent/time_it_took, send_messages );

      time_it_took = (1.0*ca_cycles)/clockfreq;
      printf("HyK_IV: PE %d: It %d: SpMVM (alle Elemente) [ms]        : %8.3f",
	    me, current_iteration, 1000*time_it_took);
      printf(" nnz = %d (@%7.3f GFlop/s)\n", lcrp->lrow_ptr[lcrp->lnRows[me]], 
	    2e-9*lcrp->lrow_ptr[lcrp->lnRows[me]]/time_it_took);

    #ifdef OPENCL
      time_it_took = (1.0*cp_in_cycles)/clockfreq;
      printf("HyK_IV: PE %d: It %d: Rhs nach Device [ms]              : %8.3f",
	    me, current_iteration, 1000*time_it_took);
      printf(" Datenvolumen: %6.3f MB (@%6.3f GB/s)\n", 8e-6*invec->nRows, 
	     8e-9*invec->nRows/time_it_took);

      time_it_took = (1.0*cp_res_cycles)/clockfreq;
      printf("HyK_IV: PE %d: It %d: Res von Device [ms]               : %8.3f",
	    me, current_iteration, 1000*time_it_took);
      printf(" Datenvolumen: %6.3f MB (@%6.3f GB/s)\n", 8e-6*res->nRows, 
	     8e-9*res->nRows/time_it_took);
    #endif

      time_it_took = (1.0*glob_cycles)/clockfreq;
      printf("HyK_IV: PE %d: It %d: Kompletter Hybrid-kernel [ms]     : %8.3f\n", 
	    me, current_iteration, 1000*time_it_took); fflush(stdout); 

   }

}
