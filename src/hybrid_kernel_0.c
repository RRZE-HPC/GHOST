#include <matricks.h>
#include <mpi.h>
#include "kernel_helper.h"

void hybrid_kernel_0(int current_iteration, VECTOR_TYPE* res, LCRP_TYPE* lcrp, VECTOR_TYPE* invec){


   /*****************************************************************************
    ********            rein OpenMP-paralleler Kernel: ca                ********   
    ********             ausschliesslich fuer den Fall np=1              ********
    ****************************************************************************/

   int me=0;
   int i, j, start, end;

   uint64 asm_cycles, asm_cyclecounter;
   uint64 ca_cycles, glob_cycles, glob_cyclecounter;
   uint64 cp_in_cycles, cp_res_cycles;
   double time_it_took;

   real hlp1;

  /*****************************************************************************
   *******            ........ Executable statements ........           ********
   ****************************************************************************/
  //IF_DEBUG(1) for_timing_start_asm_( &glob_cyclecounter);

   
  /*****************************************************************************
   *******         Calculation of SpMVM for all entries of invec->val         *******
   ****************************************************************************/
  //IF_DEBUG(1) for_timing_start_asm_( &asm_cyclecounter);

  spmvmKernAll( lcrp, invec, res, &asm_cyclecounter, &asm_cycles, &cycles4measurement,
              &ca_cycles, &cp_in_cycles, &cp_res_cycles, &me);
//#pragma omp parallel for schedule(runtime) private (hlp1, j, start, end)
/*#pragma omp parallel for schedule(static) private (hlp1, j, start, end)
  for (i=0; i<lcrp->lnRows[me]; i++){
     hlp1  = 0.0;
     start = lcrp->lrow_ptr[i];
     end   = lcrp->lrow_ptr[i+1];
#pragma nounroll
     for (j=start; j<end; j++){
	hlp1 = hlp1 + lcrp->val[j] * invec->val[lcrp->col[j]]; 
     }
     res->val[i] = hlp1;
  }
*/
 /* IF_DEBUG(1){
     for_timing_stop_asm_( &asm_cyclecounter, &asm_cycles);
     ca_cycles = asm_cycles - cycles4measurement; 
  }*/
  /*****************************************************************************
   *******    Writeout of timing res->valults for individual contributions    *******
   ****************************************************************************/

  /*IF_DEBUG(1){

     for_timing_stop_asm_( &glob_cyclecounter, &glob_cycles);
     glob_cycles = glob_cycles - cycles4measurement; 

    time_it_took = (1.0*ca_cycles)/clockfreq;
    printf("HyK_I: PE %d: It %d: SpMVM (alle Elemente) [ms]        : %8.3f",
	  me, current_iteration, 1000*time_it_took);
    printf(" nnz = %d (@%7.3f GFlop/s)\n", lcrp->lrow_ptr[lcrp->lnRows[me]], 
	   2e-9*lcrp->lrow_ptr[lcrp->lnRows[me]]/time_it_took);
    
#ifdef OPENCL
    time_it_took = (1.0*cp_in_cycles)/clockfreq;
    printf("HyK_I: PE %d: It %d: Rhs nach Device [ms]              : %8.3f",
	  me, current_iteration, 1000*time_it_took);
    printf(" Datenvolumen: %6.3f MB (@%6.3f GB/s)\n", 8e-6*invec->nRows, 
	   8e-9*invec->nRows/time_it_took);

    time_it_took = (1.0*cp_res_cycles)/clockfreq;
    printf("HyK_I: PE %d: It %d: Res von Device [ms]               : %8.3f",
	  me, current_iteration, 1000*time_it_took);
    printf(" Datenvolumen: %6.3f MB (@%6.3f GB/s)\n", 8e-6*res->nRows, 
	   8e-9*res->nRows/time_it_took);
    #endif

    time_it_took = (1.0*glob_cycles)/clockfreq;
    printf("HyK_I: PE %d: It %d: Kompletter Hybrid-kernel [ms]     : %8.3f\n", 
	   me, current_iteration, 1000*time_it_took); fflush(stdout); 

  }*/

}
