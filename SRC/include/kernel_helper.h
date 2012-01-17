#ifndef __KERNEL_HELPER_H__
#define __KERNEL_HELPER_H__

#include "mymacros.h"

/*********** kernel for all entries *********************/

inline void spmvmKernAll( LCRP_TYPE* lcrp, VECTOR_TYPE* invec, VECTOR_TYPE* res, 
                      uint64* asm_cyclecounter, uint64* asm_cycles, uint64* cycles4measurement, 
                      uint64* ca_cycles, uint64* cp_in_cycles, uint64* cp_res_cycles, 
                      int* me) {
  /* helper function to call either SpMVM kernel on device with device data transfer (if CUDAKERNEL) 
   * or OMP parallel kernel;
   * ca_cycles: timing measurement for computation of all entries
   * cp_in_cycles/cp_res_cycles: timing for copy to device of input (rhs) vector / copy from device
   *   of result vector, only valid if CUDAKERNEL */

  int i, j;
  double hlp1;

  #ifdef CUDAKERNEL
  IF_DEBUG(1) for_timing_start_asm_( asm_cyclecounter);

  copyHostToDevice(invec->val_gpu, invec->val, invec->nRows*sizeof(double));
  
  IF_DEBUG(1){
     for_timing_stop_asm_( asm_cyclecounter, asm_cycles);
     *cp_in_cycles = *asm_cycles - *cycles4measurement; 
  }
  #endif

  IF_DEBUG(1) for_timing_start_asm_( asm_cyclecounter);
  
  #ifdef CUDAKERNEL
#ifdef ELR
    #ifdef TEXCACHE
      elrCudaKernelTexCache( lcrp->celr->val, lcrp->celr->col, lcrp->celr->rowLen, lcrp->celr->nRows, 
        lcrp->celr->padding, res->val_gpu );
    #else
      elrCudaKernel( lcrp->celr->val, lcrp->celr->col, lcrp->celr->rowLen, lcrp->celr->nRows, 
        lcrp->celr->padding, invec->val_gpu, res->val_gpu );
    #endif
#else
    #ifdef TEXCACHE
      pjdsCudaKernelTexCache( lcrp->cpjds->val, lcrp->cpjds->col, lcrp->cpjds->colStart, lcrp->cpjds->rowLen, lcrp->cpjds->nRows, 
        lcrp->cpjds->padding, res->val_gpu );
    #else
      pjdsCudaKernel( lcrp->cpjds->val, lcrp->cpjds->col, lcrp->cpjds->colStart, lcrp->cpjds->rowLen, lcrp->cpjds->nRows, 
        lcrp->cpjds->padding, invec->val_gpu, res->val_gpu );
    #endif
#endif


  #else
  
    #pragma omp parallel for schedule(runtime) private (hlp1, j)
    for (i=0; i<lcrp->lnRows[*me]; i++){
      hlp1 = 0.0;
      for (j=lcrp->lrow_ptr[i]; j<lcrp->lrow_ptr[i+1]; j++){
	      hlp1 = hlp1 + lcrp->val[j] * invec->val[lcrp->col[j]]; 
      }
      res->val[i] = hlp1;
    }

  #endif

  IF_DEBUG(1){
     for_timing_stop_asm_( asm_cyclecounter, asm_cycles);
     *ca_cycles = *asm_cycles - *cycles4measurement; 
  }
  
  #ifdef CUDAKERNEL
  IF_DEBUG(1) for_timing_start_asm_( asm_cyclecounter);

  copyDeviceToHost( res->val, res->val_gpu, res->nRows*sizeof(double) );

  IF_DEBUG(1){
     for_timing_stop_asm_( asm_cyclecounter, asm_cycles);
     *cp_res_cycles = *asm_cycles - *cycles4measurement; 
  }
  #endif
    
}

/*********** kernel for local entries only *********************/

inline void spmvmKernLocal( LCRP_TYPE* lcrp, VECTOR_TYPE* invec, VECTOR_TYPE* res,
                      uint64* asm_cyclecounter, uint64* asm_cycles, uint64* cycles4measurement, 
                      uint64* lc_cycles, uint64* cp_lin_cycles, int* me) {
  /* helper function to call either SpMVM kernel on device with device data transfer (if CUDAKERNEL) 
   * or OMP parallel kernel;
   * lc_cycles: timing measurement for computation of local entries
   * cp_lin_cycles: timing for copy to device of local elements in input (rhs) vector, 
   *   only valid if CUDAKERNEL */

  int i, j;
  double hlp1;

  #ifdef CUDAKERNEL
  IF_DEBUG(1) for_timing_start_asm_( asm_cyclecounter);
  
  copyHostToDevice(invec->val_gpu, invec->val, lcrp->lnRows[*me]*sizeof(double));

  IF_DEBUG(1){
     for_timing_stop_asm_( asm_cyclecounter, asm_cycles);
     *cp_lin_cycles = *asm_cycles - *cycles4measurement; 
  }
  #endif

  IF_DEBUG(1) for_timing_start_asm_( asm_cyclecounter);
  
  #ifdef CUDAKERNEL
#ifdef ELR
    #ifdef TEXCACHE
      elrCudaKernelTexCache( lcrp->lcelr->val, lcrp->lcelr->col, lcrp->lcelr->rowLen, lcrp->lcelr->nRows, 
        lcrp->lcelr->padding, res->val_gpu );
    #else
      elrCudaKernel( lcrp->lcelr->val, lcrp->lcelr->col, lcrp->lcelr->rowLen, lcrp->lcelr->nRows, 
        lcrp->lcelr->padding, invec->val_gpu, res->val_gpu );
    #endif
#else
    #ifdef TEXCACHE
      pjdsCudaKernelTexCache( lcrp->lcpjds->val, lcrp->lcpjds->col, lcrp->lcpjds->colStart, lcrp->lcpjds->rowLen, lcrp->lcpjds->nRows, 
        lcrp->lcpjds->padding, res->val_gpu );
    #else
      pjdsCudaKernel( lcrp->lcpjds->val, lcrp->lcpjds->col, lcrp->lcpjds->colStart, lcrp->lcpjds->rowLen, lcrp->lcpjds->nRows, 
        lcrp->lcpjds->padding, invec->val_gpu, res->val_gpu );
    #endif
#endif

  #else
  
    #pragma omp parallel for schedule(runtime) private (hlp1, j)
    for (i=0; i<lcrp->lnRows[*me]; i++){
      hlp1 = 0.0;
      for (j=lcrp->lrow_ptr_l[i]; j<lcrp->lrow_ptr_l[i+1]; j++){
	      hlp1 = hlp1 + lcrp->lval[j] * invec->val[lcrp->lcol[j]]; 
      }
      res->val[i] = hlp1;
    }

  #endif

  IF_DEBUG(1){
     for_timing_stop_asm_( asm_cyclecounter, asm_cycles);
     *lc_cycles = *asm_cycles - *cycles4measurement; 
  }
}

/*********** kernel for remote entries only *********************/

inline void spmvmKernRemote( LCRP_TYPE* lcrp, VECTOR_TYPE* invec, VECTOR_TYPE* res,
                      uint64* asm_cyclecounter, uint64* asm_cycles, uint64* cycles4measurement, 
                      uint64* nl_cycles, uint64* cp_nlin_cycles, uint64* cp_res_cycles, 
                      int* me) {
  /* helper function to call either SpMVM kernel on device with device data transfer (if CUDAKERNEL) 
   * or OMP parallel kernel;
   * nl_cycles: timing measurement for computation of non-local entries
   * cp_nlin_cycles/cp_res_cycles: timing for copy to device of non-local elements in input (rhs) vector / 
   *   copy from device of result, only valid if CUDAKERNEL */

  int i, j;
  double hlp1;

  #ifdef CUDAKERNEL
  IF_DEBUG(1) for_timing_start_asm_( asm_cyclecounter);

   copyHostToDevice(invec->val_gpu+lcrp->lnRows[*me], invec->val+lcrp->lnRows[*me],
    lcrp->halo_elements*sizeof(double));
   
   IF_DEBUG(1){
      for_timing_stop_asm_( asm_cyclecounter, asm_cycles);
      *cp_nlin_cycles = *asm_cycles - *cycles4measurement; 
   }
  #endif
  
  IF_DEBUG(1) for_timing_start_asm_( asm_cyclecounter);
  
  #ifdef CUDAKERNEL
#ifdef ELR
    #ifdef TEXCACHE
      elrCudaKernelTexCacheAdd( lcrp->rcelr->val, lcrp->rcelr->col, lcrp->rcelr->rowLen, lcrp->rcelr->nRows, 
        lcrp->rcelr->padding, res->val_gpu );
    #else
      elrCudaKernelAdd( lcrp->rcelr->val, lcrp->rcelr->col, lcrp->rcelr->rowLen, lcrp->rcelr->nRows, 
        lcrp->rcelr->padding, invec->val_gpu, res->val_gpu );
    #endif
#else
    #ifdef TEXCACHE
      pjdsCudaKernelTexCacheAdd( lcrp->rcelr->val, lcrp->rcelr->col, lcrp->rcelr->rowLen, lcrp->rcelr->nRows, lcrp->rcelr->padding, res->val_gpu );
    #else
      elrCudaKernelAdd( lcrp->rcelr->val, lcrp->rcelr->col, lcrp->rcelr->rowLen, lcrp->rcelr->nRows, 
        lcrp->rcelr->padding, invec->val_gpu, res->val_gpu );
    #endif
#endif


  #else
  
    #pragma omp parallel for schedule(runtime) private (hlp1, j)
    for (i=0; i<lcrp->lnRows[*me]; i++){
      hlp1 = 0.0;
      for (j=lcrp->lrow_ptr_r[i]; j<lcrp->lrow_ptr_r[i+1]; j++){
	      hlp1 = hlp1 + lcrp->rval[j] * invec->val[lcrp->rcol[j]]; 
      }
      res->val[i] += hlp1;
    }

  #endif

   IF_DEBUG(1){
      for_timing_stop_asm_( asm_cyclecounter, asm_cycles);
      *nl_cycles = *asm_cycles - *cycles4measurement; 
   }
   
  #ifdef CUDAKERNEL
  IF_DEBUG(1) for_timing_start_asm_( asm_cyclecounter);
  
  copyDeviceToHost( res->val, res->val_gpu, res->nRows*sizeof(double) );
  
  IF_DEBUG(1){
      for_timing_stop_asm_( asm_cyclecounter, asm_cycles);
      *cp_res_cycles = *asm_cycles - *cycles4measurement; 
   }
  #endif
}


/*********** kernel for local entries only -- comm thread *********************/

#ifdef CUDAKERNEL

inline void spmvmKernLocalXThread( LCRP_TYPE* lcrp, VECTOR_TYPE* invec, VECTOR_TYPE* res,
                      uint64* asm_cyclecounter, uint64* asm_cycles, uint64* cycles4measurement, 
                      uint64* lc_cycles, uint64* cp_lin_cycles, int* me) {
  /* helper function to call SpMVM kernel only on device with device data transfer;
   * due to communication thread, OMP version must be called separately;
   * lc_cycles: timing measurement for computation of local entries
   * cp_lin_cycles: timing for copy to device of local elements in input (rhs) vector */

  IF_DEBUG(1) for_timing_start_asm_( asm_cyclecounter);
  
  copyHostToDevice(invec->val_gpu, invec->val, lcrp->lnRows[*me]*sizeof(double));

  IF_DEBUG(1){
     for_timing_stop_asm_( asm_cyclecounter, asm_cycles);
     *cp_lin_cycles = *asm_cycles - *cycles4measurement; 
  }
 
	IF_DEBUG(1) for_timing_start_asm_( asm_cyclecounter);

#ifdef ELR
    #ifdef TEXCACHE
      elrCudaKernelTexCache( lcrp->lcelr->val, lcrp->lcelr->col, lcrp->lcelr->rowLen, lcrp->lcelr->nRows, 
        lcrp->lcelr->padding, res->val_gpu );
    #else
      elrCudaKernel( lcrp->lcelr->val, lcrp->lcelr->col, lcrp->lcelr->rowLen, lcrp->lcelr->nRows, 
        lcrp->lcelr->padding, invec->val_gpu, res->val_gpu );
    #endif
#else
    #ifdef TEXCACHE
      pjdsCudaKernelTexCache( lcrp->lcpjds->val, lcrp->lcpjds->col, lcrp->lcpjds->colStart, lcrp->lcpjds->rowLen, lcrp->lcpjds->nRows, 
        lcrp->lcpjds->padding, res->val_gpu );
    #else
      pjdsCudaKernel( lcrp->lcpjds->val, lcrp->lcpjds->col, lcrp->lcpjds->colStart, lcrp->lcpjds->rowLen, lcrp->lcpjds->nRows, 
        lcrp->lcpjds->padding, invec->val_gpu, res->val_gpu );
    #endif
#endif


  IF_DEBUG(1){
     for_timing_stop_asm_( asm_cyclecounter, asm_cycles);
     *lc_cycles = *asm_cycles - *cycles4measurement; 
  }

}


/*********** kernel for remote entries only -- comm thread *********************/

inline void spmvmKernRemoteXThread( LCRP_TYPE* lcrp, VECTOR_TYPE* invec, VECTOR_TYPE* res,
                      uint64* asm_cyclecounter, uint64* asm_cycles, uint64* cycles4measurement, 
                      uint64* nl_cycles, uint64* cp_nlin_cycles, uint64* cp_res_cycles, 
                      int* me) {
  /* helper function to call SpMVM kernel only on device with device data transfer;
   * due to communication thread, OMP version must be called separately;
   * nl_cycles: timing measurement for computation of non-local entries
   * cp_nlin_cycles/cp_res_cycles: timing for copy to device of non-local elements in input (rhs) vector / 
   *   copy from device of result */

  IF_DEBUG(1) for_timing_start_asm_( asm_cyclecounter);

   copyHostToDevice(invec->val_gpu+lcrp->lnRows[*me], invec->val+lcrp->lnRows[*me],
    lcrp->halo_elements*sizeof(double));
   
   IF_DEBUG(1){
      for_timing_stop_asm_( asm_cyclecounter, asm_cycles);
      *cp_nlin_cycles = *asm_cycles - *cycles4measurement; 
   }
  
  IF_DEBUG(1) for_timing_start_asm_( asm_cyclecounter);
#ifdef ELR 
    #ifdef TEXCACHE
      elrCudaKernelTexCacheAdd( lcrp->rcelr->val, lcrp->rcelr->col, lcrp->rcelr->rowLen, lcrp->rcelr->nRows, 
        lcrp->rcelr->padding, res->val_gpu );
    #else
      elrCudaKernelAdd( lcrp->rcelr->val, lcrp->rcelr->col, lcrp->rcelr->rowLen, lcrp->rcelr->nRows, 
        lcrp->rcelr->padding, invec->val_gpu, res->val_gpu );
    #endif
#else
    #ifdef TEXCACHE
      pjdsCudaKernelTexCacheAdd( lcrp->rcelr->val, lcrp->rcelr->col, lcrp->rcelr->rowLen, lcrp->rcelr->nRows,lcrp->rcelr->padding, res->val_gpu );
    #else
      elrCudaKernelAdd( lcrp->rcelr->val, lcrp->rcelr->col, lcrp->rcelr->rowLen, lcrp->rcelr->nRows, 
        lcrp->rcelr->padding, invec->val_gpu, res->val_gpu );
    #endif
#endif

   IF_DEBUG(1){
      for_timing_stop_asm_( asm_cyclecounter, asm_cycles);
      *nl_cycles = *asm_cycles - *cycles4measurement; 
   }
   
  IF_DEBUG(1) for_timing_start_asm_( asm_cyclecounter);
  
  copyDeviceToHost( res->val, res->val_gpu, res->nRows*sizeof(double) );
  
  IF_DEBUG(1){
      for_timing_stop_asm_( asm_cyclecounter, asm_cycles);
      *cp_res_cycles = *asm_cycles - *cycles4measurement; 
   }

} 
#endif //CUDAKERNEL

#endif
