#ifndef __KERNEL_HELPER_H__
#define __KERNEL_HELPER_H__

#include "mymacros.h"

#ifdef OPENCL
#include "oclmacros.h"
#include "oclfun.h"
#endif

#include <likwid.h>

#include <stdbool.h>

extern int SPMVM_OPTIONS;

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

#ifdef OPENCL
	if (!(SPMVM_OPTIONS & SPMVM_OPTION_RHSPRESENT)) {
		IF_DEBUG(1) for_timing_start_asm_( asm_cyclecounter);

		CL_copyHostToDevice(invec->CL_val_gpu, invec->val, invec->nRows*sizeof(double));
		IF_DEBUG(1){
			for_timing_stop_asm_( asm_cyclecounter, asm_cycles);
			*cp_in_cycles = *asm_cycles - *cycles4measurement; 
		}
	}
#endif

	IF_DEBUG(1) for_timing_start_asm_( asm_cyclecounter);

#ifdef OPENCL
	CL_SpMVM(invec->CL_val_gpu,res->CL_val_gpu,SPM_KERNEL_FULL);

#else

#pragma omp parallel
	{
   	likwid_markerStartRegion("full spmvm");
#pragma omp	for schedule(runtime) private (hlp1, j)
	for (i=0; i<lcrp->lnRows[*me]; i++){
		hlp1 = 0.0;
		for (j=lcrp->lrow_ptr[i]; j<lcrp->lrow_ptr[i+1]; j++){
			hlp1 = hlp1 + lcrp->val[j] * invec->val[lcrp->col[j]]; 
		}
		if (SPMVM_OPTIONS & SPMVM_OPTION_AXPY) 
			res->val[i] += hlp1;
		else
			res->val[i] = hlp1;
	}
   	likwid_markerStopRegion("full spmvm");

	}

#endif

	IF_DEBUG(1){
		for_timing_stop_asm_( asm_cyclecounter, asm_cycles);
		*ca_cycles = *asm_cycles - *cycles4measurement; 
	}

#ifdef OPENCL
	if (!(SPMVM_OPTIONS & SPMVM_OPTION_KEEPRESULT)) {
		IF_DEBUG(1) for_timing_start_asm_( asm_cyclecounter);


		CL_copyDeviceToHost( res->val, res->CL_val_gpu, res->nRows*sizeof(double) );
		IF_DEBUG(1){
			for_timing_stop_asm_( asm_cyclecounter, asm_cycles);
			*cp_res_cycles = *asm_cycles - *cycles4measurement; 
		}
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

#ifdef OPENCL
	if (!(SPMVM_OPTIONS & SPMVM_OPTION_RHSPRESENT)) {
		IF_DEBUG(1) for_timing_start_asm_( asm_cyclecounter);

		CL_copyHostToDevice(invec->CL_val_gpu, invec->val, lcrp->lnRows[*me]*sizeof(double));

		IF_DEBUG(1){
			for_timing_stop_asm_( asm_cyclecounter, asm_cycles);
			*cp_lin_cycles = *asm_cycles - *cycles4measurement; 
		}
	}

#endif
	IF_DEBUG(1) for_timing_start_asm_( asm_cyclecounter);
#ifdef OPENCL
	CL_SpMVM(invec->CL_val_gpu,res->CL_val_gpu,SPM_KERNEL_LOCAL);

#else


#pragma omp parallel for schedule(runtime) private (hlp1, j)
	for (i=0; i<lcrp->lnRows[*me]; i++){
		hlp1 = 0.0;
		for (j=lcrp->lrow_ptr_l[i]; j<lcrp->lrow_ptr_l[i+1]; j++){
			hlp1 = hlp1 + lcrp->lval[j] * invec->val[lcrp->lcol[j]]; 
		}
		if (SPMVM_OPTIONS & SPMVM_OPTION_AXPY) 
			res->val[i] += hlp1;
		else
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

#ifdef OPENCL
	if (!(SPMVM_OPTIONS & SPMVM_OPTION_RHSPRESENT)) {
		IF_DEBUG(1) for_timing_start_asm_( asm_cyclecounter);


		CL_copyHostToDeviceOffset(invec->CL_val_gpu, invec->val+lcrp->lnRows[*me], lcrp->halo_elements*sizeof(double), lcrp->lnRows[*me]*sizeof(double));


		IF_DEBUG(1){
			for_timing_stop_asm_( asm_cyclecounter, asm_cycles);
			*cp_nlin_cycles = *asm_cycles - *cycles4measurement; 
		}
	}
#endif

	IF_DEBUG(1) for_timing_start_asm_( asm_cyclecounter);

#ifdef OPENCL
	CL_SpMVM(invec->CL_val_gpu,res->CL_val_gpu,SPM_KERNEL_REMOTE);
#else

#pragma omp parallel
	{
	//likwid_markerStartRegion("remote spmvm");
#pragma omp for schedule(runtime) private (hlp1, j)
	for (i=0; i<lcrp->lnRows[*me]; i++){
		hlp1 = 0.0;
		for (j=lcrp->lrow_ptr_r[i]; j<lcrp->lrow_ptr_r[i+1]; j++){
			hlp1 = hlp1 + lcrp->rval[j] * invec->val[lcrp->rcol[j]]; 
		}
		res->val[i] += hlp1;
	}
	//likwid_markerStopRegion("remote spmvm");

	}

#endif

	IF_DEBUG(1){
		for_timing_stop_asm_( asm_cyclecounter, asm_cycles);
		*nl_cycles = *asm_cycles - *cycles4measurement; 
	}

#ifdef OPENCL
	if (!(SPMVM_OPTIONS & SPMVM_OPTION_KEEPRESULT)) {
		IF_DEBUG(1) for_timing_start_asm_( asm_cyclecounter);

		CL_copyDeviceToHost( res->val, res->CL_val_gpu, res->nRows*sizeof(double) );

		IF_DEBUG(1){
			for_timing_stop_asm_( asm_cyclecounter, asm_cycles);
			*cp_res_cycles = *asm_cycles - *cycles4measurement; 
		}
	}
#endif
}


/*********** kernel for local entries only -- comm thread *********************/

#ifdef OPENCL

inline void spmvmKernLocalXThread( LCRP_TYPE* lcrp, VECTOR_TYPE* invec, VECTOR_TYPE* res,
		uint64* asm_cyclecounter, uint64* asm_cycles, uint64* cycles4measurement, 
		uint64* lc_cycles, uint64* cp_lin_cycles, int* me) {
	/* helper function to call SpMVM kernel only on device with device data transfer;
	 * due to communication thread, OMP version must be called separately;
	 * lc_cycles: timing measurement for computation of local entries
	 * cp_lin_cycles: timing for copy to device of local elements in input (rhs) vector */

	if (!(SPMVM_OPTIONS & SPMVM_OPTION_RHSPRESENT)) {
		IF_DEBUG(1) for_timing_start_asm_( asm_cyclecounter);

		CL_copyHostToDevice(invec->CL_val_gpu, invec->val, lcrp->lnRows[*me]*sizeof(double));

		IF_DEBUG(1){
			for_timing_stop_asm_( asm_cyclecounter, asm_cycles);
			*cp_lin_cycles = *asm_cycles - *cycles4measurement; 
		}
	}

	IF_DEBUG(1) for_timing_start_asm_( asm_cyclecounter);

	CL_SpMVM(invec->CL_val_gpu,res->CL_val_gpu,SPM_KERNEL_LOCAL);

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

	if (!(SPMVM_OPTIONS & SPMVM_OPTION_RHSPRESENT)) {
		IF_DEBUG(1) for_timing_start_asm_( asm_cyclecounter);

		CL_copyHostToDeviceOffset(invec->CL_val_gpu, invec->val+lcrp->lnRows[*me], lcrp->halo_elements*sizeof(double),lcrp->lnRows[*me]*sizeof(double));

		IF_DEBUG(1){
			for_timing_stop_asm_( asm_cyclecounter, asm_cycles);
			*cp_nlin_cycles = *asm_cycles - *cycles4measurement; 
		}
	}

	IF_DEBUG(1) for_timing_start_asm_( asm_cyclecounter);

	CL_SpMVM(invec->CL_val_gpu,res->CL_val_gpu,SPM_KERNEL_REMOTE);

	IF_DEBUG(1){
		for_timing_stop_asm_( asm_cyclecounter, asm_cycles);
		*nl_cycles = *asm_cycles - *cycles4measurement; 
	}

	if (!(SPMVM_OPTIONS & SPMVM_OPTION_KEEPRESULT)) {
		IF_DEBUG(1) for_timing_start_asm_( asm_cyclecounter);

		CL_copyDeviceToHost( res->val, res->CL_val_gpu, res->nRows*sizeof(double) );

		IF_DEBUG(1){
			for_timing_stop_asm_( asm_cyclecounter, asm_cycles);
			*cp_res_cycles = *asm_cycles - *cycles4measurement; 
		}
	}

} 
#endif //CUDAKERNEL

inline void vecscal(VECTOR_TYPE *vec, double s) {

#ifdef OPENCL
	CL_vecscal(vec->CL_val_gpu,s,vec->nRows);
#else
	int i;
#pragma omp parallel
	{
	
//#ifdef LIKWID_MARKER
	likwid_markerStartRegion("vecscal");
//#endif

#pragma omp for private(i)
	for (i=0; i<vec->nRows; i++)
		vec->val[i] = s*vec->val[i];

//#ifdef LIKWID_MARKER
	likwid_markerStopRegion("vecscal");
//#endif
	}

#endif
}

inline void dotprod(VECTOR_TYPE *v1, VECTOR_TYPE *v2, double *res, int n) {

#ifdef OPENCL
	CL_dotprod(v1->CL_val_gpu,v2->CL_val_gpu,res,n);
#else
	int i;
	double sum = 0;
#pragma omp parallel 
	{
	
#ifdef LIKWID_MARKER
	likwid_markerStartRegion("dotprod");
#endif
#pragma omp for private(i) reduction(+:sum)
	for (i=0; i<n; i++)
		sum += v1->val[i]*v2->val[i];
#ifdef LIKWID_MARKER
	likwid_markerStopRegion("dotprod");
#endif

	}
	*res = sum;
#endif
}

inline void axpy(VECTOR_TYPE *v1, VECTOR_TYPE *v2, double s) {

#ifdef OPENCL
	CL_axpy(v1->CL_val_gpu,v2->CL_val_gpu,s,v1->nRows);
#else
	int i;
#pragma omp parallel for private(i)
	for (i=0; i<v1->nRows; i++)
		v1->val[i] = v1->val[i] + s*v2->val[i];
#endif
}



#endif
