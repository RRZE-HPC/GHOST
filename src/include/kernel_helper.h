#ifndef __KERNEL_HELPER_H__
#define __KERNEL_HELPER_H__



#ifdef OPENCL
#include "oclmacros.h"
#include "oclfun.h"
#endif

#ifdef LIKWID
#include <likwid.h>
#endif

#include <stdbool.h>
#include <complex.h>

extern int SPMVM_OPTIONS;

/*********** kernel for all entries *********************/

inline void spmvmKernAll( LCRP_TYPE* lcrp, VECTOR_TYPE* invec, VECTOR_TYPE* res,
		int* me) 
{
	/* helper function to call either SpMVM kernel on device with device data transfer (if CUDAKERNEL) 
	 * or OMP parallel kernel;
	 * ca_cycles: timing measurement for computation of all entries
	 * cp_in_cycles/cp_res_cycles: timing for copy to device of input (rhs) vector / copy from device
	 *   of result vector, only valid if CUDAKERNEL */


#ifdef OPENCL
	if (!(SPMVM_OPTIONS & SPMVM_OPTION_RHSPRESENT))
		CL_copyHostToDevice(invec->CL_val_gpu, invec->val, lcrp->lnRows[*me]*sizeof(real));
	
	CL_SpMVM(invec->CL_val_gpu,res->CL_val_gpu,SPM_KERNEL_FULL);
	
	if (!(SPMVM_OPTIONS & SPMVM_OPTION_KEEPRESULT))
		CL_copyDeviceToHost( res->val, res->CL_val_gpu, res->nRows*sizeof(real) );
#else
	int i, j;
	real hlp1;

#pragma omp parallel
	{
#ifdef LIKWID_MARKER_FINE
		likwid_markerStartRegion("full spmvm");
#endif
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
#ifdef LIKWID_MARKER_FINE
		likwid_markerStopRegion("full spmvm");
#endif
	}

#endif

}

/*********** kernel for local entries only *********************/

inline void spmvmKernLocal( LCRP_TYPE* lcrp, VECTOR_TYPE* invec, VECTOR_TYPE* res,
		int* me) {
	/* helper function to call either SpMVM kernel on device with device data transfer (if CUDAKERNEL) 
	 * or OMP parallel kernel;
	 * lc_cycles: timing measurement for computation of local entries
	 * cp_lin_cycles: timing for copy to device of local elements in input (rhs) vector, 
	 *   only valid if CUDAKERNEL */



#ifdef OPENCL
	if (!(SPMVM_OPTIONS & SPMVM_OPTION_RHSPRESENT))
		CL_copyHostToDevice(invec->CL_val_gpu, invec->val, lcrp->lnRows[*me]*sizeof(real));

	CL_SpMVM(invec->CL_val_gpu,res->CL_val_gpu,SPM_KERNEL_LOCAL);
#else
	int i, j;
	real hlp1;


#pragma omp parallel
	{
#ifdef LIKWID_MARKER_FINE
		likwid_markerStartRegion("local spmvm");
#endif
#pragma omp for schedule(runtime) private (hlp1, j)
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
#ifdef LIKWID_MARKER_FINE
		likwid_markerStopRegion("local spmvm");
#endif
	}

#endif

}

/*********** kernel for remote entries only *********************/

inline void spmvmKernRemote( LCRP_TYPE* lcrp, VECTOR_TYPE* invec, VECTOR_TYPE* res,
		int* me) {
	/* helper function to call either SpMVM kernel on device with device data transfer (if CUDAKERNEL) 
	 * or OMP parallel kernel;
	 * nl_cycles: timing measurement for computation of non-local entries
	 * cp_nlin_cycles/cp_res_cycles: timing for copy to device of non-local elements in input (rhs) vector / 
	 *   copy from device of result, only valid if CUDAKERNEL */


#ifdef OPENCL
	if (!(SPMVM_OPTIONS & SPMVM_OPTION_RHSPRESENT)) {
		CL_copyHostToDeviceOffset(invec->CL_val_gpu, 
				invec->val+lcrp->lnRows[*me], lcrp->halo_elements*sizeof(real), 
				lcrp->lnRows[*me]*sizeof(real));
	}
	
	CL_SpMVM(invec->CL_val_gpu,res->CL_val_gpu,SPM_KERNEL_REMOTE);
	
	if (!(SPMVM_OPTIONS & SPMVM_OPTION_KEEPRESULT)) {
		CL_copyDeviceToHost(res->val, res->CL_val_gpu, res->nRows*sizeof(real));
	}
#else
	int i, j;
	real hlp1;

#pragma omp parallel
	{
#ifdef LIKWID_MARKER_FINE
		likwid_markerStartRegion("remote spmvm");
#endif
#pragma omp for schedule(runtime) private (hlp1, j)
		for (i=0; i<lcrp->lnRows[*me]; i++){
			hlp1 = 0.0;
			for (j=lcrp->lrow_ptr_r[i]; j<lcrp->lrow_ptr_r[i+1]; j++){
				hlp1 = hlp1 + lcrp->rval[j] * invec->val[lcrp->rcol[j]]; 
			}
			res->val[i] += hlp1;
		}
#ifdef LIKWID_MARKER_FINE
		likwid_markerStopRegion("remote spmvm");
#endif

	}

#endif

}


/*********** kernel for local entries only -- comm thread *********************/

#ifdef OPENCL

inline void spmvmKernLocalXThread( LCRP_TYPE* lcrp, VECTOR_TYPE* invec, VECTOR_TYPE* res, int* me) 
{
	/* helper function to call SpMVM kernel only on device with device data transfer;
	 * due to communication thread, OMP version must be called separately;
	 * lc_cycles: timing measurement for computation of local entries
	 * cp_lin_cycles: timing for copy to device of local elements in input (rhs) vector */

	if (!(SPMVM_OPTIONS & SPMVM_OPTION_RHSPRESENT)) {
		CL_copyHostToDevice(invec->CL_val_gpu, invec->val, 
				lcrp->lnRows[*me]*sizeof(real));
	}

	CL_SpMVM(invec->CL_val_gpu,res->CL_val_gpu,SPM_KERNEL_LOCAL);
}


/*********** kernel for remote entries only -- comm thread *********************/

inline void spmvmKernRemoteXThread( LCRP_TYPE* lcrp, VECTOR_TYPE* invec, VECTOR_TYPE* res, int* me) 
{
	/* helper function to call SpMVM kernel only on device with device data transfer;
	 * due to communication thread, OMP version must be called separately;
	 * nl_cycles: timing measurement for computation of non-local entries
	 * cp_nlin_cycles/cp_res_cycles: timing for copy to device of non-local elements in input (rhs) vector / 
	 *   copy from device of result */

	if (!(SPMVM_OPTIONS & SPMVM_OPTION_RHSPRESENT)) {
		CL_copyHostToDeviceOffset(invec->CL_val_gpu, 
				invec->val+lcrp->lnRows[*me], lcrp->halo_elements*sizeof(real),
				lcrp->lnRows[*me]*sizeof(real));
	}


	CL_SpMVM(invec->CL_val_gpu,res->CL_val_gpu,SPM_KERNEL_REMOTE);

	if (!(SPMVM_OPTIONS & SPMVM_OPTION_KEEPRESULT))
		CL_copyDeviceToHost(res->val, res->CL_val_gpu, res->nRows*sizeof(real));
} 
#endif //CUDAKERNEL

#endif
