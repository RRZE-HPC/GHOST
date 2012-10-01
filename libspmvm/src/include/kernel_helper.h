#ifndef __KERNEL_HELPER_H__
#define __KERNEL_HELPER_H__

#include "spmvm.h"
#include "spmvm_util.h"


#include <stdbool.h>
#include <complex.h>

/*********** kernel for all entries *********************/

inline void spmvmKernAll( LCRP_TYPE* lcrp, VECTOR_TYPE* invec, VECTOR_TYPE* res,
		int* me, int spmvmOptions) 
{
	/* helper function to call either SpMVM kernel on device with device data transfer (if CUDAKERNEL) 
	 * or OMP parallel kernel;
	 * ca_cycles: timing measurement for computation of all entries
	 * cp_in_cycles/cp_res_cycles: timing for copy to device of input (rhs) vector / copy from device
	 *   of result vector, only valid if CUDAKERNEL */


#ifdef OPENCL
	if (!(spmvmOptions & SPMVM_OPTION_RHSPRESENT)) {
		CL_copyHostToDevice(invec->CL_val_gpu, invec->val, (lcrp->lnRows[*me]+lcrp->halo_elements)*sizeof(data_t));
	}

	CL_SpMVM(invec->CL_val_gpu,res->CL_val_gpu,SPMVM_KERNEL_IDX_FULL);
	
	if (!(spmvmOptions & SPMVM_OPTION_KEEPRESULT))
		CL_copyDeviceToHost(res->val, res->CL_val_gpu, lcrp->lnRows[*me]*sizeof(data_t));
#else

	int i, j;
	data_t hlp1;

#pragma omp	parallel for schedule(runtime) private (hlp1, j)
		for (i=0; i<lcrp->lnRows[*me]; i++){
			hlp1 = 0.0;
			for (j=lcrp->lrow_ptr[i]; j<lcrp->lrow_ptr[i+1]; j++){
				hlp1 = hlp1 + lcrp->val[j] * invec->val[lcrp->col[j]]; 
			}
			if (spmvmOptions & SPMVM_OPTION_AXPY) 
				res->val[i] += hlp1;
			else
				res->val[i] = hlp1;
		}


#endif

}

/*********** kernel for local entries only *********************/

inline void spmvmKernLocal( LCRP_TYPE* lcrp, VECTOR_TYPE* invec, VECTOR_TYPE* res,
		int* me, int spmvmOptions) {
	/* helper function to call either SpMVM kernel on device with device data transfer (if CUDAKERNEL) 
	 * or OMP parallel kernel;
	 * lc_cycles: timing measurement for computation of local entries
	 * cp_lin_cycles: timing for copy to device of local elements in input (rhs) vector, 
	 *   only valid if CUDAKERNEL */



#ifdef OPENCL
	if (!(spmvmOptions & SPMVM_OPTION_RHSPRESENT))
		CL_copyHostToDevice(invec->CL_val_gpu, invec->val, lcrp->lnRows[*me]*sizeof(data_t));

	CL_SpMVM(invec->CL_val_gpu,res->CL_val_gpu,SPMVM_KERNEL_IDX_LOCAL);
#else
	int i, j;
	data_t hlp1;


#pragma omp parallel for schedule(runtime) private (hlp1, j)
	for (i=0; i<lcrp->lnRows[*me]; i++){
		hlp1 = 0.0;
		for (j=lcrp->lrow_ptr_l[i]; j<lcrp->lrow_ptr_l[i+1]; j++){
			hlp1 = hlp1 + lcrp->lval[j] * invec->val[lcrp->lcol[j]]; 
		}
		if (spmvmOptions & SPMVM_OPTION_AXPY) 
			res->val[i] += hlp1;
		else
			res->val[i] = hlp1;
	}

#endif

}

/*********** kernel for remote entries only *********************/

inline void spmvmKernRemote( LCRP_TYPE* lcrp, VECTOR_TYPE* invec, VECTOR_TYPE* res,
		int* me
#ifdef OPENCL
		, int spmvmOptions
#endif
		) {
	/* helper function to call either SpMVM kernel on device with device data transfer (if CUDAKERNEL) 
	 * or OMP parallel kernel;
	 * nl_cycles: timing measurement for computation of non-local entries
	 * cp_nlin_cycles/cp_res_cycles: timing for copy to device of non-local elements in input (rhs) vector / 
	 *   copy from device of result, only valid if CUDAKERNEL */


#ifdef OPENCL
	if (!(spmvmOptions & SPMVM_OPTION_RHSPRESENT)) {
		CL_copyHostToDeviceOffset(invec->CL_val_gpu, 
				invec->val+lcrp->lnRows[*me], lcrp->halo_elements*sizeof(data_t), 
				lcrp->lnRows[*me]*sizeof(data_t));
	}
	
	CL_SpMVM(invec->CL_val_gpu,res->CL_val_gpu,SPMVM_KERNEL_IDX_REMOTE);
	
	if (!(spmvmOptions & SPMVM_OPTION_KEEPRESULT)) {
		CL_copyDeviceToHost(res->val, res->CL_val_gpu, lcrp->lnRows[*me]*sizeof(data_t));
	}
#else
	int i, j;
	data_t hlp1;

#pragma omp parallel for schedule(runtime) private (hlp1, j)
		for (i=0; i<lcrp->lnRows[*me]; i++){
			hlp1 = 0.0;
			for (j=lcrp->lrow_ptr_r[i]; j<lcrp->lrow_ptr_r[i+1]; j++){
				hlp1 = hlp1 + lcrp->rval[j] * invec->val[lcrp->rcol[j]]; 
			}
			res->val[i] += hlp1;
		}

#endif

}


/*********** kernel for local entries only -- comm thread *********************/

#ifdef OPENCL

inline void spmvmKernLocalXThread( LCRP_TYPE* lcrp, VECTOR_TYPE* invec, VECTOR_TYPE* res, int* me, int spmvmOptions) 
{
	/* helper function to call SpMVM kernel only on device with device data transfer;
	 * due to communication thread, OMP version must be called separately;
	 * lc_cycles: timing measurement for computation of local entries
	 * cp_lin_cycles: timing for copy to device of local elements in input (rhs) vector */

	if (!(spmvmOptions & SPMVM_OPTION_RHSPRESENT)) {
		CL_copyHostToDevice(invec->CL_val_gpu, invec->val, 
				lcrp->lnRows[*me]*sizeof(data_t));
	}

	CL_SpMVM(invec->CL_val_gpu,res->CL_val_gpu,SPMVM_KERNEL_IDX_LOCAL);
}


/*********** kernel for remote entries only -- comm thread *********************/

inline void spmvmKernRemoteXThread( LCRP_TYPE* lcrp, VECTOR_TYPE* invec, VECTOR_TYPE* res, int* me, int spmvmOptions) 
{
	/* helper function to call SpMVM kernel only on device with device data transfer;
	 * due to communication thread, OMP version must be called separately;
	 * nl_cycles: timing measurement for computation of non-local entries
	 * cp_nlin_cycles/cp_res_cycles: timing for copy to device of non-local elements in input (rhs) vector / 
	 *   copy from device of result */

	if (!(spmvmOptions & SPMVM_OPTION_RHSPRESENT)) {
		CL_copyHostToDeviceOffset(invec->CL_val_gpu, 
				invec->val+lcrp->lnRows[*me], lcrp->halo_elements*sizeof(data_t),
				lcrp->lnRows[*me]*sizeof(data_t));
	}


	CL_SpMVM(invec->CL_val_gpu,res->CL_val_gpu,SPMVM_KERNEL_IDX_REMOTE);

	if (!(spmvmOptions & SPMVM_OPTION_KEEPRESULT))
		CL_copyDeviceToHost(res->val, res->CL_val_gpu, lcrp->lnRows[*me]*sizeof(data_t));
} 
#endif //OPENCL

#endif
