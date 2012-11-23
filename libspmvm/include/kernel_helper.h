#ifndef __KERNEL_HELPER_H__
#define __KERNEL_HELPER_H__

#include "ghost.h"
#include "ghost_util.h"
#include "matricks.h"


#include <stdbool.h>
#include <complex.h>

/*********** kernel for all entries *********************/

static inline void spmvmKernAll( CR_TYPE* cr, ghost_vec_t* invec, ghost_vec_t* res, int spmvmOptions) 
{
/* helper function to call either SpMVM kernel on device with device data transfer (if CUDAKERNEL) 
 * or OMP parallel kernel;
 * ca_cycles: timing measurement for computation of all entries
 * cp_in_cycles/cp_res_cycles: timing for copy to device of input (rhs) vector / copy from device
 *   of result vector, only valid if CUDAKERNEL */


#ifdef OPENCL
/*if (!(spmvmOptions & GHOST_OPTION_RHSPRESENT)) {
	CL_copyHostToDevice(invec->CL_val_gpu, invec->val, (cr->lnrows[*me]+cr->halo_elements)*sizeof(mat_data_t));
}

CL_SpMVM(invec->CL_val_gpu,res->CL_val_gpu,GHOST_FULL_MAT_IDX);

if (!(spmvmOptions & GHOST_OPTION_KEEPRESULT))
	CL_copyDeviceToHost(res->val, res->CL_val_gpu, cr->lnrows[*me]*sizeof(mat_data_t));*/
#else

mat_idx_t i, j;
mat_data_t hlp1;

#pragma omp	parallel for schedule(runtime) private (hlp1, j)
	for (i=0; i<cr->nrows; i++){
		hlp1 = 0.0;
		for (j=cr->rpt[i]; j<cr->rpt[i+1]; j++){
			hlp1 = hlp1 + cr->val[j] * invec->val[cr->col[j]]; 
		}
		if (spmvmOptions & GHOST_OPTION_AXPY) 
			res->val[i] += hlp1;
		else
			res->val[i] = hlp1;
	}


#endif

}

/*********** kernel for local entries only -- comm thread *********************/

#ifdef OPENCL

static inline void spmvmKernLocalXThread( CR_TYPE* cr, ghost_vec_t* invec, ghost_vec_t* res, int* me, int spmvmOptions) 
{
/* helper function to call SpMVM kernel only on device with device data transfer;
 * due to communication thread, OMP version must be called separately;
 * lc_cycles: timing measurement for computation of local entries
 * cp_lin_cycles: timing for copy to device of local elements in input (rhs) vector */

/*if (!(spmvmOptions & GHOST_OPTION_RHSPRESENT)) {
	CL_copyHostToDevice(invec->CL_val_gpu, invec->val, 
			cr->lnrows[*me]*sizeof(mat_data_t));
}

CL_SpMVM(invec->CL_val_gpu,res->CL_val_gpu,GHOST_LOCAL_MAT_IDX);*/
}


/*********** kernel for remote entries only -- comm thread *********************/

static inline void spmvmKernRemoteXThread( CR_TYPE* cr, ghost_vec_t* invec, ghost_vec_t* res, int* me, int spmvmOptions) 
{
/* helper function to call SpMVM kernel only on device with device data transfer;
	 * due to communication thread, OMP version must be called separately;
	 * nl_cycles: timing measurement for computation of non-local entries
	 * cp_nlin_cycles/cp_res_cycles: timing for copy to device of non-local elements in input (rhs) vector / 
	 *   copy from device of result */

/*	if (!(spmvmOptions & GHOST_OPTION_RHSPRESENT)) {
		CL_copyHostToDeviceOffset(invec->CL_val_gpu, 
				invec->val+cr->lnrows[*me], cr->halo_elements*sizeof(mat_data_t),
				cr->lnrows[*me]*sizeof(mat_data_t));
	}


	CL_SpMVM(invec->CL_val_gpu,res->CL_val_gpu,GHOST_REMOTE_MAT_IDX);

	if (!(spmvmOptions & GHOST_OPTION_KEEPRESULT))
		CL_copyDeviceToHost(res->val, res->CL_val_gpu, cr->lnrows[*me]*sizeof(mat_data_t));*/
} 
#endif //OPENCL

#endif
