#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#if defined(cl_intel_printf)
#pragma OPENCL EXTENSION cl_intel_printf : enable
#elif defined(cl_amd_printf)
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif

#include <cl/ghost_cl_types.h>
#include <ghost_constants.h>

kernel void CRS_kernel (global ghost_cl_dt *lhs, global ghost_cl_dt *rhsVec, int options, int nrows, global int *rpt, global int *col, global ghost_cl_dt *val) 
{
	unsigned int i = get_global_id(0);
	if (i < nrows) {
		ghost_cl_dt tmp = 0.0, rhs = 0.0;
		for(unsigned int j=rpt[i]; j<rpt[i+1]; ++j) {
			rhs = rhsVec[col[j]];

#ifdef GHOST_MAT_COMPLEX
			tmp.s0 += (val[j].s0 * rhs.s0 - val[j].s1 * rhs.s1);
			tmp.s1 += (val[j].s0 * rhs.s1 + val[j].s1 * rhs.s0);
#else
			tmp += val[j]*rhs;
#endif
		}
		if (options & GHOST_SPMVM_AXPY)
			lhs[i] += tmp;
		else 
			lhs[i] = tmp;
	}
}

