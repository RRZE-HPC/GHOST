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

#include <ghost_types.h>
#include <ghost_constants.h>

kernel void ELLPACK_kernel(global ghost_cl_mdat_t *lhs, global ghost_cl_mdat_t *rhsVec, int options, unsigned int nRows, unsigned int nRowsPadded, global unsigned int *rowLen, global unsigned int *col, global ghost_cl_mdat_t *val)
{ 
	unsigned int i = get_global_id(0);

	if (i < nRows) {
		ghost_cl_mdat_t tmp = 0.0, value = 0.0, rhs = 0.0; 
		unsigned int max = rowLen[i]; 
		int colidx;

		for(unsigned int j=0; j<max; ++j){ 
			value = val[i + j*nRowsPadded]; 
			colidx = col[i + j*nRowsPadded];
			rhs = rhsVec[colidx];
#ifdef GHOST_MAT_COMPLEX
			tmp.s0 += (value.s0 * rhs.s0 - value.s1 * rhs.s1);
			tmp.s1 += (value.s0 * rhs.s1 + value.s1 * rhs.s0);
#else
			tmp += value*rhs;
#endif
		}
		if (options & GHOST_OPTION_AXPY)
			lhs[i] += tmp;
		else 
			lhs[i] = tmp;
	}	
}

