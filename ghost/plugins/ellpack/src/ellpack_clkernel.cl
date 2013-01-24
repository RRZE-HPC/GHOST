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

kernel void ELLPACK_kernel(global ghost_cl_vdat_t *lhs, global ghost_cl_vdat_t *rhsVec, int options, unsigned int nRows, unsigned int nRowsPadded, global unsigned int *rowLen, global unsigned int *col, global ghost_cl_mdat_t *val)
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
		if (options & GHOST_SPMVM_AXPY)
			lhs[i] += tmp;
		else 
			lhs[i] = tmp;
	}	
}

kernel void ELLPACKT_kernel(global ghost_cl_vdat_t *lhs, global ghost_cl_vdat_t *rhsVec, int options, unsigned int nRows, unsigned int nRowsPadded, global unsigned int *rowLen, global unsigned int *col, global ghost_cl_mdat_t *val, local ghost_cl_mdat_t *shared)
{ 
	unsigned int i = get_global_id(0);

	if (i < nRows) {
		ghost_cl_mdat_t tmp = 0.0, value = 0.0, rhs = 0.0;
		unsigned short idb = get_local_id(1);
		int colidx;

		for(unsigned int j=0; j<rowLen[i]/T; ++j){
		
			value = val[j*nRowsPadded*T + i + idb*nRowsPadded]; 
			colidx = col[j*nRowsPadded*T + i + idb*nRowsPadded];
			rhs = rhsVec[colidx];

#ifdef GHOST_MAT_COMPLEX
			tmp.s0 += (value.s0 * rhs.s0 - value.s1 * rhs.s1);
			tmp.s1 += (value.s0 * rhs.s1 + value.s1 * rhs.s0);
#else
			tmp += value*rhs;
#endif
		}
		
		shared[get_local_id(0)*T+get_local_id(1)] = tmp;
		barrier(CLK_LOCAL_MEM_FENCE);

#if T>2
#if T>4
#if T>8
#if T>16	
		if (idb<5)
			shared[get_local_id(0)*T]+=shared[get_local_id(0)*T+5];
#endif
		if (idb<4)
			shared[get_local_id(0)*T]+=shared[get_local_id(0)*T+4];
#endif
		if (idb<3)
			shared[get_local_id(0)*T]+=shared[get_local_id(0)*T+3];
#endif
		if (idb<2)
			shared[get_local_id(0)*T]+=shared[get_local_id(0)*T+2];
#endif

		if (idb==0) {
			if (options & GHOST_SPMVM_AXPY)
				lhs[i] += shared[get_local_id(0)*T]+shared[get_local_id(0)*T+1];
			else 
				lhs[i] = shared[get_local_id(0)*T]+shared[get_local_id(0)*T+1];
		}
	}	
}



