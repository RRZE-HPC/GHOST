
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

#define BJDS_LEN 256 // TODO include header!!!

#ifdef DOUBLE
#ifdef COMPLEX
typedef double2 cl_ghost_mdat_t;
#else
typedef double cl_ghost_mdat_t;
#endif
#endif
#ifdef SINGLE
#ifdef COMPLEX
typedef float2 cl_ghost_mdat_t;
#else
typedef float cl_ghost_mdat_t;
#endif
#endif

kernel void BJDS_kernel(global cl_ghost_mdat_t *lhs, global cl_ghost_mdat_t *rhs, int options, unsigned int nRows, unsigned int nRowsPadded, global unsigned int *rowLen, global unsigned int *col, global cl_ghost_mdat_t *val, global unsigned int *chunkStart, global unsigned int *chunkLen)
{ 
	unsigned int i = get_global_id(0);

	if (i < nRows) {
		unsigned int cs = chunkStart[get_group_id(0)];
	//	unsigned int cl = chunkLen[get_group_id(0)];

		unsigned int li = get_local_id(0);

		cl_ghost_mdat_t svalue = 0.0, value = 0.0; 
		unsigned int max = rowLen[i]; 
		int colidx;

		for(unsigned int j=0; j<max; ++j){ 
			value = val[cs + li + j*BJDS_LEN]; 
			colidx = col[cs + li + j*BJDS_LEN]; 
			svalue += value * rhs[colidx];
		}
		lhs[i] += svalue;
	}	
}
