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

#ifdef DOUBLE
#ifdef COMPLEX
typedef double2 clreal;
#else
typedef double clreal;
#endif
#endif
#ifdef SINGLE
#ifdef COMPLEX
typedef float2 clreal;
#else
typedef float clreal;
#endif
#endif

kernel void axpyKernel(global clreal *a, global clreal *b, clreal s, int nRows)
{
	int i = get_global_id(0); 
	if (i<nRows)
		a[i] += s*b[i]; 
}

kernel void vecscalKernel(global clreal *a, clreal scal, int nRows)
{
	int i = get_global_id(0);
	if (i<nRows)	
		a[i] *= scal; 
} 

kernel void dotprodKernel(global clreal *a, global clreal *b, global clreal *out, unsigned int nRows, local volatile clreal *shared) 
{

	unsigned int tid = get_local_id(0);
	unsigned int i = get_global_id(0);

#ifdef COMPLEX
	shared[tid].s0 = (i < nRows) ? (a[i].s0*b[i].s0 + a[i].s1*b[i].s1) : 0;
	shared[tid].s1 = (i < nRows) ? (-a[i].s0*b[i].s1 + a[i].s1*b[i].s0) : 0;
#else
	shared[tid] = (i < nRows) ? a[i]*b[i] : 0;
#endif

	barrier(CLK_LOCAL_MEM_FENCE);

	for (unsigned int s = 1; s < get_local_size(0); s *= 2) {
		if ((tid % (2*s)) == 0) {

			shared[tid] += shared[tid + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (tid == 0)
		out[get_group_id(0)] = shared[0];
}
