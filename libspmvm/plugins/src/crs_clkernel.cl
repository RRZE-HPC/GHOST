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
typedef double2 cl_mat_data_t;
#else
typedef double cl_mat_data_t;
#endif
#endif
#ifdef SINGLE
#ifdef COMPLEX
typedef float2 cl_mat_data_t;
#else
typedef float cl_mat_data_t;
#endif
#endif

kernel void CRS_kernel (global cl_mat_data_t *lhs, global cl_mat_data_t *rhs, int options, unsigned int nrows, global unsigned int *rpt, global unsigned int *col, global cl_mat_data_t *val) 
{
	unsigned int i = get_global_id(0);
	if (i < nrows) {
		cl_mat_data_t svalue = 0.0;
		for(unsigned int j=rpt[i]; j<rpt[i+1]; ++j){
			svalue += val[j] * rhs[col[j]]; 
		}
		lhs[i] += svalue;
	}
}

