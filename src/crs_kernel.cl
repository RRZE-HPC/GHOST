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

kernel void CRS_kernel (global ghost_cl_vdat_t *lhs, global ghost_cl_vdat_t *rhsVec, int options, int nrows, global int *rpt, global int *col, global ghost_cl_mdat_t *val, ghost_cl_mdat_t shift) 
{
    unsigned int i = get_global_id(0);
    if (i < nrows) {
        ghost_cl_vdat_t tmp = 0.0, rhs = 0.0;
        ghost_cl_mdat_t value;
        for(int j=rpt[i]; j<rpt[i+1]; ++j) {
            rhs = rhsVec[col[j]];
            value = val[j]; 
            if (options & GHOST_SPMVM_APPLY_SHIFT)
                value += shift; 

#ifdef GHOST_MAT_COMPLEX
#ifdef GHOST_VEC_COMPLEX
            tmp.s0 += value.s0 * rhs.s0 - value.s1 * rhs.s1;
            tmp.s1 += value.s0 * rhs.s1 + value.s1 * rhs.s0;
#else
            tmp += value.s0 * rhs;
#endif
#endif
#ifdef GHOST_MAT_REAL
#ifdef GHOST_VEC_REAL
            tmp += value*rhs;
#else
            tmp.s0 += value * rhs.s0;
            tmp.s1 += value * rhs.s1;
#endif
#endif
        }
        if (options & GHOST_SPMVM_AXPY)
            lhs[i] += tmp;
        else 
            lhs[i] = tmp;
    }
}

