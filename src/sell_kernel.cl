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


kernel void SELL_kernel(global ghost_cl_vdat_t *lhs, global ghost_cl_vdat_t *rhsVec, int options, unsigned int nRows, unsigned int nRowsPadded, global unsigned int *rowLen, global unsigned int *col, global ghost_cl_mdat_t *val, global unsigned int *chunkStart, global unsigned int *chunkLen, ghost_cl_mdat_t shift)
{ 
    unsigned int i = get_global_id(0);

    if (i < nRows) {
        unsigned int cs = chunkStart[get_group_id(0)];
        unsigned int li = get_local_id(0);

        ghost_cl_vdat_t tmp = 0.0, rhs = 0.0;
        ghost_cl_mdat_t value = 0.0;

        unsigned int max = rowLen[i]; 
        int colidx;

        for(unsigned int j=0; j<max; ++j){ 
            colidx = col[cs + li + j*SELL_LEN];
            rhs = rhsVec[colidx];
            value = val[cs + li + j*SELL_LEN]; 
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
