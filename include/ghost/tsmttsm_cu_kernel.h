/**
 * @file tsmm_cu_kernel.h
 * @brief TSMM CUDA kernels.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_TSMTTSM_CU_KERNEL_H
#define GHOST_TSMTTSM_CU_KERNEL_H

#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/cu_complex.h"

template<typename T,int M, int K, int conjv> 
static __global__ void ghost_tsmttsm_cu_cm_rm(T * const __restrict__ x, const T * const __restrict__ v, const T * const __restrict__ w, const T alpha, const T beta, ghost_lidx_t nrows, ghost_lidx_t stridex, ghost_lidx_t stridev, ghost_lidx_t stridew)
{

    // TODO

}

#endif

