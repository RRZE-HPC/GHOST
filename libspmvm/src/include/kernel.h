#ifndef _KERNEL_H_
#define _KERNEL_H_

#include "spmvm.h"

void hybrid_kernel_0   (VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*, int);
void kern_glob_CRS_0(VECTOR_TYPE* res, CR_TYPE* cr, VECTOR_TYPE* invec, int spmvmOptions);

#ifdef MIC
void mic_kernel_0      (VECTOR_TYPE*, BJDS_TYPE*, VECTOR_TYPE*, int);
void mic_kernel_0_unr      (VECTOR_TYPE*, BJDS_TYPE*, VECTOR_TYPE*, int);
void mic_kernel_0_intr      (VECTOR_TYPE*, BJDS_TYPE*, VECTOR_TYPE*, int);
void mic_kernel_0_intr_16(VECTOR_TYPE*, BJDS_TYPE*, VECTOR_TYPE*, int);
void mic_kernel_0_intr_overlap      (VECTOR_TYPE*, BJDS_TYPE*, VECTOR_TYPE*, int);
#endif

#ifdef AVX
void avx_kernel_0_intr(VECTOR_TYPE* res, BJDS_TYPE* mv, VECTOR_TYPE* invec, int spmvmOptions);
void avx_kernel_0_intr_rem(VECTOR_TYPE* res, BJDS_TYPE* mv, VECTOR_TYPE* invec, int spmvmOptions);
#endif

#ifdef SSE
void sse_kernel_0_intr(VECTOR_TYPE* res, BJDS_TYPE* bjds, VECTOR_TYPE* invec, int spmvmOptions);
void sse_kernel_0_intr_rem(VECTOR_TYPE* res, BJDS_TYPE* bjds, VECTOR_TYPE* invec, int spmvmOptions);
#endif

#ifdef MPI
void hybrid_kernel_I   (VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*, int);
void hybrid_kernel_II  (VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*, int);
void hybrid_kernel_III (VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*, int);
#endif

#endif
