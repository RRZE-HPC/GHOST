#ifndef _KERNEL_H_
#define _KERNEL_H_

#include "spmvm.h"

void hybrid_kernel_0   (VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*, int);
void kern_glob_CRS_0(VECTOR_TYPE* res, CR_TYPE* cr, VECTOR_TYPE* invec, int spmvmOptions);
void mic_kernel_0      (VECTOR_TYPE*, BJDS_TYPE*, VECTOR_TYPE*, int);
void mic_kernel_0_unr      (VECTOR_TYPE*, BJDS_TYPE*, VECTOR_TYPE*, int);
void mic_kernel_0_intr      (VECTOR_TYPE*, BJDS_TYPE*, VECTOR_TYPE*, int);
#ifdef MPI
void hybrid_kernel_I   (VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*, int);
void hybrid_kernel_II  (VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*, int);
void hybrid_kernel_III (VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*, int);
#endif

#endif
