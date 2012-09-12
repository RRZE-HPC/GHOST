#ifndef _KERNEL_H_
#define _KERNEL_H_

#include "spmvm.h"

void hybrid_kernel_0   (VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*, int);
void hybrid_kernel_I   (VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*, int);
void hybrid_kernel_II  (VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*, int);
void hybrid_kernel_III (VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*, int);

#endif
