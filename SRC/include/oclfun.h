#ifndef _OCLFUN_H_
#define _OCLFUN_H_

#include "oclmacros.h"
#include <stdlib.h>
#include <stdio.h>
#include <CL/cl.h>
#include <stdbool.h>
#include "my_ellpack.h"
#include "matricks.h"

#define SPM_KERNEL_FULL 0
#define SPM_KERNEL_LOCAL 1
#define SPM_KERNEL_REMOTE 2
#define AXPY_KERNEL 3
#define DOTPROD_KERNEL 4
#define VECSCAL_KERNEL 5


/*#define NUM_KERNELS 20
#define KERNEL_ELR 0
#define KERNEL_ELR_ADD 1
#define KERNEL_PJDS 2
#define KERNEL_PJDS_ADD 3

#define KERNEL_ELR2 4
#define KERNEL_ELR2_ADD 5
#define KERNEL_PJDS2 6
#define KERNEL_PJDS2_ADD 7

#define KERNEL_ELR4 8
#define KERNEL_ELR4_ADD 9
#define KERNEL_PJDS4 10
#define KERNEL_PJDS4_ADD 11

#define KERNEL_ELR8 12
#define KERNEL_ELR8_ADD 13
#define KERNEL_PJDS8 14
#define KERNEL_PJDS8_ADD 15

#define KERNEL_ELR16 16
#define KERNEL_ELR16_ADD 17
#define KERNEL_PJDS16 18
#define KERNEL_PJDS16_ADD 19*/

void CL_init( int, int, const char*, MATRIX_FORMATS *);
void CL_bindMatrixToKernel(void *mat, int format, int T, int kernelIdx);

cl_mem CL_allocDeviceMemory( size_t );
void* allocHostMemory( size_t );
void CL_copyDeviceToHost( void*, cl_mem, size_t );
void CL_copyHostToDevice( cl_mem, void*, size_t );
void CL_copyHostToDeviceOffset( cl_mem, void*, size_t, size_t);
void CL_freeDeviceMemory( cl_mem );
void freeHostMemory( void* );
void CL_finish();

void CL_SpMVM(cl_mem rhsVec, cl_mem resVec, int type); 
void CL_vecscal(cl_mem a, double s, int nRows);
void CL_axpy(cl_mem a, cl_mem b, double s, int nRows);
void CL_dotprod(cl_mem a, cl_mem b, double *out, int nRows);

#endif
