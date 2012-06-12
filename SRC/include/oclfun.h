#ifndef _OCLFUN_H_
#define _OCLFUN_H_

#include "oclmacros.h"
#include <stdlib.h>
#include <stdio.h>
#include <CL/cl.h>
#include <stdbool.h>
#include "my_ellpack.h"

#define NUM_KERNELS 4
#define KERNEL_ELR 0
#define KERNEL_ELR_ADD 1
#define KERNEL_PJDS 2
#define KERNEL_PJDS_ADD 3


void CL_selectDevice( int, int, const char* );

cl_mem CL_allocDeviceMemory( size_t );
void* allocHostMemory( size_t );
void CL_copyDeviceToHost( void*, cl_mem, size_t );
void CL_copyHostToDevice( cl_mem, void*, size_t );
void CL_copyHostToDeviceOffset( cl_mem, void*, size_t, size_t);
void CL_freeDeviceMemory( cl_mem );
void freeHostMemory( void* );
void CL_finish();

void oclKernel(void *,  cl_mem, cl_mem, bool, bool);

/*void elrCudaKernel( const double* , const int* , const int* , const int, const int, const double*, double* );
void elrCudaKernelTexCache( const double* , const int* , const int* , const int, const int, double* );

void elrCudaKernelAdd( const double* , const int* , const int* , const int, const int, const double*, double* );
void elrCudaKernelTexCacheAdd( const double* , const int* , const int* , const int, const int, double* );*/

#endif
