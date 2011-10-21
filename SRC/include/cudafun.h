#ifndef _CUDAFUN_H_
#define _CUDAFUN_H_
#include <stdlib.h>

void getDeviceInfo( int, int, const char* );
int  selectDevice( int, int, const char* );

void setKernelDims( const int, const int );

#ifdef TEXCACHE
void bindTexRefToPtr();
void bindMemoryToTexCache( double*, int );
#endif

void* allocDeviceMemory( size_t );
void* allocHostMemory( size_t );
void copyDeviceToHost( void*, void*, size_t );
void copyHostToDevice( void*, void*, size_t );
void freeDeviceMemory( void* );
void freeHostMemory( void* );

void elrCudaKernel( const double* , const int* , const int* , const int, const int, const double*, double* );
void elrCudaKernelTexCache( const double* , const int* , const int* , const int, const int, double* );

void elrCudaKernelAdd( const double* , const int* , const int* , const int, const int, const double*, double* );
void elrCudaKernelTexCacheAdd( const double* , const int* , const int* , const int, const int, double* );

#endif
