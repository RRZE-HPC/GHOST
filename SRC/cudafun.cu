#include <stdio.h>
//#include <cuda_runtime_api.h>
#include "cudamacros.h"


/* *********** DEVICE SELECTION ************************* */

extern "C" void getDeviceInfo( int rank, int size, const char* hostname) {
	int deviceCount, device;
	cudaDeviceProp deviceProp;

	cudaGetDeviceCount(&deviceCount);

	if ( 0 == rank ) {
		printf("## rank %i/%i on %s --\t Device Test: No. Cards: %d\n", 
				rank, size-1, hostname, deviceCount);
		for( device = 0; device < deviceCount; ++device) {
			cudaGetDeviceProperties(&deviceProp, device);
			printf("## rank %i/%i on %s --\t Device %d: %s\n", 
					rank, size-1, hostname, device, deviceProp.name);
		}
	}
}


extern "C" int selectDevice( int rank, int size, const char* hostname ) {
	int deviceCount, takedevice, device;
	cudaDeviceProp deviceProp;

	cudaGetDeviceCount(&deviceCount);

	takedevice = (rank%deviceCount);
	cudaSetDevice(takedevice);
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&deviceProp, device);

	printf("rank %i/%i on %s --\t Selecting Device %d: %s\n", 
			rank, size-1, hostname, device, deviceProp.name);

	return device;
}


/* *********** KERNEL LAUNCH PARAMETERS ***************** */

typedef struct {
	int gridDim;
	int blockDim;
} KERNEL_LAUNCHER;

KERNEL_LAUNCHER _launcher_;

extern "C" void setKernelDims( const int gridDim, const int blockDim ) {

	/* set kernel launch parameters in global object _launcher_;
	 * _launcher_ used for all subsequent CUDA kernels */

	_launcher_.gridDim  = gridDim;
	_launcher_.blockDim = blockDim;
}


/* *********** TEXTURE CACHE *************************** */

#ifdef TEXCACHE
texture<int2, 1, cudaReadModeElementType> texRef;
const textureReference* texRefPtr;
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int2>();
extern "C" void prepareTexCacheRhs(double * rhsVec, size_t memSize) {
	safecall(cudaGetTextureReference( &texRefPtr, "texRef" ));
	safecall(cudaBindTexture( 0, texRefPtr, rhsVec, &channelDesc, memSize ));
}
#endif

#ifdef COLSTARTTC
texture<int, 1, cudaReadModeElementType> colStartTexRef;
const textureReference* colStartTexRefPtr;
cudaChannelFormatDesc colStartChannelDesc = cudaCreateChannelDesc<int>();
extern "C" void prepareTexCacheCS(double * colStartVec, size_t memSize) {
	safecall(cudaGetTextureReference( &colStartTexRefPtr, "colStartTexRef" ));
	safecall(cudaBindTexture( 0, colStartTexRefPtr, colStartVec, &colStartChannelDesc, memSize ));
}
#endif




/* *********** CUDA MEMORY **************************** */

extern "C" void* allocDeviceMemory( size_t bytesize ) {
	char* mem = NULL;
	safecall(cudaMalloc( (void**)&mem, bytesize ));

	return (void*)mem;
}

extern "C" void* allocHostMemory( size_t bytesize ) {
	char* mem = NULL;
	safecall(cudaHostAlloc( (void**)&mem, bytesize, 0 ));
	//mem = (char*) malloc( bytesize );
	//if( NULL == mem ) printf("failed to allocate %lu bytes of memory\n",bytesize);

	return (void*)mem;
}


extern "C" void copyDeviceToHost( void* hostmem, void* devmem, size_t bytesize ) {
	safecall(cudaMemcpy( hostmem, devmem, bytesize, cudaMemcpyDeviceToHost ));
}

extern "C" void copyHostToDevice( void* devmem, void* hostmem, size_t bytesize ) {
	safecall(cudaMemcpy( devmem, hostmem, bytesize, cudaMemcpyHostToDevice ));
}


extern "C" void freeDeviceMemory( void* mem ) {
	safecall(cudaFree( mem ));
}

extern "C" void freeHostMemory( void* mem ) {
	safecall(cudaFreeHost( mem ));
}


/* *********** KERNEL **************************** */

#ifdef TEXCACHE
__global__ void __pJDS_kernel_tex__(  const double* val, 
		const int* col, 
		const int* colStart,
		const int* rowLen, 
		const int N, 
		const int pad, 
		double* resVec ) {
	/* SpMVM kernel, pJDS format, texture cache */

	int idx, i, idcol, max;
	double svalue, value;
	int2 rhstmp;

	for( idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x ) {
		svalue = 0.0;
		max = rowLen[idx];
		for( i = 0; i < max; ++i) {
			value = val[colStart[i]+idx];
			idcol = col[colStart[i]+idx];
			rhstmp = tex1Dfetch(texRef, idcol);
			svalue += value * __hiloint2double(rhstmp.y,rhstmp.x);
		}
		resVec[idx] = svalue;
	}
}

__global__ void __ELR_kernel_tex__(   const double* val, 
		const int* col, 
		const int* rowLen, 
		const int N, 
		const int pad, 
		double* resVec ) {
	/* SpMVM kernel, ELR format, texture cache */

	int idx, i, idcol, max;
	double svalue, value;
	int2 rhstmp;

	for( idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x ) {
		svalue = 0.0;
		max = rowLen[idx];
		for( i = 0; i < max; ++i) {
			value = val[i*pad+idx];
			idcol = col[i*pad+idx];
			rhstmp = tex1Dfetch(texRef, idcol);
			svalue += value * __hiloint2double(rhstmp.y,rhstmp.x);
		}
		resVec[idx] = svalue;
	}
}


__global__ void __pJDS_kernel_tex_add__(   const double* val, 
		const int* col, 
		const int* colStart,
		const int* rowLen, 
		const int N, 
		const int pad, 
		double* resVec ) {
	/* SpMVM kernel, ELR format, Daxpy, texture cache */

	int idx, i, idcol, max;
	double svalue, value;
	int2 rhstmp;

	for( idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x ) {
		svalue = 0.0;
		max = rowLen[idx];
		for( i = 0; i < max; ++i) {
			value = val[colStart[i]+idx];
			idcol = col[colStart[i]+idx];
			rhstmp = tex1Dfetch(texRef, idcol);
			svalue += value * __hiloint2double(rhstmp.y,rhstmp.x);
		}
		resVec[idx] += svalue;
	}
}
__global__ void __ELR_kernel_tex_add__(   const double* val, 
		const int* col, 
		const int* rowLen, 
		const int N, 
		const int pad, 
		double* resVec ) {
	/* SpMVM kernel, ELR format, Daxpy, texture cache */

	int idx, i, idcol, max;
	double svalue, value;
	int2 rhstmp;

	for( idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x ) {
		svalue = 0.0;
		max = rowLen[idx];
		for( i = 0; i < max; ++i) {
			value = val[i*pad+idx];
			idcol = col[i*pad+idx];
			rhstmp = tex1Dfetch(texRef, idcol);
			svalue += value * __hiloint2double(rhstmp.y,rhstmp.x);
		}
		resVec[idx] += svalue;
	}
}

extern "C" void pjdsCudaKernelTexCache( const double* val,
		const int* col, 
		const int* colStart,
		const int* rowLen,
		const int N, 
		const int pad,
		double* resVec ) {
	/* SpMVM kernel wrapper, ELR, texture cache */

	__pJDS_kernel_tex__ <<< _launcher_.gridDim, _launcher_.blockDim >>> ( val, col, colStart, rowLen, N, pad, resVec );

	safecall(cudaThreadSynchronize());
	safecall(cudaGetLastError());
}

extern "C" void elrCudaKernelTexCache( const double* val,
		const int* col, 
		const int* rowLen,
		const int N, 
		const int pad,
		double* resVec ) {
	/* SpMVM kernel wrapper, ELR, texture cache */

	__ELR_kernel_tex__ <<< _launcher_.gridDim, _launcher_.blockDim >>> ( val, col, rowLen, N, pad, resVec );

	safecall(cudaThreadSynchronize());
	safecall(cudaGetLastError());
}

extern "C" void pjdsCudaKernelTexCacheAdd( const double* val,
		const int* col, 
		const int* colStart,
		const int* rowLen,
		const int N, 
		const int pad,
		double* resVec ) {
	/* SpMVM kernel wrapper, ELR, Daxpy, texture cache */

	__pJDS_kernel_tex_add__ <<< _launcher_.gridDim, _launcher_.blockDim >>> ( val, col, colStart, rowLen, N, pad, resVec );

	safecall(cudaThreadSynchronize());
	safecall(cudaGetLastError());
}

extern "C" void elrCudaKernelTexCacheAdd( const double* val,
		const int* col, 
		const int* rowLen,
		const int N, 
		const int pad,
		double* resVec ) {
	/* SpMVM kernel wrapper, ELR, Daxpy, texture cache */

	__ELR_kernel_tex_add__ <<< _launcher_.gridDim, _launcher_.blockDim >>> ( val, col, rowLen, N, pad, resVec );

	safecall(cudaThreadSynchronize());
	safecall(cudaGetLastError());
}
#endif

__global__ void __pJDS_kernel__(   const double* val, 
		const int* col, 
		const int* colStart,
		const int* rowLen, 
		const int N, 
		const int pad,
		const double* rhs,
		double* resVec ) {
	/* SpMVM kernel, ELR format */

	int idx, i, idcol, max;
	double svalue, value;

	for( idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x ) {
		svalue = 0.0;
		max = rowLen[idx];
		for( i = 0; i < max; ++i) {
#ifdef COLSTARTTC
			value = val[tex1Dfetch(colStartTexRef,i)+idx];
			idcol = col[tex1Dfetch(colStartTexRef,i)+idx];
#else
			value = val[colStart[i]+idx];
			idcol = col[colStart[i]+idx];
#endif
			svalue += value * rhs[idcol];
		}
		resVec[idx] = svalue;
	}
}
__global__ void __ELR_kernel__(   const double* val, 
		const int* col, 
		const int* rowLen, 
		const int N, 
		const int pad,
		const double* rhs,
		double* resVec ) {
	/* SpMVM kernel, ELR format */

	int idx, i, idcol, max;
	double svalue, value;

	for( idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x ) {
		svalue = 0.0;
		max = rowLen[idx];
		for( i = 0; i < max; ++i) {
			value = val[i*pad+idx];
			idcol = col[i*pad+idx];
			svalue += value * rhs[idcol];
		}
		resVec[idx] = svalue;
	}
}

extern "C" void pjdsCudaKernel( const double* val,
		const int* col, 
		const int* colStart,
		const int* rowLen,
		const int N, 
		const int pad,
		const double* rhs,
		double* resVec ) {
	/* SpMVM kernel wrapper, ELR format */

	__pJDS_kernel__ <<< _launcher_.gridDim, _launcher_.blockDim >>> ( val, col, colStart, rowLen, N, pad, rhs, resVec );

	safecall(cudaThreadSynchronize());
	safecall(cudaGetLastError());
}

extern "C" void elrCudaKernel( const double* val,
		const int* col, 
		const int* rowLen,
		const int N, 
		const int pad,
		const double* rhs,
		double* resVec ) {
	/* SpMVM kernel wrapper, ELR format */

	__ELR_kernel__ <<< _launcher_.gridDim, _launcher_.blockDim >>> ( val, col, rowLen, N, pad, rhs, resVec );

	safecall(cudaThreadSynchronize());
	safecall(cudaGetLastError());
}

__global__ void __pJDS_kernel_add__(   const double* val, 
		const int* col, 
		const int* colStart,
		const int* rowLen, 
		const int N, 
		const int pad,
		const double* rhs,
		double* resVec ) {
	/* SpMVM kernel, ELR format, Daxpy */

	int idx, i, idcol, max;
	double svalue, value;

	for( idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x ) {
		svalue = 0.0;
		max = rowLen[idx];
		for( i = 0; i < max; ++i) {
			value = val[colStart[i]+idx];
			idcol = col[colStart[i]+idx];
			svalue += value * rhs[idcol];
		}
		resVec[idx] += svalue;
	}
}

__global__ void __ELR_kernel_add__(   const double* val, 
		const int* col, 
		const int* rowLen, 
		const int N, 
		const int pad,
		const double* rhs,
		double* resVec ) {
	/* SpMVM kernel, ELR format, Daxpy */

	int idx, i, idcol, max;
	double svalue, value;

	for( idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x ) {
		svalue = 0.0;
		max = rowLen[idx];
		for( i = 0; i < max; ++i) {
			value = val[i*pad+idx];
			idcol = col[i*pad+idx];
			svalue += value * rhs[idcol];
		}
		resVec[idx] += svalue;
	}
}

extern "C" void pjdsCudaKernelAdd( const double* val,
		const int* col, 
		const int* colStart,
		const int* rowLen,
		const int N, 
		const int pad,
		const double* rhs,
		double* resVec ) {
	/* SpMVM kernel wrapper, ELR format, Daxpy */

	__pJDS_kernel_add__ <<< _launcher_.gridDim, _launcher_.blockDim >>> ( val, col, colStart, rowLen, N, pad, rhs, resVec );

	safecall(cudaThreadSynchronize());
	safecall(cudaGetLastError());
}

extern "C" void elrCudaKernelAdd( const double* val,
		const int* col, 
		const int* rowLen,
		const int N, 
		const int pad,
		const double* rhs,
		double* resVec ) {
	/* SpMVM kernel wrapper, ELR format, Daxpy */

	__ELR_kernel_add__ <<< _launcher_.gridDim, _launcher_.blockDim >>> ( val, col, rowLen, N, pad, rhs, resVec );

	safecall(cudaThreadSynchronize());
	safecall(cudaGetLastError());
}

