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

extern "C" void prepareTexCacheRhs(double * rhsVec, size_t memSize) 
{
	safecall(cudaGetTextureReference( &texRefPtr, "texRef" ));
	safecall(cudaBindTexture( 0, texRefPtr, rhsVec, &channelDesc, memSize ));
}
#endif

#ifdef COLSTARTTC
texture<int, 1, cudaReadModeElementType> colStartTexRef;
const textureReference* colStartTexRefPtr;
cudaChannelFormatDesc colStartChannelDesc = cudaCreateChannelDesc<int>();

extern "C" void prepareTexCacheCS(int * colStartVec, size_t memSize) 
{
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

#ifdef TEXCACHE
static __inline__ __device__ double fetch_double(texture<int2, 1> t, int i)
{
	int2 v = tex1Dfetch(t,i);
	return __hiloint2double(v.y, v.x);
}
#endif

/* *********** KERNEL **************************** */
template<bool add> __global__ void __ELRkernel__(  ELRkernelArgs args ) {

	int idx, i, idcol, max;
	double svalue, value;

	for( idx = blockIdx.x * blockDim.x + threadIdx.x; idx < args.N; idx += gridDim.x * blockDim.x ) {
		svalue = 0.0;
		max = args.rowLen[idx];
		for( i = 0; i < max; ++i) {
			value = args.val[i*args.pad+idx];
			idcol = args.col[i*args.pad+idx];
			svalue += value * RHS(idcol);
		}
		if (add)
			args.resVec[idx] += svalue;
		else
			args.resVec[idx] = svalue;
	}
}
template<bool add> __global__ void __pJDSkernel__(  pJDSkernelArgs args ) {

	int idx, i, idcol, max;
	double svalue, value;

	for( idx = blockIdx.x * blockDim.x + threadIdx.x; idx < args.N; idx += gridDim.x * blockDim.x ) {
		svalue = 0.0;
		max = args.rowLen[idx];
		for( i = 0; i < max; ++i) {
			value = args.val[COLSTART(i)+idx];
			idcol = args.col[COLSTART(i)+idx];
			svalue += value * RHS(idcol);
		}
		if (add)
			args.resVec[idx] += svalue;
		else
			args.resVec[idx] = svalue;
	}
}

extern "C" void cudaKernel( void* args, bool add, bool elr) {

	if (elr) {
		if (add)
			__ELRkernel__<true> <<< _launcher_.gridDim, _launcher_.blockDim >>> ( *((ELRkernelArgs *)(args)) );
		else
			__ELRkernel__<false> <<< _launcher_.gridDim, _launcher_.blockDim >>> ( *((ELRkernelArgs *)(args)) );
	} else {
		if (add)
			__pJDSkernel__<true> <<< _launcher_.gridDim, _launcher_.blockDim >>> ( *((pJDSkernelArgs *)(args)) );
		else
			__pJDSkernel__<false> <<< _launcher_.gridDim, _launcher_.blockDim >>> ( *((pJDSkernelArgs *)(args)) );
	}

	safecall(cudaThreadSynchronize());
	safecall(cudaGetLastError());
}
