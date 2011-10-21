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
  _launcher_.gridDim  = gridDim;
  _launcher_.blockDim = blockDim;
}


/* *********** TEXTURE CACHE *************************** */

#ifdef TEXCACHE
texture<int2, 1, cudaReadModeElementType> texRef;

const textureReference* texRefPtr;

cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int2>();

extern "C" void bindTexRefToPtr() {
  safecall(cudaGetTextureReference( &texRefPtr, "texRef" ));
}

extern "C" void bindMemoryToTexCache( double* dblptr, int nElem ) {
  size_t bytesize = (size_t) sizeof(double) * nElem;
  safecall(cudaBindTexture( 0, texRefPtr, dblptr, &channelDesc, bytesize ));
  //safecall(cudaBindTexture( 0, texRef, dblptr, channelDesc, nElem ));
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
__global__ void __ELR_kernel_tex__(   const double* val, 
                                      const int* col, 
                                      const int* rowLen, 
                                      const int N, 
                                      const int pad, 
                                      double* resVec ) {
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


__global__ void __ELR_kernel_tex_add__(   const double* val, 
                                      const int* col, 
                                      const int* rowLen, 
                                      const int N, 
                                      const int pad, 
                                      double* resVec ) {
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


extern "C" void elrCudaKernelTexCache( const double* val,
                               const int* col, 
                               const int* rowLen,
                               const int N, 
                               const int pad,
                               double* resVec ) {

    __ELR_kernel_tex__ <<< _launcher_.gridDim, _launcher_.blockDim >>> ( val, col, rowLen, N, pad, resVec );

    safecall(cudaThreadSynchronize());
    safecall(cudaGetLastError());
}


extern "C" void elrCudaKernelTexCacheAdd( const double* val,
                               const int* col, 
                               const int* rowLen,
                               const int N, 
                               const int pad,
                               double* resVec ) {

    __ELR_kernel_tex_add__ <<< _launcher_.gridDim, _launcher_.blockDim >>> ( val, col, rowLen, N, pad, resVec );

    safecall(cudaThreadSynchronize());
    safecall(cudaGetLastError());
}
#endif

__global__ void __ELR_kernel__(   const double* val, 
                                      const int* col, 
                                      const int* rowLen, 
                                      const int N, 
                                      const int pad,
                                      const double* rhs,
                                      double* resVec ) {
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


extern "C" void elrCudaKernel( const double* val,
                               const int* col, 
                               const int* rowLen,
                               const int N, 
                               const int pad,
                               const double* rhs,
                               double* resVec ) {

    __ELR_kernel__ <<< _launcher_.gridDim, _launcher_.blockDim >>> ( val, col, rowLen, N, pad, rhs, resVec );

    safecall(cudaThreadSynchronize());
    safecall(cudaGetLastError());
}

__global__ void __ELR_kernel_add__(   const double* val, 
                                      const int* col, 
                                      const int* rowLen, 
                                      const int N, 
                                      const int pad,
                                      const double* rhs,
                                      double* resVec ) {
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


extern "C" void elrCudaKernelAdd( const double* val,
                               const int* col, 
                               const int* rowLen,
                               const int N, 
                               const int pad,
                               const double* rhs,
                               double* resVec ) {

    __ELR_kernel_add__ <<< _launcher_.gridDim, _launcher_.blockDim >>> ( val, col, rowLen, N, pad, rhs, resVec );

    safecall(cudaThreadSynchronize());
    safecall(cudaGetLastError());
}
