#ifndef _CUDA_MACROS_H_
#define _CUDA_MACROS_H_

#define safecall(call) do{\
  cudaError_t err = call ;\
  if( cudaSuccess != err ){\
    fprintf(stdout, "cuda error at %s:%d, %s\n",\
      __FILE__, __LINE__, cudaGetErrorString(err));\
    fflush(stdout);\
  }\
  } while(0)

#endif
