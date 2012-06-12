#ifndef _CUDA_MACROS_H_
#define _CUDA_MACROS_H_

#include "gpumacros.h"

#define safecall(call) do{\
  cudaError_t err = call ;\
  if( cudaSuccess != err ){\
    fprintf(stdout, "cuda error at %s:%d, %s\n",\
      __FILE__, __LINE__, cudaGetErrorString(err));\
    fflush(stdout);\
  }\
  } while(0)

#ifdef TEXCACHE
#define RHS(i) fetch_double(texRef,i)
#else
#define RHS(i) args.rhs[i]
#endif

#ifdef COLSTARTTC
#define COLSTART(i) tex1Dfetch(colStartTexRef,i)
#else
#define COLSTART(i) args.colStart[i]
#endif

#ifdef TEXCACHE
typedef struct{
	double* val;
	int* col;
	int* rowLen;
	int  N;
	int  pad;
	double*    resVec;
} ELRkernelArgs;
#ifdef COLSTARTTC
typedef struct{
	double* val;
	int* col;
	int* rowLen;
	int  N;
	int  pad;
	double*    resVec;
} pJDSkernelArgs;
#else
typedef struct{
	double* val;
	int* col;
	int* colStart;
	int* rowLen;
	int  N;
	int  pad;
	double*    resVec;
} PJDSkernelArgs;
#endif // COLSTARTTC
#else
typedef struct{
	double* val;
	int*    col;
	int*    rowLen;
	int     N;
	int     pad;
	double* rhs;
	double*       resVec;
} ELRkernelArgs;
#ifdef COLSTARTTC
typedef struct{
	double* val;
	int* col;
	int* rowLen;
	int  N;
	int  pad;
	double* rhs;
	double*    resVec;
} pJDSkernelArgs;
#else
typedef struct{
	double* val;
	int* col;
	int* colStart;
	int* rowLen;
	int  N;
	int  pad;
	double* rhs;
	double*    resVec;
} pJDSkernelArgs;
#endif // COLSTARTTC
#endif // TEXCACHE

#endif
