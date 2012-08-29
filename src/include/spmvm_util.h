#ifndef _SPMVM_UTIL_H_
#define _SPMVM_UTIL_H_

#include "spmvm_globals.h"

#ifdef OPENCL
#include "spmvm_cl_util.h"
#endif

void              SpMVM_printMatrixInfo(LCRP_TYPE *lcrp, char *matrixName);
void              SpMVM_printEnvInfo();
HOSTVECTOR_TYPE * SpMVM_createGlobalHostVector(int nRows, real (*fp)(int));
void              SpMVM_referenceSolver(CR_TYPE *cr, real *rhs, real *lhs, int nIter);
int               SpMVM_kernelValid(int kernel, LCRP_TYPE *lcrp);
void              SpMVM_zeroVector(VECTOR_TYPE *vec);
HOSTVECTOR_TYPE*  SpMVM_newHostVector( const int nRows, real (*fp)(int));
VECTOR_TYPE*      SpMVM_newVector( const int nRows );
void              SpMVM_swapVectors(VECTOR_TYPE *v1, VECTOR_TYPE *v2);
void              SpMVM_normalize( real *vec, int nRows);

void SpMVM_freeVector( VECTOR_TYPE* const vec );
void SpMVM_freeHostVector( HOSTVECTOR_TYPE* const vec );
void SpMVM_freeCRS( CR_TYPE* const cr );
void SpMVM_freeLCRP( LCRP_TYPE* const );
void SpMVM_permuteVector( real* vec, int* perm, int len);
#endif
