#ifndef _SPMVM_UTIL_H_
#define _SPMVM_UTIL_H_

#include "spmvm_globals.h"

void fortrancrsaxpyc_(int *, int *, real *, real *, real *, int *, int *);
void fortrancrsaxpy_(int *, int *, real *, real *, real *, int *, int *);
void fortrancrsaxpycf_(int *, int *, real *, real *, real *, int *, int *);
void fortrancrsaxpyf_(int *, int *, real *, real *, real *, int *, int *);
void fortrancrsc_(int *, int *, real *, real *, real *, int *, int *);
void fortrancrs_(int *, int *, real *, real *, real *, int *, int *);
void fortrancrscf_(int *, int *, real *, real *, real *, int *, int *);
void fortrancrsf_(int *, int *, real *, real *, real *, int *, int *);

void              SpMVM_printMatrixInfo(LCRP_TYPE *lcrp, char *matrixName);
void              SpMVM_printEnvInfo();
int               SpMVM_init(int argc, char **argv);
void              SpMVM_finish();
CR_TYPE *         SpMVM_createCRS (char *matrixPath);
LCRP_TYPE *       SpMVM_distributeCRS (CR_TYPE *cr);
VECTOR_TYPE *     SpMVM_distributeVector(LCRP_TYPE *lcrp, HOSTVECTOR_TYPE *vec);
void              SpMVM_collectVectors(LCRP_TYPE *lcrp, VECTOR_TYPE *vec, HOSTVECTOR_TYPE *totalVec);
HOSTVECTOR_TYPE * SpMVM_createGlobalHostVector(int nRows, real (*fp)(int));
void              SpMVM_referenceSolver(CR_TYPE *cr, real *rhs, real *lhs, int nIter);
int SpMVM_kernelValid(int kernel, LCRP_TYPE *lcrp);

#endif
