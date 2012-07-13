#ifndef _SPMVM_UTIL_H_
#define _SPMVM_UTIL_H_

#include "spmvm_globals.h"


void              SpMVM_printMatrixInfo(LCRP_TYPE *lcrp, char *matrixName);
int               SpMVM_init(int argc, char **argv);
CR_TYPE *         SpMVM_createCRS (char *matrixPath);
LCRP_TYPE *       SpMVM_distributeCRS (CR_TYPE *cr);
VECTOR_TYPE *     SpMVM_distributeVector(LCRP_TYPE *lcrp, HOSTVECTOR_TYPE *vec);
HOSTVECTOR_TYPE * SpMVM_createGlobalHostVector(int nRows, double (*fp)(int));

#endif
