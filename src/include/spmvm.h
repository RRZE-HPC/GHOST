#ifndef _SPMVM_H_
#define _SPMVM_H_

#include "spmvm_globals.h"

int               SpMVM_init(int argc, char **argv);
void              SpMVM_finish();
CR_TYPE *         SpMVM_createCRS (char *matrixPath);
LCRP_TYPE *       SpMVM_distributeCRS (CR_TYPE *cr, void *deviceFormats);
VECTOR_TYPE *     SpMVM_distributeVector(LCRP_TYPE *lcrp, HOSTVECTOR_TYPE *vec);
void              SpMVM_collectVectors(LCRP_TYPE *lcrp, VECTOR_TYPE *vec, HOSTVECTOR_TYPE *totalVec);

#endif
