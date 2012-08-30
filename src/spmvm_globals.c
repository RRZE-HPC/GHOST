
#include "kernel.h"


const char *SPM_FORMAT_NAMES[]= {"ELR", "pJDS"};
const char *DATATYPE_NAMES[] = {"float","double","cmplx float","cmplx double"};
const char *WORKDIST_NAMES[] = {"equal rows","equal nze","equal lnze"};
Hybrid_kernel SPMVM_KERNELS[SPMVM_NUMKERNELS] = {
    { &hybrid_kernel_0 },
    { &hybrid_kernel_I },
    { &hybrid_kernel_II },
    { &hybrid_kernel_III },
}; 
