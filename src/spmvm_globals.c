#include "spmvm_globals.h"

const char *SPM_FORMAT_NAMES[]= {"ELR", "pJDS"};
const char *DATATYPE_NAMES[] = {"float","double","cmplx float","cmplx double"};
const char *WORKDIST_NAMES[] = {"equal rows","equal nze","equal lnze"};

Hybrid_kernel SPMVM_KERNELS[SPMVM_NUMKERNELS] = {

    { &hybrid_kernel_0,    0, 0, "SPMVM_KERNELS_0", "ca :\npure OpenMP-kernel" },

    { &hybrid_kernel_I,    0, 0, "SPMVM_KERNELS_I", "ir -- cs -- wa -- ca :\nISend/IRecv; \
		serial copy"},

    { &hybrid_kernel_II,    0, 0, "SPMVM_KERNELS_II", "ir -- cs -- cl -- wa -- nl :\
		\nISend/IRecv; good faith hybrid" },
 
    { &hybrid_kernel_III,  0, 0, "SPMVM_KERNELS_III", "ir -- lc|csw -- nl:\ncopy in \
		overlap region; dedicated comm-thread " },

}; 
