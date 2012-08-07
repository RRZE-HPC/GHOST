#ifndef _SPMVM_CONSTANTS_H_
#define _SPMVM_CONSTANTS_H_


/**********************************************/
/****** SpMVM kernels *************************/
/**********************************************/
#define SPMVM_NUMKERNELS 4

#define SPMVM_KERNEL_NOMPI      (0x1<<0)
#define SPMVM_KERNEL_VECTORMODE (0x1<<1)
#define SPMVM_KERNEL_GOODFAITH  (0x1<<2)
#define SPMVM_KERNEL_TASKMODE   (0x1<<3)

#define SPMVM_KERNELS_COMBINED (SPMVM_KERNEL_NOMPI | SPMVM_KERNEL_VECTORMODE)
#define SPMVM_KERNELS_SPLIT    (SPMVM_KERNEL_GOODFAITH | SPMVM_KERNEL_TASKMODE)
#define SPMVM_KERNELS_ALL      (SPMVM_KERNELS_COMBINED | SPMVM_KERNELS_SPLIT)

#define SPM_KERNEL_FULL 0
#define SPM_KERNEL_LOCAL 1
#define SPM_KERNEL_REMOTE 2
/**********************************************/


/**********************************************/
/****** GPU matrix formats ********************/
/**********************************************/
#define SPM_GPUFORMAT_ELR  0
#define SPM_GPUFORMAT_PJDS 1
#define PJDS_CHUNK_HEIGHT 32
extern const char *SPM_FORMAT_NAMES[];
/**********************************************/


/**********************************************/
/****** Options for the SpMVM *****************/
/**********************************************/
#define SPMVM_OPTION_NONE       (0x0)    // no special options applied
#define SPMVM_OPTION_AXPY       (0x1<<0) // perform y = y+A*x instead of y = A*x
#define SPMVM_OPTION_KEEPRESULT (0x1<<1) // keep result on OpenCL device 
#define SPMVM_OPTION_RHSPRESENT (0x1<<2) // assume that RHS vector is present
//#define SPMVM_OPTION_PERMCOLS   (0x1<<3) // NOT SUPPORTED 
/**********************************************/


/**********************************************/
/****** Available datatypes *******************/
/**********************************************/
#define DATATYPE_FLOAT 0
#define DATATYPE_DOUBLE 1
#define DATATYPE_COMPLEX_FLOAT 2
#define DATATYPE_COMPLEX_DOUBLE 3
extern const char *DATATYPE_NAMES[];
/**********************************************/


#define xMAIN_DIAGONAL_FIRST

#define EQUAL_NZE  1
#define EQUAL_LNZE 2

#define IF_DEBUG(level) if( DEBUG >= level )

#endif
