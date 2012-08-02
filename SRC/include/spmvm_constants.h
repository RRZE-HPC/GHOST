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
/**********************************************/


/**********************************************/
/****** GPU matrix formats ********************/
/**********************************************/
#define SPM_GPUFORMAT_ELR  0
#define SPM_GPUFORMAT_PJDS 1
static char *SPM_FORMAT_NAME[]= {"ELR", "pJDS"};
/**********************************************/


/**********************************************/
/****** Options for the SpMVM *****************/
/**********************************************/
#define SPMVM_OPTION_NONE       (0x0)    // no special options applied
#define SPMVM_OPTION_AXPY       (0x1<<0) // perform y = y+A*x instead of y = A*x
#define SPMVM_OPTION_KEEPRESULT (0x1<<1) // keep result on OpenCL device 
#define SPMVM_OPTION_RHSPRESENT (0x1<<2) // assume that RHS vector is present
/**********************************************/


/**********************************************/
/****** Available datatypes *******************/
/**********************************************/
#define DATATYPE_FLOAT 0
#define DATATYPE_DOUBLE 1
#define DATATYPE_COMPLEX_FLOAT 2
#define DATATYPE_COMPLEX_DOUBLE 3
static char *DATATYPE_NAMES[] = {"float","double","cmplx float","cmplx double"};
/**********************************************/

#endif
