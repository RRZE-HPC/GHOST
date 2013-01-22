#ifndef __GHOST_CONSTANTS_H__
#define __GHOST_CONSTANTS_H__

/******************************************************************************/
/*----  SpMVM modes  ---------------------------------------------------------*/
/******************************************************************************/
#define GHOST_NUM_MODES       (4)

#define GHOST_FULL_MAT_IDX   0
#define GHOST_LOCAL_MAT_IDX  1
#define GHOST_REMOTE_MAT_IDX 2
/******************************************************************************/

#define GHOST_SUCCESS 0
#define GHOST_FAILURE 1

#define GHOST_CORENUMBERING_PHYSICAL_FIRST 0
#define GHOST_CORENUMBERING_SMT_FIRST 1
#define GHOST_CORENUMBERING_INVALID 2


#define GHOST_DATAFORMAT_NAME_MAX 16


#define GHOST_IO_MPI (0)
#define GHOST_IO_STD (1)

#define GHOST_CONTEXT_DEFAULT       (0)
#define GHOST_CONTEXT_GLOBAL        (0x1<<0)
#define GHOST_CONTEXT_DISTRIBUTED   (0x1<<1)

#define GHOST_SPM_DEFAULT       (0)
#define GHOST_SPM_HOST          (0x1<<0)
#define GHOST_SPM_DEVICE        (0x1<<1)
#define GHOST_SPM_PERMUTECOLIDX (0x1<<2)
#define GHOST_SPM_COLMAJOR      (0x1<<3)
#define GHOST_SPM_ROWMAJOR      (0x1<<4)
#define GHOST_SPM_SORTED        (0x1<<5)

/******************************************************************************/
/*----  Vector type  --------------------------------------------------------**/
/******************************************************************************/
#define GHOST_VEC_DEFAULT   (0)
#define GHOST_VEC_RHS    (0x1<<0)
#define GHOST_VEC_LHS    (0x1<<1)
#define GHOST_VEC_HOST   (0x1<<2)
#define GHOST_VEC_DEVICE (0x1<<3)
#define GHOST_VEC_GLOBAL (0x1<<4)
/******************************************************************************/


/******************************************************************************/
/*----  Options for ghost  ---------------------------------------------------*/
/******************************************************************************/
#define GHOST_OPTION_NONE       (0x0)    // no special options applied
#define GHOST_OPTION_NO_COMBINED_SOLVERS (0x1<<1) // not configure comb. kernels
#define GHOST_OPTION_NO_SPLIT_SOLVERS    (0x1<<2) // not configure split kernels
#define GHOST_OPTION_SERIAL_IO  (0x1<<3) // read matrix with one process only
#define GHOST_OPTION_PIN        (0x1<<4) // pin threads to physical cores
#define GHOST_OPTION_PIN_SMT    (0x1<<5) // pin threads to _all_ cores
#define GHOST_OPTION_WORKDIST_NZE   (0x1<<6) // distribute by # of nonzeros
#define GHOST_OPTION_WORKDIST_LNZE  (0x1<<7) // distribute by # of loc nonzeros
/******************************************************************************/


/******************************************************************************/
/*----  Specific options for the SpMVM  --------------------------------------*/
/******************************************************************************/
#define GHOST_SPMVM_DEFAULT    (0)
#define GHOST_SPMVM_AXPY       (0x1<<0) // perform y = y+A*x instead of y = A*x
#define GHOST_SPMVM_RHSPRESENT (0x1<<1) // assume that RHS vector is on device
#define GHOST_SPMVM_KEEPRESULT (0x1<<2) // keep result on the device
#define GHOST_SPMVM_MODE_NOMPI      (0x1<<3)
#define GHOST_SPMVM_MODE_VECTORMODE (0x1<<4)
#define GHOST_SPMVM_MODE_GOODFAITH  (0x1<<5)
#define GHOST_SPMVM_MODE_TASKMODE   (0x1<<6)

#define GHOST_SPMVM_MODE_NOMPI_IDX      0
#define GHOST_SPMVM_MODE_VECTORMODE_IDX 1
#define GHOST_SPMVM_MODE_GOODFAITH_IDX  2
#define GHOST_SPMVM_MODE_TASKMODE_IDX   3

#define GHOST_SPMVM_MODES_COMBINED (GHOST_SPMVM_MODE_NOMPI | GHOST_SPMVM_MODE_VECTORMODE)
#define GHOST_SPMVM_MODES_SPLIT    (GHOST_SPMVM_MODE_GOODFAITH | GHOST_SPMVM_MODE_TASKMODE)
#define GHOST_SPMVM_MODES_ALL      (GHOST_SPMVM_MODES_COMBINED | GHOST_SPMVM_MODES_SPLIT)
/******************************************************************************/


#define GHOST_IMPL_C      (0x1<<0)
#define GHOST_IMPL_SSE    (0x1<<1)
#define GHOST_IMPL_AVX    (0x1<<2)
#define GHOST_IMPL_MIC    (0x1<<3)
#define GHOST_IMPL_OPENCL (0x1<<4)


/******************************************************************************/
/*----  Constants for binary CRS format  -------------------------------------*/
/******************************************************************************/
#define GHOST_BINCRS_SIZE_HEADER 44 // header consumes 44 bytes
#define GHOST_BINCRS_SIZE_RPT_EL 8 // one rpt element is 8 bytes
#define GHOST_BINCRS_SIZE_COL_EL 8 // one col element is 8 bytes

#define GHOST_BINCRS_LITTLE_ENDIAN (0)

#define GHOST_BINCRS_SYMM_GENERAL        (0x1<<0)
#define GHOST_BINCRS_SYMM_SYMMETRIC      (0x1<<1)
#define GHOST_BINCRS_SYMM_SKEW_SYMMETRIC (0x1<<2)
#define GHOST_BINCRS_SYMM_HERMITIAN      (0x1<<3)

#define GHOST_BINCRS_DT_FLOAT   (0x1<<0)
#define GHOST_BINCRS_DT_DOUBLE  (0x1<<1)
#define GHOST_BINCRS_DT_REAL    (0x1<<2)
#define GHOST_BINCRS_DT_COMPLEX (0x1<<3)
/******************************************************************************/


#endif
