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


#define GHOST_PIN_PHYS   (16) /* pin threads to physical cores */
#define GHOST_PIN_SMT    (32) /* pin threads to _all_ cores */
#define GHOST_PIN_MANUAL (64) /* pin threads manually */


#define GHOST_DATAFORMAT_NAME_MAX 16


#define GHOST_IO_MPI (0)
#define GHOST_IO_STD (1)

#define GHOST_CONTEXT_DEFAULT       (0)
#define GHOST_CONTEXT_GLOBAL        (1)
#define GHOST_CONTEXT_DISTRIBUTED   (2)
#define GHOST_CONTEXT_WORKDIST_NZE  (4) /* distribute by # of nonzeros */
//#define GHOST_CONTEXT_WORKDIST_LNZE (8) /* distribute by # of loc nonzeros */
#define GHOST_CONTEXT_NO_COMBINED_SOLVERS (16) /* not configure comb. kernels */
#define GHOST_CONTEXT_NO_SPLIT_SOLVERS    (32) /* not configure split kernels */

#define GHOST_GET_DIM_FROM_MATRIX -1

#define GHOST_SPM_DEFAULT       (0)
#define GHOST_SPM_HOST          (1)
#define GHOST_SPM_DEVICE        (2)
#define GHOST_SPM_PERMUTECOLIDX (4)
#define GHOST_SPM_COLMAJOR      (8)
#define GHOST_SPM_ROWMAJOR      (16)
#define GHOST_SPM_SORTED        (32)

/******************************************************************************/
/*----  Vector type  --------------------------------------------------------**/
/******************************************************************************/
#define GHOST_VEC_DEFAULT   (0)
#define GHOST_VEC_RHS    (1)
#define GHOST_VEC_LHS    (2)
#define GHOST_VEC_HOST   (4)
#define GHOST_VEC_DEVICE (8)
#define GHOST_VEC_GLOBAL (16)
#define GHOST_VEC_DUMMY  (32)
/******************************************************************************/



/******************************************************************************/
/*----  Specific options for the SpMVM  --------------------------------------*/
/******************************************************************************/
#define GHOST_SPMVM_DEFAULT    (0)
#define GHOST_SPMVM_AXPY       (1) /* perform y = y+A*x instead of y = A*x */
#define GHOST_SPMVM_RHSPRESENT (2) /* assume that RHS vector is on device */
#define GHOST_SPMVM_KEEPRESULT (4) /* keep result on the device */
#define GHOST_SPMVM_MODE_NOMPI      (8)
#define GHOST_SPMVM_MODE_VECTORMODE (16)
#define GHOST_SPMVM_MODE_GOODFAITH  (32)
#define GHOST_SPMVM_MODE_TASKMODE   (64)
#define GHOST_SPMVM_APPLY_SHIFT     (128)
#define GHOST_SPMVM_APPLY_SCALE     (256)

#define GHOST_SPMVM_MODE_NOMPI_IDX      0
#define GHOST_SPMVM_MODE_VECTORMODE_IDX 1
#define GHOST_SPMVM_MODE_GOODFAITH_IDX  2
#define GHOST_SPMVM_MODE_TASKMODE_IDX   3

#define GHOST_SPMVM_MODES_COMBINED (GHOST_SPMVM_MODE_NOMPI | GHOST_SPMVM_MODE_VECTORMODE)
#define GHOST_SPMVM_MODES_SPLIT    (GHOST_SPMVM_MODE_GOODFAITH | GHOST_SPMVM_MODE_TASKMODE)
#define GHOST_SPMVM_MODES_ALL      (GHOST_SPMVM_MODES_COMBINED | GHOST_SPMVM_MODES_SPLIT)
/******************************************************************************/


#define GHOST_IMPL_C      (1)
#define GHOST_IMPL_SSE    (2)
#define GHOST_IMPL_AVX    (4)
#define GHOST_IMPL_MIC    (8)
#define GHOST_IMPL_OPENCL (16)


/******************************************************************************/
/*----  Constants for binary CRS format  -------------------------------------*/
/******************************************************************************/
#define GHOST_BINCRS_SIZE_HEADER 44 /* header consumes 44 bytes */
#define GHOST_BINCRS_SIZE_RPT_EL 8 /* one rpt element is 8 bytes */
#define GHOST_BINCRS_SIZE_COL_EL 8 /* one col element is 8 bytes */

#define GHOST_BINCRS_LITTLE_ENDIAN (0)

#define GHOST_BINCRS_SYMM_GENERAL        (1)
#define GHOST_BINCRS_SYMM_SYMMETRIC      (2)
#define GHOST_BINCRS_SYMM_SKEW_SYMMETRIC (4)
#define GHOST_BINCRS_SYMM_HERMITIAN      (8)

#define GHOST_BINCRS_DT_FLOAT   (1)
#define GHOST_BINCRS_DT_DOUBLE  (2)
#define GHOST_BINCRS_DT_REAL    (4)
#define GHOST_BINCRS_DT_COMPLEX (8)
/******************************************************************************/

#define GHOST_BINVEC_ORDER_COL_FIRST 0
#define GHOST_BINVEC_ORDER_ROW_FIRST 1

#define GHOST_DT_S_IDX 0
#define GHOST_DT_D_IDX 1
#define GHOST_DT_C_IDX 2
#define GHOST_DT_Z_IDX 3

//#define GHOST_THREAD_RUNNING 0
//#define GHOST_THREAD_HALTED 1
//#define GHOST_THREAD_RESERVED 2
//#define GHOST_THREAD_MGMT 3
//#define GHOST_TASK_ASYNC 1
//#define GHOST_TASK_SYNC 2
//#define GHOST_TASK_EXCLUSIVE 4

#define GHOST_SPM_FORMAT_CRS 0
#define GHOST_SPM_FORMAT_SELL 1

#define GHOST_SELL_CHUNKHEIGHT_ELLPACK 0
#define GHOST_SELL_CHUNKHEIGHT_AUTO -1
#define GHOST_SELL_SORT_GLOBALLY -1

#define GHOST_PAD_MAX 1024

#define GHOST_DATA_ALIGNMENT 1024

#define GHOST_GEMM_ALL_REDUCE -1
#define GHOST_GEMM_NO_REDUCE -2
#define GHOST_MAX_THREADS 8192

#endif
