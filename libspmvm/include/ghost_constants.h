#ifndef __GHOST_CONSTANTS_H__
#define __GHOST_CONSTANTS_H__

/******************************************************************************/
/*----  SpMVM modes  ---------------------------------------------------------*/
/******************************************************************************/
#define GHOST_NUM_MODES       (4)
#define GHOST_MODE_NOMPI      (0)
#define GHOST_MODE_VECTORMODE (1)
#define GHOST_MODE_GOODFAITH  (2)
#define GHOST_MODE_TASKMODE   (3)

#define GHOST_MODES_COMBINED (GHOST_MODE_NOMPI | GHOST_MODE_VECTORMODE)
#define GHOST_MODES_SPLIT    (GHOST_MODE_GOODFAITH | GHOST_MODE_TASKMODE)
#define GHOST_MODES_ALL      (GHOST_MODES_COMBINED | GHOST_MODES_SPLIT)

#define GHOST_FULL_MAT_IDX   0
#define GHOST_LOCAL_MAT_IDX  1
#define GHOST_REMOTE_MAT_IDX 2
/******************************************************************************/

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
/*----  Options for the SpMVM  -----------------------------------------------*/
/******************************************************************************/
#define GHOST_NUM_OPTIONS 10
#define GHOST_OPTION_NONE       (0x0)    // no special options applied
#define GHOST_OPTION_AXPY       (0x1<<0) // perform y = y+A*x instead of y = A*x
#define GHOST_OPTION_KEEPRESULT (0x1<<1) // keep result on OpenCL device 
#define GHOST_OPTION_RHSPRESENT (0x1<<2) // assume that RHS vector is present
#define GHOST_OPTION_NO_COMBINED_KERNELS (0x1<<3) // not configure comb. kernels
#define GHOST_OPTION_NO_SPLIT_KERNELS    (0x1<<4) // not configure split kernels
#define GHOST_OPTION_SERIAL_IO  (0x1<<5) // read matrix with one process only
#define GHOST_OPTION_PIN        (0x1<<6) // pin threads to physical cores
#define GHOST_OPTION_PIN_SMT    (0x1<<7) // pin threads to _all_ cores
#define GHOST_OPTION_WORKDIST_NZE   (0x1<<8) // distribute by # of nonzeros
#define GHOST_OPTION_WORKDIST_LNZE  (0x1<<9) // distribute by # of loc nonzeros
/******************************************************************************/

#define GHOST_IMPL_C      (0x1<<0)
#define GHOST_IMPL_SSE    (0x1<<1)
#define GHOST_IMPL_AVX    (0x1<<2)
#define GHOST_IMPL_MIC    (0x1<<3)
#define GHOST_IMPL_OPENCL (0x1<<4)

/******************************************************************************/
/*----  Available datatypes  -------------------------------------------------*/
/******************************************************************************/
//#define GHOST_DATATYPE_S 0
//#define GHOST_DATATYPE_D 1
//#define GHOST_DATATYPE_C 2
//#define GHOST_DATATYPE_Z 3
/******************************************************************************/


/******************************************************************************/
/*----  Constants for binary CRS format  -------------------------------------*/
/******************************************************************************/
#define GHOST_BINCRS_SIZE_HEADER 40 // header consumes 36 bytes
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
