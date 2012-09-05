#ifndef _SPMVM_H_
#define _SPMVM_H_

#include <complex.h>
#include <mpi.h>
#include <math.h>

#ifdef OPENCL
#include <CL/cl.h>
#endif

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
#define SPMVM_OPTION_NONE       (0x1<<0)    // no special options applied
#define SPMVM_OPTION_AXPY       (0x1<<1) // perform y = y+A*x instead of y = A*x
#define SPMVM_OPTION_KEEPRESULT (0x1<<2) // keep result on OpenCL device 
#define SPMVM_OPTION_RHSPRESENT (0x1<<3) // assume that RHS vector is present
#define SPMVM_OPTION_NO_COMBINED_KERNELS (0x1<<4) 
#define SPMVM_OPTION_NO_SPLIT_KERNELS (0x1<<5)
#define SPMVM_OPTION_NO_TASKMODE_KERNEL (0x1<<6)
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

/**********************************************/
/****** Available work distributions **********/
/**********************************************/
#define WORKDIST_EQUAL_ROWS 0
#define WORKDIST_EQUAL_NZE  1
#define WORKDIST_EQUAL_LNZE 2
extern const char *WORKDIST_NAMES[];
/**********************************************/

#define IF_DEBUG(level) if( DEBUG >= level )


/******************************************************************************/
/****** Global definitions ****************************************************/
/******************************************************************************/
#define WORKDIST_DESIRED WORKDIST_EQUAL_ROWS
#define CL_MY_DEVICE_TYPE CL_DEVICE_TYPE_GPU
/******************************************************************************/


/******************************************************************************/
/****** Consequences **********************************************************/
/******************************************************************************/
#ifdef LIKWID_MARKER
#define LIKWID
#endif
#ifdef LIKWID_MARKER_FINE
#define LIKWID
#endif
/******************************************************************************/


/******************************************************************************/
/****** Definitions depending on datatype *************************************/
/******************************************************************************/
#ifdef DOUBLE
#ifdef COMPLEX
typedef _Complex double real;
MPI_Datatype MPI_MYDATATYPE;
MPI_Op MPI_MYSUM;
#define DATATYPE_DESIRED DATATYPE_COMPLEX_DOUBLE
#else // COMPLEX
typedef double real;
#define MPI_MYDATATYPE MPI_DOUBLE
#define MPI_MYSUM MPI_SUM
#define DATATYPE_DESIRED DATATYPE_DOUBLE
#endif // COMPLEX
#endif // DOUBLE

#ifdef SINGLE
#ifdef COMPLEX
typedef _Complex float real;
MPI_Datatype MPI_MYDATATYPE;
MPI_Op MPI_MYSUM;
#define DATATYPE_DESIRED DATATYPE_COMPLEX_FLOAT
#else // COMPLEX
typedef float real;
#define MPI_MYDATATYPE MPI_FLOAT
#define MPI_MYSUM MPI_SUM
#define DATATYPE_DESIRED DATATYPE_FLOAT
#endif // COMPLEX
#endif // SINGLE

#ifdef COMPLEX
#define FLOPS_PER_ENTRY 8.0
#else
#define FLOPS_PER_ENTRY 2.0
#endif

#ifdef DOUBLE
#ifdef COMPLEX
#define ABS(a) cabs(a)
#define REAL(a) creal(a)
#define IMAG(a) cimag(a)
#define SQRT(a) csqrt(a)
#else
#define ABS(a) fabs(a)
#define REAL(a) a
#define IMAG(a) 0.0
#define SQRT(a) sqrt(a)
#endif
#endif

#ifdef SINGLE
#ifdef COMPLEX
#define ABS(a) cabsf(a)
#define REAL(a) crealf(a)
#define IMAG(a) cimagf(a)
#define SQRT(a) csqrtf(a)
#else
#define ABS(a) fabsf(a)
#define REAL(a) a
#define IMAG(a) 0.0
#define SQRT(a) sqrtf(a)
#endif
#endif

#ifndef MIN
#define MIN(x,y) ((x)<(y)?(x):(y))
#endif
#ifndef MAX
#define MAX(x,y) ((x)<(y)?(y):(x))
#endif

#ifdef DOUBLE
#define EPSILON 1e-9
#endif
#ifdef SINGLE
#define EPSILON 1e-0
#endif
/******************************************************************************/


/******************************************************************************/
/****** Type definitions ******************************************************/
/******************************************************************************/
typedef struct{
	int nDistinctDevices;
	int *nDevices;
	char **names;
} CL_DEVICE_INFO;

typedef struct {
	int nRows;
	real* val;
#ifdef OPENCL
  cl_mem CL_val_gpu;
#endif
} VECTOR_TYPE;

typedef struct {
	int format[3];
	int T[3];
} SPM_GPUFORMATS;

typedef struct {
	int nRows;
	real* val;
} HOSTVECTOR_TYPE;

typedef struct {
  int nodes, threads, halo_elements;
  int nEnts, nRows;
  int* lnEnts;
  int* lnRows;
  int* lfEnt;
  int* lfRow;
  int* wishes;
  int* wishlist_mem;
  int** wishlist;
  int* dues;
  int* duelist_mem;
  int** duelist;
  int* due_displ;
  int* wish_displ;
  int* hput_pos;
  real* val;
  int* col;
  int* lrow_ptr;
  int* lrow_ptr_l;
  int* lrow_ptr_r;
  int* lcol;
  int* rcol;
  real* lval;
  real* rval;
  int fullFormat;
  int localFormat;
  int remoteFormat;
  int fullT;
  int localT;
  int remoteT;
  void *fullMatrix;
  void *localMatrix;
  void *remoteMatrix;
  int *fullRowPerm;     // may be NULL
  int *fullInvRowPerm;  // may be NULL
  int *splitRowPerm;    // may be NULL
  int *splitInvRowPerm; // may be NULL
} LCRP_TYPE;

typedef struct {
	int nRows, nCols, nEnts;
	int* rowOffset;
	int* col;
	real* val;
} CR_TYPE;

typedef void (*FuncPrototype)(VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*, int);

typedef struct {
    FuncPrototype kernel;
} Hybrid_kernel;
/******************************************************************************/


/******************************************************************************/
/****** Global variables ******************************************************/
/******************************************************************************/
/******************************************************************************/



/******************************************************************************/
/****** Function prototypes ***************************************************/
/******************************************************************************/
int SpMVM_init(int argc, char **argv, int options);
void SpMVM_finish();
CR_TYPE * SpMVM_createCRS (char *matrixPath);
LCRP_TYPE * SpMVM_distributeCRS (CR_TYPE *cr, void *deviceFormats);
VECTOR_TYPE * SpMVM_distributeVector(LCRP_TYPE *lcrp, HOSTVECTOR_TYPE *vec);
double SpMVM_solve(VECTOR_TYPE *res, LCRP_TYPE *lcrp, VECTOR_TYPE *invec, int kernel, int nIter);
void SpMVM_collectVectors(LCRP_TYPE *lcrp, VECTOR_TYPE *vec, 
		HOSTVECTOR_TYPE *totalVec, int kernel);
/******************************************************************************/

#endif
