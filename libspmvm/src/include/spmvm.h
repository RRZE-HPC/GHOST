#ifndef _SPMVM_H_
#define _SPMVM_H_

#include <complex.h>
#include <mpi.h>
#include <math.h>

#ifdef OPENCL
#include <CL/cl.h>
#endif

#include "spmvm_type.h"

/******************************************************************************/
/*----  SpMVM kernels  -------------------------------------------------------*/
/******************************************************************************/
#define SPMVM_NUMKERNELS 4

#define SPMVM_KERNEL_NOMPI      (0x1<<0)
#define SPMVM_KERNEL_VECTORMODE (0x1<<1)
#define SPMVM_KERNEL_GOODFAITH  (0x1<<2)
#define SPMVM_KERNEL_TASKMODE   (0x1<<3)

#define SPMVM_KERNELS_COMBINED (SPMVM_KERNEL_NOMPI | SPMVM_KERNEL_VECTORMODE)
#define SPMVM_KERNELS_SPLIT    (SPMVM_KERNEL_GOODFAITH | SPMVM_KERNEL_TASKMODE)
#define SPMVM_KERNELS_ALL      (SPMVM_KERNELS_COMBINED | SPMVM_KERNELS_SPLIT)

#define SPMVM_KERNEL_IDX_FULL 0
#define SPMVM_KERNEL_IDX_LOCAL 1
#define SPMVM_KERNEL_IDX_REMOTE 2
/******************************************************************************/


/******************************************************************************/
/*----  Vector type  --------------------------------------------------------**/
/******************************************************************************/
#define VECTOR_TYPE_RHS 0
#define VECTOR_TYPE_LHS 1
#define VECTOR_TYPE_BOTH 2
/******************************************************************************/


/******************************************************************************/
/*----  GPU matrix formats  --------------------------------------------------*/
/******************************************************************************/
#define SPM_GPUFORMAT_ELR  0
#define SPM_GPUFORMAT_PJDS 1
#define PJDS_CHUNK_HEIGHT 32
extern const char *SPM_FORMAT_NAMES[];
/******************************************************************************/


/******************************************************************************/
/*----  Options for the SpMVM  -----------------------------------------------*/
/******************************************************************************/
#define SPMVM_OPTION_NONE       (0x1<<0) // no special options applied
#define SPMVM_OPTION_AXPY       (0x1<<1) // perform y = y+A*x instead of y = A*x
#define SPMVM_OPTION_KEEPRESULT (0x1<<2) // keep result on OpenCL device 
#define SPMVM_OPTION_RHSPRESENT (0x1<<3) // assume that RHS vector is present
#define SPMVM_OPTION_NO_COMBINED_KERNELS (0x1<<4) // not configure comb. kernels
#define SPMVM_OPTION_NO_SPLIT_KERNELS    (0x1<<5) // not configure split kernels
#define SPMVM_OPTION_NO_TASKMODE_KERNEL  (0x1<<6) // not configure task mode
#define SPMVM_OPTION_SERIAL_IO   (0x1<<7) // read matrix with one process only
//#define SPMVM_OPTION_PERMCOLS   (0x1<<3) // NOT SUPPORTED 
/******************************************************************************/


/******************************************************************************/
/*----  Available datatypes  -------------------------------------------------*/
/******************************************************************************/
#define DATATYPE_FLOAT 0
#define DATATYPE_DOUBLE 1
#define DATATYPE_COMPLEX_FLOAT 2
#define DATATYPE_COMPLEX_DOUBLE 3
extern const char *DATATYPE_NAMES[];
/******************************************************************************/

/******************************************************************************/
/*----  Available work distributions  ----------------------------------------*/
/******************************************************************************/
#define WORKDIST_EQUAL_ROWS 0 // equal number of rows for each process
#define WORKDIST_EQUAL_NZE  1 // equal number of nonzeros for each process 
#define WORKDIST_EQUAL_LNZE 2 // equal number of local nonzeros for each process
extern const char *WORKDIST_NAMES[];
/******************************************************************************/

#define IF_DEBUG(level) if( DEBUG >= level )


/******************************************************************************/
/*----  Global definitions  --------------------------------------------------*/
/******************************************************************************/
#define WORKDIST_DESIRED WORKDIST_EQUAL_ROWS
#define CL_MY_DEVICE_TYPE CL_DEVICE_TYPE_GPU
/******************************************************************************/


/******************************************************************************/
/*----  Consequences  --------------------------------------------------------*/
/******************************************************************************/
#ifdef LIKWID_MARKER
#define LIKWID
#endif
#ifdef LIKWID_MARKER_FINE
#define LIKWID
#endif
/******************************************************************************/


/******************************************************************************/
/*----  Definitions depending on datatype  -----------------------------------*/
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
/*----  Type definitions  ----------------------------------------------------*/
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
/*----  Function prototypes  -------------------------------------------------*/
/******************************************************************************/

/******************************************************************************
  * Initialize the basic functionality of the library. This includes:
  *   - initialize MPI
  *   - create and commit custom MPI datatypes (if necessary)
  *   - pin threads to CPU cores (if defined)
  *   - setup the MPI communicator for the node
  *   - initialize the OpenCL functionality of the library (if enabled)
  *   - initialize the Likwid Marker API (if defined)
  *
  * Arguments:
  *   - int argc
  *     The number of arguments of the main function (will be passed to
  *     MPI_init_thread())
  *   - char ** argv
  *     The arguments of the main functions (will be passed to 
  *     MPI_init_thread())
  *   - int options
  *     This argument contains the options for the sparse matrix-vector product.
  *     It can be assembled by OR-ing several of the available options which
  *     are defined as SPMVM_OPTION_* (explained above).
  *
  * Returns:
  *   an integer which holds the rank of the calling MPI process within
  *   MPI_COMM_WORLD
  *
  * The call to SpMVM_init() has to be done before any other SpMVM_*() call.
  *****************************************************************************/
int SpMVM_init(int argc, char **argv, int options);

/******************************************************************************
  * Clean up and finalize before termination. This includes:
  *   - call MPI_Finalize()
  *   - finish the OpenCL functionality
  *   - close the Likwid Marker API
  *
  * The SpMVM_finish() call is usually the last call of the main program. 
  *****************************************************************************/
void SpMVM_finish();

/******************************************************************************
  * Create a distributed CRS matrix from a given path. The matrix is read-in
  * from the processes in a parallel way (unless defined differently via
  * SPMVM_OPTION_SERIAL_IO) and necessary data structures for communication are
  * created. 
  * If OpenCL is enabled, the matrices are also converted into a GPU-friendly
  * format (as defined by the second argument) and uploaded to the device.
  *
  * Arguments:
  *   - char *matrixPath
  *     The full path to the matrix which is to be read. The matrix may either
  *     be present in the ASCII Matrix Market format as explained in
  *       http://math.nist.gov/MatrixMarket/formats.html
  *     or in a binary CRS format as explained in the README file.
  *   - void *deviceFormats
  *     If OpenCL is disabled, this argument has to be NULL.
  *     If OpenCL is enabled, this has to be a pointer to a SPM_GPUFORMATS
  *     structure as defined above.
  *
  * Returns:
  *   a pointer to an LCRP_TYPE structure which holds the local matrix data as
  *   well as the necessary data structures for communication.
  *****************************************************************************/
LCRP_TYPE * SpMVM_createCRS (char *matrixPath, void *deviceFormats);

/******************************************************************************
  * Create a distributed vector with specified values in order to use it for 
  * SpMVM. Depending on the type, the length of the vector may differ.
  * If OpenCL is enabled, the vector is also being created and initialized on
  * the device.
  *
  * Arguments:
  *   - LCRP_TYPE *lcrp
  *     The local CRS matrix portion to use with the vector.
  *   - int type
  *     Specifies whether the vector is a right hand side vector 
  *     (VECTOR_TYPE_RHS), left hand side vector (VECTOR_TYPE_LHS) or a vector
  *     which may be used as both right and left hand side (VECTOR_TYPE_BOTH).
  *     The length of the vector depends on this argument.
  *   - real (*fp)(int)
  *     A function pointer to a function taking an integer value and returning
  *     a real. This function returns the initial value for the i-th (globally)
  *     element of the vector.
  *     If NULL, the vector is initialized to zero.
  *
  * Returns:
  *   a pointer to an LCRP_TYPE structure which holds the local matrix data as
  *   well as the necessary data structures for communication.
  *****************************************************************************/
VECTOR_TYPE *SpMVM_createVector(LCRP_TYPE *lcrp, int type, real (*fp)(int));

/******************************************************************************
  * Perform the sparse matrix vector product using a specified kernel with a
  * fixed number of iterations.
  *
  * Arguments:
  *   - VECTOR_TYPE *res 
  *     The result vector. Its values are being accumulated if SPMVM_OPTION_AXPY
  *     is defined.  
  *   - LCRP_TYPE *lcrp
  *     The local CRS matrix part.
  *   - VECTOR_TYPE *invec
  *     The left hand side vector.
  *   - int kernel
  *     The kernel which should be used. This has to be one out of
  *       + SPMVM_KERNEL_NOMPI
  *       + SPMVM_KERNEL_VECTORMODE
  *       + SPMVM_KERNEL_GOODFAITH
  *       + SPMVM_KERNEL_TASKMODE
  *   - int nIter
  *     The number of iterations to run.
  *     
  * Returns:
  *   the wallclock time (in seconds) the kernel execution took. 
  *****************************************************************************/
double SpMVM_solve(VECTOR_TYPE *res, LCRP_TYPE *lcrp, VECTOR_TYPE *invec, 
		int kernel, int nIter);
/******************************************************************************/

#define DATA_T real

#endif
