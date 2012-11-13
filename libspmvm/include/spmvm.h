#ifndef _SPMVM_H_
#define _SPMVM_H_

#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <inttypes.h>

#ifdef OPENCL
#include <CL/cl.h>
#endif

#include "spmvm_type.h"

#define LIBSPMVM_VERSION "0.1a"

/******************************************************************************/
/*----  SpMVM kernels  -------------------------------------------------------*/
/******************************************************************************/
#define SPMVM_NUMKERNELS 4

#define SPMVM_KERNEL_NOMPI      (0)
#define SPMVM_KERNEL_VECTORMODE (1)
#define SPMVM_KERNEL_GOODFAITH  (2)
#define SPMVM_KERNEL_TASKMODE   (3)

#define SPMVM_KERNELS_COMBINED (SPMVM_KERNEL_NOMPI | SPMVM_KERNEL_VECTORMODE)
#define SPMVM_KERNELS_SPLIT    (SPMVM_KERNEL_GOODFAITH | SPMVM_KERNEL_TASKMODE)
#define SPMVM_KERNELS_ALL      (SPMVM_KERNELS_COMBINED | SPMVM_KERNELS_SPLIT)

#define SPMVM_KERNEL_IDX_FULL 0
#define SPMVM_KERNEL_IDX_LOCAL 1
#define SPMVM_KERNEL_IDX_REMOTE 2
/******************************************************************************/

typedef unsigned short mat_format_t;
typedef unsigned short mat_flags_t;
typedef void * mat_aux_t;

typedef struct
{
	mat_format_t format;
	mat_flags_t flags;
	mat_aux_t aux;
} 
mat_trait_t;


typedef unsigned int setup_flags_t;
// formats
#define SPM_NUMFORMATS 7
#define SPM_FORMAT_NONE   (0)
#define SPM_FORMAT_CRS    (1)
#define SPM_FORMAT_BJDS   (2)
#define SPM_FORMAT_SBJDS  (3)
#define SPM_FORMAT_TBJDS  (4)
#define SPM_FORMAT_STBJDS (5)
//#define SPM_FORMAT_TCBJDS (6)
#define SPM_FORMAT_CRSCD  (6)

// TODO sorting as part of trait and not own format

// flags
#define SETUP_DEFAULT       (0)
#define SETUP_HOSTONLY      (0x1<<0)
#define SETUP_DEVICEONLY    (0x1<<1)
#define SETUP_HOSTANDDEVICE (0x1<<2)
#define SETUP_GLOBAL        (0x1<<3)
#define SETUP_DISTRIBUTED   (0x1<<4)

#define SPM_DEFAULT       (0)
#define SPM_PERMUTECOLIDX (0x1<<0)
#define SPM_COLMAJOR      (0x1<<1)
#define SPM_ROWMAJOR      (0x1<<2)

#ifdef MIC
//#define BJDS_LEN 8
#define BJDS_LEN 16
#elif defined (AVX)
#define BJDS_LEN 4 // TODO single/double precision
#elif defined (SSE)
#define BJDS_LEN 2
#else
#define BJDS_LEN 1
#endif

/******************************************************************************/
/*----  Vector type  --------------------------------------------------------**/
/******************************************************************************/
#define VECTOR_TYPE_RHS (0x1<<0)
#define VECTOR_TYPE_LHS (0x1<<1)
#define VECTOR_TYPE_BOTH (VECTOR_TYPE_RHS | VECTOR_TYPE_LHS)
#define VECTOR_TYPE_HOSTONLY (0x1<<2)
/******************************************************************************/


/******************************************************************************/
/*----  GPU matrix formats  --------------------------------------------------*/
/******************************************************************************/
#define SPM_GPUFORMAT_ELR  0
#define SPM_GPUFORMAT_PJDS 1
#define PJDS_CHUNK_HEIGHT 32
#define ELR_PADDING 1024
extern const char *SPM_FORMAT_NAMES[];
/******************************************************************************/


/******************************************************************************/
/*----  Options for the SpMVM  -----------------------------------------------*/
/******************************************************************************/
#define SPMVM_NUMOPTIONS 10
#define SPMVM_OPTION_NONE       (0x0)    // no special options applied
#define SPMVM_OPTION_AXPY       (0x1<<0) // perform y = y+A*x instead of y = A*x
#define SPMVM_OPTION_KEEPRESULT (0x1<<1) // keep result on OpenCL device 
#define SPMVM_OPTION_RHSPRESENT (0x1<<2) // assume that RHS vector is present
#define SPMVM_OPTION_NO_COMBINED_KERNELS (0x1<<3) // not configure comb. kernels
#define SPMVM_OPTION_NO_SPLIT_KERNELS    (0x1<<4) // not configure split kernels
#define SPMVM_OPTION_SERIAL_IO  (0x1<<5) // read matrix with one process only
#define SPMVM_OPTION_PIN        (0x1<<6) // pin threads to physical cores
#define SPMVM_OPTION_PIN_SMT    (0x1<<7) // pin threads to _all_ cores
#define SPMVM_OPTION_WORKDIST_NZE   (0x1<<8) // distribute by # of nonzeros
#define SPMVM_OPTION_WORKDIST_LNZE  (0x1<<9) // distribute by # of loc nonzeros
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

#define IF_DEBUG(level) if( DEBUG >= level )


/******************************************************************************/
/*----  Global definitions  --------------------------------------------------*/
/******************************************************************************/
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
typedef _Complex double mat_data_t;
#ifdef MPI
MPI_Datatype MPI_MYDATATYPE;
MPI_Op MPI_MYSUM;
#endif
#define DATATYPE_DESIRED DATATYPE_COMPLEX_DOUBLE
#else // COMPLEX
typedef double mat_data_t;
#ifdef MPI
#define MPI_MYDATATYPE MPI_DOUBLE
#define MPI_MYSUM MPI_SUM
#endif
#define DATATYPE_DESIRED DATATYPE_DOUBLE
#endif // COMPLEX
#endif // DOUBLE

#ifdef SINGLE
#ifdef COMPLEX
typedef _Complex float mat_data_t;
#ifdef MPI
MPI_Datatype MPI_MYDATATYPE;
MPI_Op MPI_MYSUM;
#endif
#define DATATYPE_DESIRED DATATYPE_COMPLEX_FLOAT
#else // COMPLEX
typedef float mat_data_t;
#ifdef MPI
#define MPI_MYDATATYPE MPI_FLOAT
#define MPI_MYSUM MPI_SUM
#endif
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

// TODO adjust

#ifdef DOUBLE
#define EPSILON 1e-8
#endif
#ifdef SINGLE
#define EPSILON 1e-0
#endif
#define EQUALS(a,b) (ABS(REAL(a)-REAL(b)<EPSILON))
/******************************************************************************/


/******************************************************************************/
/*----  Type definitions  ----------------------------------------------------*/
/******************************************************************************/
typedef struct
{
	int nDistinctDevices;
	int *nDevices;
	char **names;
} 
CL_DEVICE_INFO;

typedef struct 
{
	int nrows;
	mat_data_t* val;
#ifdef OPENCL
	cl_mem CL_val_gpu;
#endif
} 
VECTOR_TYPE;

typedef struct 
{
	int format[3];
	int T[3];
} 
SPM_GPUFORMATS;

typedef struct 
{
	int nrows;
	mat_data_t* val;
} 
HOSTVECTOR_TYPE;

typedef uint32_t mat_idx_t;
typedef uint32_t mat_nnz_t;

typedef struct 
{
	unsigned int nodes; // TODO delete
	unsigned int threads; // TODO delete
	mat_idx_t halo_elements;

	mat_nnz_t* lnEnts;
	mat_nnz_t* lfEnt;
	mat_idx_t* lnrows;
	mat_idx_t* lfRow;
	mat_idx_t* wishes;
	//int* wishlist_mem; // TODO delete
	//int** wishlist;    // TODO delete
	mat_idx_t* dues;
	//int* duelist_mem;  // TODO delete
	int** duelist;
	int* due_displ;    
	//int* wish_displ;   // TODO delete
	int* hput_pos;
} 
LCRP_TYPE; // TODO rename



#define PRmatNNZ PRIu32
#define PRmatIDX PRIu32


typedef struct 
{
	mat_trait_t trait; // TODO rename
	mat_nnz_t nnz; // TODO rename
	mat_idx_t nrows;
	mat_idx_t ncols;

	mat_idx_t *rowPerm;     // may be NULL
	mat_idx_t *invRowPerm;  // may be NULL

	void *data;
} 
MATRIX_TYPE;

#define TRAIT_INIT(...) { .format = SPM_FORMAT_NONE, .flags = SPM_DEFAULT, .aux = NULL, ## __VA_ARGS__ }
#define MATRIX_INIT(...) { .trait = TRAIT_INIT(), .nnz = 0, .nrows = 0, .ncols = 0, .rowPerm = NULL, .invRowPerm = NULL, .data = NULL, ## __VA_ARGS__ }

typedef struct 
{
	LCRP_TYPE *communicator; // TODO shorter
	MATRIX_TYPE *fullMatrix; // TODO array
	MATRIX_TYPE *localMatrix;
	MATRIX_TYPE *remoteMatrix;

	mat_idx_t nrows;
	mat_idx_t ncols;
	mat_nnz_t nnz;

	setup_flags_t flags;
#ifdef OPENCL
	GPUMATRIX_TYPE *devMatrix;
#endif
} 
SETUP_TYPE;

typedef struct 
{
	int fullFormat;
	int localFormat;
	int remoteFormat;
	int fullT;
	int localT;
	int remoteT;
	void *fullMatrix;
	void *localMatrix;
	void *remoteMatrix;
} 
GPUMATRIX_TYPE;



typedef struct
{
	mat_idx_t len;
	mat_idx_t idx;
	mat_data_t val;
	mat_idx_t minRow;
	mat_idx_t maxRow;
}
CONST_DIAG;


typedef struct 
{
	mat_idx_t nrows, ncols;
	mat_nnz_t nEnts;
	mat_idx_t*        rpt;
	mat_idx_t*        col;
	mat_data_t* val;

	mat_idx_t nConstDiags;
	CONST_DIAG *constDiags;
} 
CR_TYPE;

typedef struct 
{
	mat_data_t *val;
	mat_idx_t *col;
	mat_nnz_t *chunkStart;
	mat_idx_t *chunkMin; // for version with remainder loop
	mat_idx_t *chunkLen; // for version with remainder loop
	mat_idx_t *rowLen;   // for version with remainder loop
	mat_idx_t nrows;
	mat_idx_t nrowsPadded;
	mat_nnz_t nnz;
	mat_nnz_t nEnts;
	double nu;
} 
BJDS_TYPE;


typedef void (*SpMVM_kernelFunc)(VECTOR_TYPE*, void *, VECTOR_TYPE*, int);

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
 *   - mat_data_t (*fp)(int)
 *     A function pointer to a function taking an integer value and returning
 *     a mat_data_t. This function returns the initial value for the i-th (globally)
 *     element of the vector.
 *     If NULL, the vector is initialized to zero.
 *
 * Returns:
 *   a pointer to an LCRP_TYPE structure which holds the local matrix data as
 *   well as the necessary data structures for communication.
 *****************************************************************************/
void *SpMVM_createVector(SETUP_TYPE *setup, int type, mat_data_t (*fp)(int));

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
double SpMVM_solve(VECTOR_TYPE *res, SETUP_TYPE *setup, VECTOR_TYPE *invec, 
		int kernel, int nIter);


//MATRIX_TYPE *SpMVM_createGlobalMatrix (char *matrixPath, int format);
SETUP_TYPE *SpMVM_createSetup(char *matrixPath, mat_trait_t *trait, int nTraits, setup_flags_t, void *deviceFormats); 
//MATRIX_TYPE *SpMVM_createMatrix(char *matrixPath, mat_trait_t trait, void *deviceFormats); 
/******************************************************************************/

#endif
