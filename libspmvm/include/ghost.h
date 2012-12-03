#ifndef _SPMVM_H_
#define _SPMVM_H_

#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <inttypes.h>

#ifdef OPENCL
#include <CL/cl.h>
#endif

#include "ghost_types.h"

#define GHOST_NAME "ghost"
#define GHOST_VERSION "0.1a"

/******************************************************************************/
/*----  SpMVM kernels  -------------------------------------------------------*/
/******************************************************************************/
#define GHOST_NUM_MODES 4

#define GHOST_MODE_NOMPI      (0)
#define GHOST_MODE_VECTORMODE (1)
#define GHOST_MODE_GOODFAITH  (2)
#define GHOST_MODE_TASKMODE   (3)

#define GHOST_MODES_COMBINED (GHOST_MODE_NOMPI | GHOST_MODE_VECTORMODE)
#define GHOST_MODES_SPLIT    (GHOST_MODE_GOODFAITH | GHOST_MODE_TASKMODE)
#define GHOST_MODES_ALL      (GHOST_MODES_COMBINED | GHOST_MODES_SPLIT)

#define GHOST_FULL_MAT_IDX 0
#define GHOST_LOCAL_MAT_IDX 1
#define GHOST_REMOTE_MAT_IDX 2
/******************************************************************************/

typedef struct
{
//	unsigned int format; // TODO delete
	const char * format;
	unsigned int flags;
	void * aux;
} 
ghost_mtraits_t;


// formats
#define GHOST_NUM_SPMFORMATS (5)
#define GHOST_SPMFORMAT_NONE   (0)
#define GHOST_SPMFORMAT_CRS    (1)
#define GHOST_SPMFORMAT_BJDS   (2)
#define GHOST_SPMFORMAT_TBJDS  (3)
#define GHOST_SPMFORMAT_CRSCD  (4)

// flags
#define GHOST_SETUP_DEFAULT       (0)
#define GHOST_SETUP_GLOBAL        (0x1<<0)
#define GHOST_SETUP_DISTRIBUTED   (0x1<<1)

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
#define GHOST_DATATYPE_S 0
#define GHOST_DATATYPE_D 1
#define GHOST_DATATYPE_C 2
#define GHOST_DATATYPE_Z 3
extern const char *DATATYPE_NAMES[];
/******************************************************************************/



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

#ifdef OPENCL
typedef cl_double ghost_cl_mdat_t; // TODO
typedef cl_double ghost_cl_vdat_t; // TODO
#endif

/******************************************************************************/
/*----  Definitions depending on datatype  -----------------------------------*/
/******************************************************************************/
#ifdef GHOST_MAT_DP
#ifdef GHOST_MAT_COMPLEX
#define GHOST_CLFLAGS " -DDOUBLE -DCOMPLEX "
typedef _Complex double ghost_mdat_t;
#ifdef MPI
MPI_Datatype MPI_MYDATATYPE;
MPI_Op MPI_MYSUM;
#endif
#define DATATYPE_DESIRED GHOST_DATATYPE_Z
#else // GHOST_MAT_COMPLEX
#define GHOST_CLFLAGS " -DDOUBLE "
typedef double ghost_mdat_t;
#ifdef MPI
#define MPI_MYDATATYPE MPI_DOUBLE
#define MPI_MYSUM MPI_SUM
#endif
#define DATATYPE_DESIRED GHOST_DATATYPE_D
#endif // GHOST_MAT_COMPLEX
#endif // GHOST_MAT_DP

#ifdef GHOST_MAT_SP
#ifdef GHOST_MAT_COMPLEX
typedef _Complex float ghost_mdat_t;
#define GHOST_CLFLAGS " -DSINGLE -DCOMPLEX "
#ifdef MPI
MPI_Datatype MPI_MYDATATYPE;
MPI_Op MPI_MYSUM;
#endif
#define DATATYPE_DESIRED GHOST_DATATYPE_C
#else // GHOST_MAT_COMPLEX
#define GHOST_CLFLAGS " -DSINGLE "
typedef float ghost_mdat_t;
#ifdef MPI
#define MPI_MYDATATYPE MPI_FLOAT
#define MPI_MYSUM MPI_SUM
#endif
#define DATATYPE_DESIRED GHOST_DATATYPE_S
#endif // GHOST_MAT_COMPLEX
#endif // GHOST_MAT_SP

#ifdef GHOST_MAT_COMPLEX
#define FLOPS_PER_ENTRY 8.0
#else
#define FLOPS_PER_ENTRY 2.0
#endif

#ifdef GHOST_MAT_DP
#ifdef GHOST_MAT_COMPLEX
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

#ifdef GHOST_MAT_SP
#ifdef GHOST_MAT_COMPLEX
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

#ifdef GHOST_MAT_DP
#define EPSILON 1e-4
#endif
#ifdef GHOST_MAT_SP
#define EPSILON 1e-0 // TODO
#endif
#define EQUALS(a,b) (ABS(REAL(a)-REAL(b))<EPSILON && ABS(IMAG(a)-IMAG(b))<EPSILON)
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
	unsigned int flags;
	int nrows;
	ghost_mdat_t* val;
#ifdef OPENCL
	cl_mem CL_val_gpu;
#endif
} 
ghost_vec_t;

typedef struct 
{
	int format[3];
	int T[3];
} 
GHOST_SPM_GPUFORMATS;

typedef uint32_t mat_idx_t; // type for the index of the matrix
typedef uint32_t mat_nnz_t; // type for the number of nonzeros in the matrix
#ifdef OPENCL
typedef cl_uint ghost_cl_midx_t;
typedef cl_uint ghost_cl_mnnz_t;
#endif

#define PRmatNNZ PRIu32
#define PRmatIDX PRIu32

typedef struct ghost_mat_t ghost_mat_t;
typedef struct ghost_setup_t ghost_setup_t;
typedef struct ghost_comm_t ghost_comm_t;
typedef struct ghost_spmf_plugin_t ghost_spmf_plugin_t;

typedef void (*ghost_kernel_t)(ghost_mat_t*, ghost_vec_t*, ghost_vec_t*, int);
typedef void (*ghost_solver_t)(ghost_vec_t*, ghost_setup_t *setup, ghost_vec_t*, int);
typedef ghost_mat_t * (*ghost_spmf_init_t) (ghost_mtraits_t *);

struct ghost_comm_t 
{
	mat_idx_t halo_elements; // number of nonlocal RHS vector elements
	mat_nnz_t* lnEnts;
	mat_nnz_t* lfEnt;
	mat_idx_t* lnrows;
	mat_idx_t* lfRow;
	mat_nnz_t* wishes;
	int* wishlist_mem; // TODO delete
	int** wishlist;    // TODO delete
	mat_nnz_t* dues;
	int* duelist_mem;  // TODO delete
	int** duelist;
	int* due_displ;    
	int* wish_displ;   // TODO delete
	int* hput_pos;
}; 

struct ghost_mat_t 
{
	ghost_mtraits_t *traits; // TODO rename

	// access functions
	void       (*destroy) (ghost_mat_t *);
	void       (*printInfo) (ghost_mat_t *);
	mat_nnz_t  (*nnz) (ghost_mat_t *);
	mat_idx_t  (*nrows) (ghost_mat_t *);
	mat_idx_t  (*ncols) (ghost_mat_t *);
	mat_idx_t  (*rowLen) (ghost_mat_t *, mat_idx_t i);
	ghost_mdat_t (*entry) (ghost_mat_t *, mat_idx_t i, mat_idx_t j);
	char *     (*formatName) (ghost_mat_t *);
	void       (*fromBin)(ghost_mat_t *, char *matrixPath);
	void       (*fromMM)(ghost_mat_t *, char *matrixPath);
	size_t     (*byteSize) (ghost_mat_t *);
	ghost_kernel_t kernel;
#ifdef OPENCL
	cl_kernel clkernel;
#endif

	mat_idx_t *rowPerm;     // may be NULL
	mat_idx_t *invRowPerm;  // may be NULL

	void *data;
}; 

struct ghost_spmf_plugin_t
{
	void *so;
	ghost_spmf_init_t init;
	char *name;
	char *version;
	char *formatID;
};

struct ghost_setup_t
{
	ghost_solver_t *solvers;

	ghost_comm_t *communicator; // TODO shorter
	ghost_mat_t *fullMatrix; // TODO array
	ghost_mat_t *localMatrix;
	ghost_mat_t *remoteMatrix;

	mat_idx_t nrows;
	mat_idx_t ncols;
	mat_nnz_t nnz;

	mat_idx_t lnrows;

	char *matrixName;

	unsigned int flags;
/*#ifdef OPENCL
	ghost_mat_t *fullCLMatrix; // TODO array
	ghost_mat_t *localCLMatrix;
	ghost_mat_t *remoteCLMatrix;
#endif*/
};

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
 *     are defined as GHOST_OPTION_* (explained above).
 *
 * Returns:
 *   an integer which holds the rank of the calling MPI process within
 *   MPI_COMM_WORLD
 *
 * The call to ghost_init() has to be done before any other ghost_*() call.
 *****************************************************************************/
int ghost_init(int argc, char **argv, int options);

/******************************************************************************
 * Clean up and finalize before termination. This includes:
 *   - call MPI_Finalize()
 *   - finish the OpenCL functionality
 *   - close the Likwid Marker API
 *
 * The ghost_finish() call is usually the last call of the main program. 
 *****************************************************************************/
void ghost_finish();

/******************************************************************************
 * Create a distributed CRS matrix from a given path. The matrix is read-in
 * from the processes in a parallel way (unless defined differently via
 * GHOST_OPTION_SERIAL_IO) and necessary data structures for communication are
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
 *     If OpenCL is enabled, this has to be a pointer to a GHOST_SPM_GPUFORMATS
 *     structure as defined above.
 *
 * Returns:
 *   a pointer to an ghost_comm_t structure which holds the local matrix data as
 *   well as the necessary data structures for communication.
 *****************************************************************************/
ghost_comm_t * ghost_createCRS (char *matrixPath, void *deviceFormats);


/******************************************************************************
 * Create a distributed vector with specified values in order to use it for 
 * SpMVM. Depending on the type, the length of the vector may differ.
 * If OpenCL is enabled, the vector is also being created and initialized on
 * the device.
 *
 * Arguments:
 *   - ghost_comm_t *lcrp
 *     The local CRS matrix portion to use with the vector.
 *   - int type
 *     Specifies whether the vector is a right hand side vector 
 *     (GHOST_VEC_RHS), left hand side vector (GHOST_VEC_LHS) or a vector
 *     which may be used as both right and left hand side (GHOST_VEC_BOTH).
 *     The length of the vector depends on this argument.
 *   - ghost_mdat_t (*fp)(int)
 *     A function pointer to a function taking an integer value and returning
 *     a ghost_mdat_t. This function returns the initial value for the i-th (globally)
 *     element of the vector.
 *     If NULL, the vector is initialized to zero.
 *
 * Returns:
 *   a pointer to an ghost_comm_t structure which holds the local matrix data as
 *   well as the necessary data structures for communication.
 *****************************************************************************/
ghost_vec_t *ghost_createVector(ghost_setup_t *setup, unsigned int type, ghost_mdat_t (*fp)(int));

/******************************************************************************
 * Perform the sparse matrix vector product using a specified kernel with a
 * fixed number of iterations.
 *
 * Arguments:
 *   - ghost_vec_t *res 
 *     The result vector. Its values are being accumulated if GHOST_OPTION_AXPY
 *     is defined.  
 *   - ghost_comm_t *lcrp
 *     The local CRS matrix part.
 *   - ghost_vec_t *invec
 *     The left hand side vector.
 *   - int kernel
 *     The kernel which should be used. This has to be one out of
 *       + GHOST_MODE_NOMPI
 *       + GHOST_MODE_VECTORMODE
 *       + GHOST_MODE_GOODFAITH
 *       + GHOST_MODE_TASKMODE
 *   - int nIter
 *     The number of iterations to run.
 *     
 * Returns:
 *   the wallclock time (in seconds) the kernel execution took. 
 *****************************************************************************/
double ghost_solve(ghost_vec_t *res, ghost_setup_t *setup, ghost_vec_t *invec, 
		int kernel, int nIter);

ghost_setup_t *ghost_createSetup(char *matrixPath, ghost_mtraits_t *trait, int nTraits, unsigned int); 
ghost_mat_t * ghost_initMatrix(ghost_mtraits_t *);
void ghost_freeSetup(ghost_setup_t *setup);
/******************************************************************************/

#endif
