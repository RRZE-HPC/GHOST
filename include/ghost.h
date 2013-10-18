#ifndef __GHOST_H__
#define __GHOST_H__

#include <ghost_config.h>

#ifdef CUDAKERNEL
#undef GHOST_HAVE_MPI
#endif

#ifdef GHOST_HAVE_MPI
#ifdef __INTEL_COMPILER
#pragma warning (disable : 869)
#pragma warning (disable : 424)
#endif
#include <mpi.h>
#ifdef __INTEL_COMPILER
#pragma warning (enable : 424)
#pragma warning (enable : 869)
#endif
#else
typedef int MPI_Comm;
#define MPI_COMM_WORLD 0 // TODO unschoen
#endif

#ifndef GHOST_CLKERNEL
#include <stdlib.h>
#ifndef __cplusplus
#include <math.h>
#include <complex.h>
#else
#include <complex>
#endif
#include <inttypes.h>
#include <sys/types.h>
#include <pthread.h>

#ifndef CUDAKERNEL
#include <hwloc.h>
#endif

#ifdef GHOST_HAVE_OPENCL
#include <CL/cl.h>
#endif

#ifdef GHOST_HAVE_CUDA
#include <cuda.h>
#endif
#endif

#include "ghost_types.h"
#include "ghost_constants.h"

#define GHOST_NAME "ghost"
#define GHOST_VERSION "0.5"


/******************************************************************************/
/*----  Global definitions  --------------------------------------------------*/
/******************************************************************************/
#define CL_MY_DEVICE_TYPE CL_DEVICE_TYPE_GPU
//#define CUDA_PINNEDMEM
/******************************************************************************/

#define GHOST_REGISTER_DT_D(name) \
	typedef double name ## _t; \
	int name = GHOST_BINCRS_DT_DOUBLE|GHOST_BINCRS_DT_REAL; \

#define GHOST_REGISTER_DT_S(name) \
	typedef float name ## _t; \
	int name = GHOST_BINCRS_DT_FLOAT|GHOST_BINCRS_DT_REAL; \

#define GHOST_REGISTER_DT_C(name) \
	typedef complex float name ## _t; \
	int name = GHOST_BINCRS_DT_FLOAT|GHOST_BINCRS_DT_COMPLEX; \

#define GHOST_REGISTER_DT_Z(name) \
	typedef complex double name ## _t; \
	int name = GHOST_BINCRS_DT_DOUBLE|GHOST_BINCRS_DT_COMPLEX; \


#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************/
/*----  Type definitions  ----------------------------------------------------*/
/******************************************************************************/

typedef struct ghost_vec_t ghost_vec_t;
typedef struct ghost_mat_t ghost_mat_t;
typedef struct ghost_context_t ghost_context_t;
typedef struct ghost_comm_t ghost_comm_t;
typedef struct ghost_mtraits_t ghost_mtraits_t;
typedef struct ghost_vtraits_t ghost_vtraits_t;

/**
 * @brief This struct represents a vector (dense matrix) datatype.  The
 * according functions are accessed via function pointers. The first argument of
 * each member function always has to be a pointer to the vector itself.
 */
struct ghost_vec_t 
{
	/**
	 * @brief The vector's traits.
	 */
	ghost_vtraits_t *traits;
	/**
	 * @brief The context in which the vector is living.
	 */
	ghost_context_t *context;
	/**
	 * @brief The values of the vector.
	 */
	void* val;
	/**
	 * @brief Indicates whether the vector is a view. A view is a vector whose
	 * #val pointer points to some data which is not allocated withing the
	 * vector.
	 */
//	int isView;
	
	/**
	 * @brief Performs <em>y := a*x + y</em> with scalar a
	 *
	 * @param y The in-/output vector
	 * @param x The input vector
	 * @param a Points to the scale factor.
	 */
	void          (*axpy) (ghost_vec_t *y, ghost_vec_t *x, void *a);
	/**
	 * @brief Performs <em>y := a*x + b*y</em> with scalar a and b 
	 *
	 * @param y The in-/output vector.
	 * @param x The input vector
	 * @param a Points to the scale factor a.
	 * @param b Points to the scale factor b.
	 */
	void          (*axpby) (ghost_vec_t *y, ghost_vec_t *x, void *a, void *b);
	/**
	 * @brief \deprecated
	 */
	void          (*CLdownload) (ghost_vec_t *);
	/**
	 * @brief Clones a given number of columns of a source vector at a given
	 * column offset. 
	 *
	 * @param vec The source vector.
	 * @param ncols The number of columns to clone.
	 * @param coloffset The first column to clone.
	 *
	 * @return A clone of the source vector.
	 */
	ghost_vec_t * (*clone) (ghost_vec_t *vec, ghost_vidx_t ncols, ghost_vidx_t
			coloffset);
	/**
	 * @brief \deprecated
	 */
	void          (*CLupload) (ghost_vec_t *);
	/**
	 * @brief Collects vec from all MPI ranks and combines them into globalVec.
	 * The row permutation (if present) if vec's context is used.
	 *
	 * @param vec The distributed vector. 
	 * @param globalVec The global vector.
	 */
	void          (*collect) (ghost_vec_t *vec, ghost_vec_t *globalVec);
	/**
	 * @brief \deprecated
	 */
	void          (*CUdownload) (ghost_vec_t *);
	/**
	 * @brief \deprecated
	 */
	void          (*CUupload) (ghost_vec_t *);
	/**
	 * @brief Destroys a vector, i.e., frees all its data structures.
	 *
	 * @param vec The vector
	 */
	void          (*destroy) (ghost_vec_t *vec);
	/**
	 * @brief Distributes a global vector into node-local vetors.
	 *
	 * @param vec The global vector.
	 * @param localVec The local vector.
	 */
	void          (*distribute) (ghost_vec_t *vec, ghost_vec_t *localVec);
	/**
	 * @brief Computes the dot product of two vectors and stores the result in
	 * res.
	 *
	 * @param a The first vector.
	 * @param b The second vector.
	 * @param res A pointer to where the result should be stored.
	 */
	void          (*dotProduct) (ghost_vec_t *a, ghost_vec_t *b, void *res);
	/**
	 * @brief Downloads an entire vector from a compute device. Does nothing if
	 * the vector is not present on the device.
	 *
	 * @param vec The vector.
	 */
	void          (*download) (ghost_vec_t *vec);
	/**
	 * @brief Downloads only a vector's halo elements from a compute device.
	 * Does nothing if the vector is not present on the device.
	 *
	 * @param vec The vector.
	 */
	void          (*downloadHalo) (ghost_vec_t *vec);
	/**
	 * @brief Downloads only a vector's local elements (i.e., without halo
	 * elements) from a compute device. Does nothing if the vector is not
	 * present on the device.
	 *
	 * @param vec The vector.
	 */
	void          (*downloadNonHalo) (ghost_vec_t *vec);
	/**
	 * @brief Stores the entry of the vector at a given index (row i, column j)
	 * into entry.
	 *
	 * @param vec The vector.
	 * @param ghost_vidx_t i The row.
	 * @param ghost_vidx_t j The column.
	 * @param entry Where to store the entry.
	 */
	void          (*entry) (ghost_vec_t *vec, ghost_vidx_t i, ghost_vidx_t j,
			void *entry);
	/**
	 * @brief Initializes a vector from a given function.
	 * Malloc's memory for the vector's values if this hasn't happened before. 
	 *
	 * @param vec The vector.
	 * @param fp The function pointer. The function takes three arguments: The row index, the column index and a pointer to where to store the value at this position.
	 */
	void          (*fromFunc) (ghost_vec_t *vec, void (*fp)(int,int,void *)); // TODO ghost_vidx_t
	/**
	 * @brief Initializes a vector from another vector at a given column offset.
	 * Malloc's memory for the vector's values if this hasn't happened before. 
	 *
	 * @param vec The vector.
	 * @param src The source vector.
	 * @param ghost_vidx_t The column offset in the source vector.
	 */
	void          (*fromVec) (ghost_vec_t *vec, ghost_vec_t *src, ghost_vidx_t offset);
	/**
	 * @brief Initializes a vector from a file at a given row offset.
	 * Malloc's memory for the vector's values if this hasn't happened before. 
	 *
	 * @param vec The vector.
	 * @param filename Path to the file.
	 * @param offset The row offset.
	 */
	void          (*fromFile) (ghost_vec_t *vec, char *filename, off_t offset);
	/**
	 * @brief Initiliazes a vector from random values.
	 *
	 * @param vec The vector.
	 */
	void          (*fromRand) (ghost_vec_t *vec);
	/**
	 * @brief Initializes a vector from a given scalar value.
	 *
	 * @param vec The vector.
	 * @param val A pointer to the value.
	 */
	void          (*fromScalar) (ghost_vec_t *vec, void *val);
	/**
	 * @brief Normalize a vector, i.e., scale it such that its 2-norm is one.
	 *
	 * @param vec The vector.
	 */
	void          (*normalize) (ghost_vec_t *vec);
	/**
	 * @brief Permute a vector with a given permutation.
	 *
	 * @param vec The vector.
	 * @param perm The permutation.
	 */
	void          (*permute) (ghost_vec_t *vec, ghost_vidx_t *perm);
	/**
	 * @brief Print a vector.
	 *
	 * @param vec The vector.
	 */
	void          (*print) (ghost_vec_t *vec);
	/**
	 * @brief Scale a vector with a given scalar.
	 *
	 * @param vec The vector.
	 * @param scale The scale factor.
	 */
	void          (*scale) (ghost_vec_t *vec, void *scale);
	/**
	 * @brief Swap two vectors.
	 *
	 * @param vec1 The first vector.
	 * @param vec2 The second vector.
	 */
	void          (*swap) (ghost_vec_t *vec1, ghost_vec_t *vec2);
	/**
	 * @brief Write a vector to a file.
	 *
	 * @param vec The vector.
	 * @param filename The path to the file.
	 */
	void          (*toFile) (ghost_vec_t *vec, char *filename);
	/**
	 * @brief Uploads an entire vector to a compute device. Does nothing if
	 * the vector is not present on the device.
	 *
	 * @param vec The vector.
	 */
	void          (*upload) (ghost_vec_t *vec);
	/**
	 * @brief Uploads only a vector's halo elements to a compute device.
	 * Does nothing if the vector is not present on the device.
	 *
	 * @param vec The vector.
	 */
	void          (*uploadHalo) (ghost_vec_t *vec);
	/**
	 * @brief Uploads only a vector's local elements (i.e., without halo
	 * elements) to a compute device. Does nothing if the vector is not
	 * present on the device.
	 *
	 * @param vec The vector.
	 */
	void          (*uploadNonHalo) (ghost_vec_t *vec);
	/**
	 * @brief View plain data in the vector.
	 * That means that the vector has no memory malloc'd but its data pointer only points to the memory provided. 
	 *
	 * @param vec The vector.
	 * @param data The plain data.
	 * @param ghost_vidx_t nr The number of rows.
	 * @param ghost_vidx_t nc The number of columns.
	 * @param ghost_vidx_t roffs The row offset.
	 * @param ghost_vidx_t coffs The column offset.
	 * @param ghost_vidx_t lda The number of rows per column.
	 */
	void          (*viewPlain) (ghost_vec_t *vec, void *data, ghost_vidx_t nr, ghost_vidx_t nc, ghost_vidx_t roffs, ghost_vidx_t coffs, ghost_vidx_t lda);

	ghost_vec_t * (*viewScatteredVec) (ghost_vec_t *src, ghost_vidx_t nc, ghost_vidx_t *coffs);


	/**
	 * @brief Create a vector as a view of another vector.
	 *
	 * @param src The source vector.
	 * @param nc The nunber of columns to view.
	 * @param coffs The column offset.
	 *
	 * @return The new vector.
	 */
	ghost_vec_t * (*viewVec) (ghost_vec_t *src, ghost_vidx_t nc, ghost_vidx_t coffs);
	/**
	 * @brief Scale each column of a vector with a given scale factor.
	 *
	 * @param vec The vector.
	 * @param scale The scale factors.
	 */
	void          (*vscale) (ghost_vec_t *, void *);
	void          (*vaxpy) (ghost_vec_t *, ghost_vec_t *, void *);
	void          (*vaxpby) (ghost_vec_t *, ghost_vec_t *, void *, void *);
	void          (*zero) (ghost_vec_t *);

#ifdef GHOST_HAVE_OPENCL
	cl_mem CL_val_gpu;
#endif
#ifdef GHOST_HAVE_CUDA
	void * CU_val;
#endif
};

struct ghost_vtraits_t
{
	ghost_midx_t nrows;
	ghost_midx_t nrowshalo;
	ghost_midx_t nrowspadded;
	ghost_midx_t nvecs;
	int flags;
	int datatype;
	void * aux;
}; 
#define GHOST_VTRAITS_INIT(...) {.flags = GHOST_VEC_DEFAULT, .aux = NULL, .datatype = GHOST_BINCRS_DT_DOUBLE|GHOST_BINCRS_DT_REAL, .nrows = 0, .nrowshalo = 0, .nrowspadded = 0, .nvecs = 1, ## __VA_ARGS__ }

typedef struct 
{
	int format[3];
	int T[3];
} 
GHOST_SPM_GPUFORMATS;

typedef void (*ghost_kernel_t)(ghost_mat_t*, ghost_vec_t*, ghost_vec_t*, int);
typedef void (*ghost_solver_t)(ghost_context_t *, ghost_vec_t*, ghost_mat_t *, ghost_vec_t*, int);

struct ghost_comm_t 
{
	ghost_midx_t halo_elements; // number of nonlocal RHS vector elements
	ghost_mnnz_t* lnEnts;
	ghost_mnnz_t* lfEnt;
	ghost_midx_t* lnrows;
	ghost_midx_t* lfRow;
	int* wishes;
	int** wishlist;    
	int* dues;
	int** duelist;
	ghost_midx_t* due_displ;    
	ghost_midx_t* wish_displ;   
	ghost_midx_t* hput_pos;
}; 

struct ghost_mat_t 
{
	ghost_mtraits_t *traits;
	int symmetry;
	ghost_mat_t *localPart;
	ghost_mat_t *remotePart;
	ghost_context_t *context;
	char *name;
	void *data;
	ghost_kernel_t kernel;

	// access functions
	void       (*destroy) (ghost_mat_t *);
	void       (*printInfo) (ghost_mat_t *);
	ghost_mnnz_t  (*nnz) (ghost_mat_t *);
	ghost_midx_t  (*nrows) (ghost_mat_t *);
	ghost_midx_t  (*ncols) (ghost_mat_t *);
	ghost_midx_t  (*rowLen) (ghost_mat_t *, ghost_midx_t);
	char *     (*formatName) (ghost_mat_t *);
	void       (*fromFile)(ghost_mat_t *, char *);
	void       (*CLupload)(ghost_mat_t *);
	void       (*CUupload)(ghost_mat_t *);
	size_t     (*byteSize)(ghost_mat_t *);
	void       (*fromCRS)(ghost_mat_t *, void *);
	void       (*split)(ghost_mat_t *);
#ifdef GHOST_HAVE_OPENCL
	cl_kernel clkernel[4];
#endif
}; 

struct ghost_context_t
{
	ghost_solver_t *solvers;

	// if the context is distributed by nnz, the row pointers are being read 
	// at context creation in order to create the distribution. once the matrix 
	// is being created, the row pointers are distributed
	ghost_midx_t *rpt; 

	ghost_comm_t *communicator;
	ghost_midx_t gnrows;
	ghost_midx_t gncols;
	int flags;
	double weight;

	ghost_midx_t *rowPerm;    // may be NULL
	ghost_midx_t *invRowPerm; // may be NULL

	MPI_Comm mpicomm;
};

struct ghost_mtraits_t
{
	int format;
	int flags;
	void * aux;
	int nAux;
	int datatype;
	void * shift; 
	void * scale; 
};

#define GHOST_MTRAITS_INIT(...) {.flags = GHOST_SPM_DEFAULT, .aux = NULL, .nAux = 0, .datatype = GHOST_BINCRS_DT_DOUBLE|GHOST_BINCRS_DT_REAL, .format = GHOST_SPM_FORMAT_CRS, .shift = NULL, .scale = NULL, ## __VA_ARGS__ }

typedef struct
{
	int nDistinctDevices;
	int *nDevices;
	char **names;
} 
ghost_acc_info_t;


typedef struct
{
	int32_t endianess;
	int32_t version;
	int32_t base;
	int32_t symmetry;
	int32_t datatype;
	int64_t nrows;
	int64_t ncols;
	int64_t nnz;
} ghost_matfile_header_t;


void ghost_normalizeVec(ghost_vec_t *);
void ghost_dotProduct(ghost_vec_t *, ghost_vec_t *, void *);
void ghost_vecToFile(ghost_vec_t *, char *);
void ghost_vecFromFile(ghost_vec_t *, char *);
void ghost_vecFromScalar(ghost_vec_t *v, void *s);
void ghost_vecFromFunc(ghost_vec_t *v, void (*func)(int,int,void*));
void ghost_freeVec(ghost_vec_t *vec);

/******************************************************************************/

/******************************************************************************/
/*----  Function prototypes  -------------------------------------------------*/
/******************************************************************************/

/******************************************************************************
 * Initializes the basic functionality of the ghost. This includes:
 *   - initialize MPI
 *   - create and commit custom MPI datatypes (if necessary)
 *   - pin threads to CPU cores (if defined)
 *   - context the MPI communicator for the node
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
 *
 * The call to ghost_init() has to be done before any other ghost_*() call.
 *****************************************************************************/
int ghost_init(int argc, char **argv);

/******************************************************************************
 * Clean up and finalize before termination. This includes:
 *   - call MPI_Finalize()
 *   - finish the OpenCL functionality
 *   - close the Likwid Marker API
 *
 * The ghost_finish() call is usually the last call of the main program. 
 *****************************************************************************/
void ghost_finish();

int ghost_spmvm(ghost_context_t *context, ghost_vec_t *res, ghost_mat_t *mat, ghost_vec_t *invec, 
		int *spmvmOptions);

ghost_context_t * ghost_createContext(int64_t, int64_t, int, char *, MPI_Comm, double weight);
ghost_mat_t     * ghost_createMatrix(ghost_context_t *, ghost_mtraits_t *, int);
void              ghost_freeContext(ghost_context_t *);
/******************************************************************************/
void ghost_matFromFile(ghost_mat_t *, char *);

int ghost_gemm(char *, ghost_vec_t *,  ghost_vec_t *, ghost_vec_t *, void *, void *, int); 

extern int hasCUDAdevice;
extern int hasOPENCLdevice;
#ifndef CUDAKERNEL
extern hwloc_topology_t topology;
#endif

#ifdef __cplusplus
} // extern "C"
#endif

#endif
