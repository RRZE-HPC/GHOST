#ifndef _SPMVM_GLOBALS_H_
#define _SPMVM_GLOBALS_H_

#include <complex.h>
#include <mpi.h>

#ifdef OPENCL
#include <CL/cl.h>
#endif


/**********************************************/
/****** global definitions ********************/
/**********************************************/
//#define LIKWID_MARKER
//#define LIKWID_MARKER_FINE
#define DOUBLE
//#define COMPLEX
/**********************************************/


#ifdef LIKWID_MARKER_FINE
#define LIKWID_MARKER
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


/**********************************************/
/****** Definitions depending on datatype *****/
/**********************************************/
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
#else
#define ABS(a) fabs(a)
#define REAL(a) a
#define IMAG(a) 0.0
#endif
#endif

#ifdef SINGLE
#ifdef COMPLEX
#define ABS(a) cabsf(a)
#define REAL(a) crealf(a)
#define IMAG(a) cimagf(a)
#else
#define ABS(a) fabsf(a)
#define REAL(a) a
#define IMAG(a) 0.0
#endif
#endif
/**********************************************/


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

extern void hybrid_kernel_0   (int, VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);
extern void hybrid_kernel_I   (int, VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);
extern void hybrid_kernel_II  (int, VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);
extern void hybrid_kernel_III (int, VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);

typedef void (*FuncPrototype)( int, VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);

typedef struct {
    FuncPrototype kernel;
    real  cycles;
    real  time;
    char*   tag;
    char*   name;
} Hybrid_kernel;

static Hybrid_kernel HyK[SPMVM_NUMKERNELS] = {

    { &hybrid_kernel_0,    0, 0, "HyK_0", "ca :\npure OpenMP-kernel" },

    { &hybrid_kernel_I,    0, 0, "HyK_I", "ir -- cs -- wa -- ca :\nISend/IRecv; \
		serial copy"},

    { &hybrid_kernel_II,    0, 0, "HyK_II", "ir -- cs -- cl -- wa -- nl :\
		\nISend/IRecv; good faith hybrid" },
 
    { &hybrid_kernel_III,  0, 0, "HyK_III", "ir -- lc|csw -- nl:\ncopy in \
		overlap region; dedicated comm-thread " },

}; 


/**********************************************/
/****** global variables **********************/
/**********************************************/
int SPMVM_OPTIONS;
int SPMVM_KERNELS;

#endif


