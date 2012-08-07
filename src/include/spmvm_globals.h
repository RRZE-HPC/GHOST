#ifndef _SPMVM_GLOBALS_H_
#define _SPMVM_GLOBALS_H_

#include <spmvm_constants.h>
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

extern void hybrid_kernel_0   (VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);
extern void hybrid_kernel_I   (VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);
extern void hybrid_kernel_II  (VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);
extern void hybrid_kernel_III (VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);

typedef void (*FuncPrototype)(VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);

typedef struct {
    FuncPrototype kernel;
    real  cycles;
    real  time;
    char*   tag;
    char*   name;
} Hybrid_kernel;

extern Hybrid_kernel SPMVM_KERNELS[SPMVM_NUMKERNELS];

/**********************************************/
/****** global variables **********************/
/**********************************************/
int SPMVM_OPTIONS;
int SPMVM_KERNELS_SELECTED;

#endif


