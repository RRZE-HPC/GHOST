#ifndef _SPMVM_GLOBALS_H_
#define _SPMVM_GLOBALS_H_

#define NUMKERNELS 18

#define SPM_FORMAT_ELR 0
#define SPM_FORMAT_PJDS 1

#define SPMVM_OPTION_NONE (0x0)
#define SPMVM_OPTION_AXPY (0x1<<0)
#define SPMVM_OPTION_KEEPRESULT (0x1<<1)
#define SPMVM_OPTION_RHSPRESENT (0x1<<2)

#ifdef OPENCL
#include <CL/cl.h>
#endif

#include <complex.h>
#include <mpi.h>

#define DATATYPE_FLOAT 0
#define DATATYPE_DOUBLE 1
#define DATATYPE_COMPLEX_FLOAT 2
#define DATATYPE_COMPLEX_DOUBLE 3


static char *datatypeNames[] = {"float","double","cfloat","cdouble"};

#define DOUBLE
//#define COMPLEX









#ifdef DOUBLE
#ifdef COMPLEX

typedef _Complex double real;
#ifdef OPENCL
typedef cl_double2 clreal;
#endif
MPI_Datatype MPI_MYDATATYPE;
MPI_Op MPI_MYSUM;
#define DATATYPE_DESIRED DATATYPE_COMPLEX_DOUBLE

#else

typedef double real;
#ifdef OPENCL
typedef double clreal;
#endif
#define MPI_MYDATATYPE MPI_DOUBLE
#define MPI_MYSUM MPI_SUM
#define DATATYPE_DESIRED DATATYPE_DOUBLE

#endif
#endif


#ifdef SINGLE
#ifdef COMPLEX

typedef _Complex float real;
#ifdef OPENCL
typedef cl_float2 clreal;
#endif
MPI_Datatype MPI_MYDATATYPE;
MPI_Op MPI_MYSUM;
#define DATATYPE_DESIRED DATATYPE_COMPLEX_FLOAT

#else

typedef float real;
#ifdef OPENCL
typedef float clreal;
#endif
#define MPI_MYDATATYPE MPI_FLOAT
#define MPI_MYSUM MPI_SUM
#define DATATYPE_DESIRED DATATYPE_FLOAT


#endif
#endif

#ifdef COMPLEX
#define FLOPS_PER_ENTRY 4.0
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


typedef struct {
	real x;
	real y;
} COMPLEX_TYPE;

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
} MATRIX_FORMATS;
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
  int* row_ptr;
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
extern void hybrid_kernel_II   (int, VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);
extern void hybrid_kernel_III (int, VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);

typedef void (*FuncPrototype)( int, VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);

typedef struct {
    FuncPrototype kernel;
    real  cycles;
    real  time;
    char*   tag;
    char*   name;
} Hybrid_kernel;

static Hybrid_kernel HyK[NUMKERNELS] = {

    { &hybrid_kernel_0,    0, 0, "HyK_0", "ca :\npure OpenMP-kernel" },

    { &hybrid_kernel_I,    0, 0, "HyK_I", "ir -- cs -- wa -- ca :\nISend/IRecv; serial copy"},

    { &hybrid_kernel_II,    0, 0, "HyK_II", "ir -- cs -- cl -- wa -- nl :\nISend/IRecv; good faith hybrid" },
 
    { &hybrid_kernel_III,  0, 0, "HyK_III", "ir -- lc|csw -- nl:\ncopy in overlap region; dedicated comm-thread " },

}; 


void zeroVector(VECTOR_TYPE *vec);
VECTOR_TYPE* newVector( const int nRows );
void swapVectors(VECTOR_TYPE *v1, VECTOR_TYPE *v2);
HOSTVECTOR_TYPE* newHostVector( const int nRows, real (*fp)(int));
void normalize( real *vec, int nRows);

LCRP_TYPE* setup_communication(CR_TYPE* const, int);
void CL_setup_communication(LCRP_TYPE* const, MATRIX_FORMATS *);

int SPMVM_OPTIONS;
int JOBMASK;

#endif


