#ifndef _SPMVM_GLOBALS_H_
#define _SPMVM_GLOBALS_H_

#define NUMKERNELS 18

#define SPM_FORMAT_ELR 0
#define SPM_FORMAT_PJDS 1

#define SPMVM_OPTION_NONE (0x0)
#define SPMVM_OPTION_AXPY (0x1<<0)
#define SPMVM_OPTION_KEEPRESULT (0x1<<1)
#define SPMVM_OPTION_RHSPRESENT (0x1<<2)

#ifdef OCLKERNEL
#include <CL/cl.h>
#endif

typedef struct {
	int nRows;
	double* val;
#ifdef OCLKERNEL
  cl_mem CL_val_gpu;
#endif

} VECTOR_TYPE;

typedef struct {
	int format[3];
	int T[3];
} MATRIX_FORMATS;
typedef struct {
	int nRows;
	double* val;
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
  double* val;
  int* col;
  int* row_ptr;
  int* lrow_ptr;
  int* lrow_ptr_l;
  int* lrow_ptr_r;
  int* lcol;
  int* rcol;
  double* lval;
  double* rval;
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
	double* val;
} CR_TYPE;

extern void hybrid_kernel_0   (int, VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);
extern void hybrid_kernel_I   (int, VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);
extern void hybrid_kernel_II  (int, VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);
extern void hybrid_kernel_III (int, VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);
extern void hybrid_kernel_IV  (int, VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);
extern void hybrid_kernel_V   (int, VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);
extern void hybrid_kernel_VI  (int, VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);
extern void hybrid_kernel_VII (int, VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);
extern void hybrid_kernel_VIII(int, VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);
extern void hybrid_kernel_IX  (int, VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);
extern void hybrid_kernel_X   (int, VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);
extern void hybrid_kernel_XI  (int, VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);
extern void hybrid_kernel_XII (int, VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);
extern void hybrid_kernel_XIII(int, VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);
extern void hybrid_kernel_XIV (int, VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);
extern void hybrid_kernel_XV  (int, VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);
extern void hybrid_kernel_XVI  (int, VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);
extern void hybrid_kernel_XVII (int, VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);

typedef void (*FuncPrototype)( int, VECTOR_TYPE*, LCRP_TYPE*, VECTOR_TYPE*);

typedef struct {
    FuncPrototype kernel;
    double  cycles;
    double  time;
    char*   tag;
    char*   name;
} Hybrid_kernel;

static Hybrid_kernel HyK[NUMKERNELS] = {

    { &hybrid_kernel_0,    0, 0, "HyK_0", "ca :\npure OpenMP-kernel" },

    { &hybrid_kernel_I,    0, 0, "HyK_I", "cp -- co -- ca :\nSendRecv; serial copy" },

    { &hybrid_kernel_II,   0, 0, "HyK_II", "cp -- co -- ca :\nSendRecv; parallel copy" },
 
    { &hybrid_kernel_III,  0, 0, "HyK_III", "cp -- co -- ca :\n not true anymore!" },
 
    { &hybrid_kernel_IV,   0, 0, "HyK_IV", "cp -- co -- ca :\nAlltoallv" },
 
    { &hybrid_kernel_V,    0, 0, "HyK_V", "ir -- cs -- wa -- ca :\nISend/IRecv; serial copy"},

    { &hybrid_kernel_VI,   0, 0, "HyK_VI", "ir -- cs -- wa -- ca :\nISend/IRecv; copy in parallel inner loop" },
 
    { &hybrid_kernel_VII,  0, 0, "HyK_VII", "ir -- cs -- wa -- ca :\nISend/IRecv; parallel region, ISend protected by single" },
 
    { &hybrid_kernel_VIII, 0, 0, "HyK_VIII", "ir -- cs -- wa -- ca :\nISend/IRecv; parallelisation over to_PE" } ,
 
    { &hybrid_kernel_IX,   0, 0, "HyK_IX", "ir -- cs -- wa -- cl -- nl :\nISend/IRecv; overhead LNL compared to HyK_V" },
 
    { &hybrid_kernel_X,    0, 0, "HyK_X", "ir -- cs -- cl -- wa -- nl :\nISend/IRecv; good faith hybrid" },
 
    { &hybrid_kernel_XI,   0, 0, "HyK_XI", "ir -- cp -- lc|sw -- nl :\ndedicated communication thread " },

    { &hybrid_kernel_XII,  0, 0, "HyK_XII", "ir -- lc|csw -- nl:\ncopy in overlap region; dedicated comm-thread " },

    { &hybrid_kernel_XIII, 0, 0, "HyK_XIII", "ir -- cp|x -- x|sw -- lc|x -- nl|x :\nadditional comm-thread; contributions" },

    { &hybrid_kernel_XIV,  0, 0, "HyK_XIV", "ir -- cp|x -- lc|sw -- nl|x :\nadditional comm-thread" },

    { &hybrid_kernel_XV,   0, 0, "HyK_XV", "cp|x -- lc|cc -- nl|x :\nadditional comm-thread; also IRecv in overlap region" },

    { &hybrid_kernel_XVI,  0, 0, "HyK_XVI", "cp|x -- lc|cc -- nl|x :\nadditional comm-thread; load balancing" },

    { &hybrid_kernel_XVII, 0, 0, "HyK_XVII", "ir -- cs -- cl -- wa -- nl:\ngood faith hybrid; fused request array" },
  

}; 


void zeroVector(VECTOR_TYPE *vec);
VECTOR_TYPE* newVector( const int nRows );
void swapVectors(VECTOR_TYPE *v1, VECTOR_TYPE *v2);
HOSTVECTOR_TYPE* newHostVector( const int nRows, double (*fp)(int));
void normalize( double *vec, int nRows);

LCRP_TYPE* setup_communication(CR_TYPE* const, int);
void CL_setup_communication(LCRP_TYPE* const, MATRIX_FORMATS *);

int SPMVM_OPTIONS;
int JOBMASK;

#endif


