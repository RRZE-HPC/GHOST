#ifndef _MATRICKS_H_
#define _MATRICKS_H_

#include <stdio.h>
#include <stdlib.h>
#include <mymacros.h>
#include <mpi.h>

#ifdef OCLKERNEL
#include "my_ellpack.h"
#include <CL/cl.h>
#endif

#define SPM_FORMAT_ELR 0
#define SPM_FORMAT_PJDS 1


typedef struct {
	int nRows;
	double* val;
#ifdef OCLKERNEL
  cl_mem CL_val_gpu;
#endif

} VECTOR_TYPE;

typedef struct {
	int nRows;
	int* val;
#ifdef OCLKERNEL
  cl_mem CL_val_gpu;
#endif
} INT_VECTOR_TYPE;


typedef struct {
	int format[3];
	int T[3];
} MATRIX_FORMATS;

typedef struct {
	int row, col, nThEntryInRow;
	double val;
} NZE_TYPE;

typedef struct {
	int nRows, nCols, nEnts;
	NZE_TYPE* nze;
} MM_TYPE;

typedef struct {
	int nRows, nCols, nEnts;
	int* rowOffset;
	int* col;
	double* val;
} CR_TYPE;

typedef struct {
	int nRows, nCols, nEnts, nDiags;
	int* rowPerm;
	int* diagOffset;
	int* col;
	double* val;
} JD_TYPE;

typedef struct {
	int row, nEntsInRow;
} JD_SORT_TYPE;


typedef struct {
        int nEnts, nRows, totalblocks;
        int* blockdim_rows;
        int* blockdim_cols;
        int* resorted_col;
        double* resorted_val;
        int* blockinfo;
        int* tbi;
} JD_RESORTED_TYPE;

typedef struct {
	int pagesize, cachesize;
        int vecdim;
        int ppvec, offset, numvecs, globdim;
        double* mem;
        double** vec;
} REVBUF_TYPE;


typedef struct {
        int nEnts, nRows, nDiags, totalblocks;
        int* blockdim_rows;
        int* blockdim_cols;
        int* rowPerm;
        int* diagOffset;
        int* col;
        int* resorted_col;
        double* val;
        double* resorted_val;
        int* blockinfo;
        int* tbi;
} JD_OFFDIAGONAL_TYPE;

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
  int nEnts;
  int nRows;
} LMMP_TYPE;


typedef struct {
  int nodes, threads;
  int* on_me;
  int* pseudo_on_me;
  int* wish;
  int* wishlist;
  int* pseudo_nidx;
  int* glob2loc;
  int* loc2glob;
} CR_P_TYPE;

typedef struct {
	int offset, numelements;
        float portion;
} VIP_ENTRY_TYPE;

typedef struct {
        int items;
        VIP_ENTRY_TYPE* entry;
} VIP_TYPE;

typedef struct {
        int row, col, put;
        double val;
} BLOCKENTRY_TYPE;
        
typedef struct {
        int offset, entries;
} TARGET_TYPE;


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




 
/* ########################################################################## */

typedef unsigned long long uint64;

void getMatrixPath(char *given, char *complete);
int isMMfile(const char *filename);

void* allocateMemory( const size_t size, const char* desc );

VECTOR_TYPE* newVector( const int nRows );
#ifdef CUDAKERNEL
void vectorDeviceCopyCheck( VECTOR_TYPE* testvec, int me );
#endif

void permuteVector( double* vec, int* perm, int len);

MM_TYPE* readMMFile( const char* filename, const double epsilon );

CR_TYPE* convertMMToCRMatrix( const MM_TYPE* mm );
JD_TYPE* convertMMToJDMatrix( MM_TYPE* mm, int blocklen );

JD_RESORTED_TYPE* resort_JDS(const JD_TYPE*, const int);
VIP_TYPE* check_divide(MM_TYPE*, float);
JD_OFFDIAGONAL_TYPE* convertMMToODJDMatrix(const MM_TYPE*, const int);

BOOL multiplyMMWithVector( VECTOR_TYPE* res, const MM_TYPE* mm, const VECTOR_TYPE* vec );
BOOL multiplyCRWithVector( VECTOR_TYPE* res, const CR_TYPE* cr, const VECTOR_TYPE* vec );
BOOL multiplyJDWithVector( VECTOR_TYPE* res, const JD_TYPE* jd, const VECTOR_TYPE* vec );

void crColIdToFortran( CR_TYPE* cr );
void crColIdToC( CR_TYPE* cr );

void for_timing_start_asm_(uint64*);
void for_timing_stop_asm_(uint64*, uint64*);
void fortrancrs_(int*, int*, double*, double*, double*, int*, int*);
void fortranjds_(int*, int*, int*, double*, double*, int*, double*, int*, int*, int*);

void freeVector( VECTOR_TYPE* const vec );
void freeMMMatrix( MM_TYPE* const mm );
void freeCRMatrix( CR_TYPE* const cr );
void freeJDMatrix( JD_TYPE* const cr );
void tmpwrite_d(int, int, double*);
void tmpwrite_i(int, int, int*, char*);

void bin_read_cr(CR_TYPE*, const char* );
void bin_write_cr(const CR_TYPE*, const char*);
void bin_read_jd(JD_TYPE*, const int, const char*);
void bin_write_jd(const JD_TYPE*, const char*);

void pio_write_cr_rownumbers(const CR_TYPE*, const char*);
void pio_read_cr_rownumbers(CR_TYPE*, const char*);

LCRP_TYPE* setup_communication(CR_TYPE* const, int, MATRIX_FORMATS);
LCRP_TYPE* setup_communication_parallel(CR_TYPE* const, int, const char* );
LCRP_TYPE* new_pio_read(char*, int);
LCRP_TYPE* parallel_MatRead(char*, int);
LCRP_TYPE* setup_communication_parRead(LMMP_TYPE* const);
void freeLcrpType( LCRP_TYPE* const );

void check_lcrp(int, LCRP_TYPE* const);
void pio_write_cr(const CR_TYPE*, const char*);
void myabort(char*);
void mypabort(char*);
void myaborti(char*, int);
void myabortf(char*, float);
void myaborttf(char*, int, float);
void mypaborts(const char*, const char*);
uint64 cycles4measurement, p_cycles4measurement;
double clockfreq;
double total_mem;
double RecalFrequency(uint64, double);
void freeRevBuf(REVBUF_TYPE*);

REVBUF_TYPE* revolvingBuffer(const uint64, const int, const int);
/* ########################################################################## */
void sweepMemory(int);
float myCpuClockFrequency();
void Correctness_check( VECTOR_TYPE*, LCRP_TYPE*, double* );
unsigned long thishost(char*); 
double my_amount_of_mem(void);
unsigned long machname(char* );
unsigned long kernelversion(char* );
unsigned long modelname(char* );
int get_NUMA_info(int*, int*, int*, int*);
uint64 cachesize(void);

void freeMemory(size_t, const char*, void*);
int compareNZEOrgPos( const void* a, const void* b );
int compareNZEPos(const void*, const void*);
int compareNZEPerRow( const void*, const void*);
int compareNZEForJD( const void*, const void* );

int successful;
int jobmask;
size_t allocatedMem;
MPI_Comm single_node_comm;

#endif /* _MATRICKS_H_ */
