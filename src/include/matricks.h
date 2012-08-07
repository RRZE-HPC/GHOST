#ifndef _MATRICKS_H_
#define _MATRICKS_H_

#include "spmvm_globals.h"

#include <stdio.h>
#include <stdlib.h>
#include <mymacros.h>
#include <mpi.h>

#ifdef OPENCL
#include "my_ellpack.h"
#include <CL/cl.h>
#endif




typedef struct {
	int nRows;
	int* val;
#ifdef OPENCL
  cl_mem CL_val_gpu;
#endif
} INT_VECTOR_TYPE;




typedef struct {
	int row, col, nThEntryInRow;
	real val;
} NZE_TYPE;

typedef struct {
	int nRows, nCols, nEnts;
	NZE_TYPE* nze;
} MM_TYPE;


typedef struct {
	int nRows, nCols, nEnts, nDiags;
	int* rowPerm;
	int* diagOffset;
	int* col;
	real* val;
} JD_TYPE;

typedef struct {
	int row, nEntsInRow;
} JD_SORT_TYPE;


typedef struct {
        int nEnts, nRows, totalblocks;
        int* blockdim_rows;
        int* blockdim_cols;
        int* resorted_col;
        real* resorted_val;
        int* blockinfo;
        int* tbi;
} JD_RESORTED_TYPE;

typedef struct {
	int pagesize, cachesize;
        int vecdim;
        int ppvec, offset, numvecs, globdim;
        real* mem;
        real** vec;
} REVBUF_TYPE;


typedef struct {
        int nEnts, nRows, nDiags, totalblocks;
        int* blockdim_rows;
        int* blockdim_cols;
        int* rowPerm;
        int* diagOffset;
        int* col;
        int* resorted_col;
        real* val;
        real* resorted_val;
        int* blockinfo;
        int* tbi;
} JD_OFFDIAGONAL_TYPE;

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
        real val;
} BLOCKENTRY_TYPE;
        
typedef struct {
        int offset, entries;
} TARGET_TYPE;






 
/* ########################################################################## */

typedef unsigned long long uint64;

void getMatrixPath(char *given, char *path);
int isMMfile(const char *filename);

void* allocateMemory( const size_t size, const char* desc );

void             zeroVector(VECTOR_TYPE *vec);
VECTOR_TYPE*     newVector( const int nRows );
HOSTVECTOR_TYPE* newHostVector( const int nRows, real (*fp)(int));
void             swapVectors(VECTOR_TYPE *v1, VECTOR_TYPE *v2);
void             normalize( real *vec, int nRows);

void permuteVector( real* vec, int* perm, int len);

MM_TYPE* readMMFile( const char* filename );

CR_TYPE* convertMMToCRMatrix( const MM_TYPE* mm );
JD_TYPE* convertMMToJDMatrix( MM_TYPE* mm );

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
void fortrancrs_(int*, int*, real*, real*, real*, int*, int*);
void fortranjds_(int*, int*, int*, real*, real*, int*, real*, int*, int*, int*);

void freeVector( VECTOR_TYPE* const vec );
void freeHostVector( HOSTVECTOR_TYPE* const vec );
void freeMMMatrix( MM_TYPE* const mm );
void freeCRMatrix( CR_TYPE* const cr );
void freeJDMatrix( JD_TYPE* const cr );
void tmpwrite_d(int, int, real*);
void tmpwrite_i(int, int, int*, char*);

void readCRbinFile(CR_TYPE*, const char* );
void readJDbinFile(JD_TYPE*, const int, const char*);

void pio_write_cr_rownumbers(const CR_TYPE*, const char*);
void pio_read_cr_rownumbers(CR_TYPE*, const char*);

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
float myCpuClockFrequency();
int Correctness_check( real*, LCRP_TYPE*, real* );
unsigned long thishost(char*); 
double my_amount_of_mem(void);
unsigned long machname(char* );
unsigned long kernelversion(char* );
unsigned long modelname(char* );
uint64 cachesize(void);

void freeMemory(size_t, const char*, void*);
int compareNZEOrgPos( const void* a, const void* b );
int compareNZEPos(const void*, const void*);
int compareNZEPerRow( const void*, const void*);
int compareNZEForJD( const void*, const void* );

int successful;
size_t allocatedMem;
MPI_Comm single_node_comm;

#endif /* _MATRICKS_H_ */
